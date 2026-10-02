import asyncio
import time
from itertools import pairwise

import httpx
import pytest

from src.config import settings
from src.games.exceptions import GameNotFound
from src.reviews import constants, ingestion
from src.reviews.exceptions import SteamUnavailable
from tests.steam_fakes import make_reviews, mock_steam


def test_parse_review_maps_fields():
    raw = make_reviews(1)[0]
    parsed = ingestion.parse_review(raw)
    assert parsed["id"] == 1000
    assert parsed["voted_up"] is False
    assert parsed["playtime_forever"] == 0
    assert parsed["created_at"].tzinfo is not None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(review="   "),
        lambda r: r.pop("recommendationid"),
        lambda r: r.pop("voted_up"),
        lambda r: r.update(timestamp_created="nope"),
    ],
)
def test_parse_review_drops_unusable(mutate):
    raw = make_reviews(1)[0]
    mutate(raw)
    assert ingestion.parse_review(raw) is None


async def test_fetch_reviews_paginates_and_caps(steam):
    _, reviews_route = mock_steam(steam, 1, 250)
    async with httpx.AsyncClient() as http:
        everything = await ingestion.fetch_reviews(http, 1, max_reviews=1000)
        capped = await ingestion.fetch_reviews(http, 1, max_reviews=120)
    assert len(everything) == 250
    assert len(capped) == 120
    assert reviews_route.called


async def test_fetch_reviews_dedupes_and_stops_on_repeated_cursor(steam):
    page = {"success": 1, "reviews": make_reviews(5), "cursor": "same"}
    route = steam.get(url__regex=r".*appreviews.*").mock(
        return_value=httpx.Response(200, json=page)
    )
    async with httpx.AsyncClient() as http:
        result = await ingestion.fetch_reviews(http, 1, max_reviews=1000)
    assert len(result) == 5
    assert route.call_count == 2  # "*" then "same"; the repeated cursor ends the loop


async def test_fetch_app_name_unknown_app(steam):
    steam.get(ingestion.constants.APPDETAILS_URL).mock(
        return_value=httpx.Response(200, json={"7": {"success": False}})
    )
    async with httpx.AsyncClient() as http:
        with pytest.raises(GameNotFound):
            await ingestion.fetch_app_name(http, 7)


async def test_fetch_app_name_entry_keyed_by_other_id(steam):
    # Real response shape seen 2026-09-27 for Hades: keyed by 1206340, not the requested appid.
    steam.get(ingestion.constants.APPDETAILS_URL).mock(
        return_value=httpx.Response(
            200, json={"1206340": {"success": True, "data": {"name": "Hades", "steam_appid": 1145360}}}
        )
    )
    async with httpx.AsyncClient() as http:
        assert await ingestion.fetch_app_name(http, 1145360) == "Hades"


async def test_fetch_app_name_rejects_entry_for_another_app(steam):
    steam.get(ingestion.constants.APPDETAILS_URL).mock(
        return_value=httpx.Response(
            200, json={"999": {"success": True, "data": {"name": "Parent Game", "steam_appid": 999}}}
        )
    )
    async with httpx.AsyncClient() as http:
        with pytest.raises(GameNotFound):
            await ingestion.fetch_app_name(http, 7)


async def test_retries_then_gives_up_on_429(steam):
    route = steam.get(ingestion.constants.APPDETAILS_URL).mock(return_value=httpx.Response(429))
    async with httpx.AsyncClient() as http:
        with pytest.raises(SteamUnavailable):
            await ingestion.fetch_app_name(http, 7)
    assert route.call_count == ingestion.constants.RETRY_ATTEMPTS


async def test_every_steam_call_shares_one_pace(steam, monkeypatch):
    # Ingestion pages, store search and the rating fetch, all at once: their calls start at
    # least steam_request_delay_s apart, whichever code path made them.
    delay = 0.1
    monkeypatch.setattr(settings, "steam_request_delay_s", delay)
    mock_steam(steam, 1, 250)  # 3 review pages, plus an empty one that ends the cursor
    steam.get(constants.STORESEARCH_URL).mock(return_value=httpx.Response(200, json={"items": []}))
    started: list[tuple[float, str]] = []

    async def record(request: httpx.Request) -> None:
        started.append((time.monotonic(), request.url.path))

    async with httpx.AsyncClient(event_hooks={"request": [record]}) as http:
        await asyncio.gather(
            ingestion.fetch_reviews(http, 1, max_reviews=1000),
            ingestion.search_store(http, "hades"),
            ingestion.fetch_review_summary(http, 1),
            ingestion.search_store(http, "portal"),
        )
    times = sorted(t for t, _ in started)
    assert len(times) == 7
    # Paced per code path, a search would start right after the first page (gap near 0).
    assert min(b - a for a, b in pairwise(times)) >= delay * 0.9  # timer slack
