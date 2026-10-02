"""M8: the homepage's game list and the Steam store search proxy."""

import asyncio
from datetime import UTC, datetime

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.games import service
from src.games.constants import AspectStatus
from src.games.models import Game
from src.reviews import constants
from src.reviews.models import Review

DONE_AT = datetime(2026, 10, 1, 12, tzinfo=UTC)
SEARCH = {
    "total": 3,
    "items": [
        {"type": "app", "name": "Hades", "id": 1145360, "tiny_image": "https://img/hades.jpg"},
        {"type": "sub", "name": "Hades Bundle", "id": 5},  # not an app
        {"type": "app", "name": "Hades II", "id": 1145350},
    ],
}


@pytest.fixture(autouse=True)
def _empty_search_cache() -> None:
    service._search_cache.clear()


async def test_lists_only_done_games_with_counts(client, engine):
    async with AsyncSession(engine) as session:
        session.add_all([
            Game(appid=2, name="Beta", aspects_status=AspectStatus.DONE,
                 aspects_processed_at=DONE_AT, last_ingested_at=DONE_AT),
            Game(appid=1, name="Alpha", aspects_status=AspectStatus.DONE,
                 aspects_processed_at=DONE_AT, last_ingested_at=DONE_AT),
            Game(appid=3, name="Pending", last_ingested_at=DONE_AT),
        ])
        await session.flush()
        session.add_all(
            Review(id=i, appid=1, review_text="x", voted_up=True, votes_up=0, playtime_forever=0,
                   language="english", created_at=DONE_AT)
            for i in range(3)
        )
        await session.commit()

    resp = await client.get("/games")
    assert resp.status_code == 200
    assert resp.json() == [
        {"appid": 1, "name": "Alpha", "review_count": 3, "analyzed_at": "2026-10-01T12:00:00Z"},
        {"appid": 2, "name": "Beta", "review_count": 0, "analyzed_at": "2026-10-01T12:00:00Z"},
    ]


async def test_search_maps_apps_and_caches_per_term(client, steam):
    route = steam.get(constants.STORESEARCH_URL).mock(
        return_value=httpx.Response(200, json=SEARCH)
    )
    first = await client.get("/steam/search", params={"q": "Hades"})
    assert first.status_code == 200
    assert first.json() == [
        {"appid": 1145360, "name": "Hades", "image": "https://img/hades.jpg"},
        {"appid": 1145350, "name": "Hades II", "image": None},
    ]
    assert route.calls.last.request.url.params["term"] == "hades"
    assert (await client.get("/steam/search", params={"q": " hades "})).json() == first.json()
    assert route.call_count == 1  # same term within the TTL: served from the cache

    await client.get("/steam/search", params={"q": "portal"})
    assert route.call_count == 2


async def test_search_rejects_short_terms_and_reports_steam_failure(client, steam):
    assert (await client.get("/steam/search", params={"q": "h"})).status_code == 422
    steam.get(constants.STORESEARCH_URL).mock(return_value=httpx.Response(503))
    resp = await client.get("/steam/search", params={"q": "hades"})
    assert resp.status_code == 502
    assert service._search_cache == {}  # a failure is not cached


async def test_a_hung_search_does_not_hold_up_other_terms(client, steam):
    release = asyncio.Event()

    async def answer(request: httpx.Request) -> httpx.Response:
        if request.url.params["term"] == "slow":
            await release.wait()  # Steam hanging on this one term
        return httpx.Response(200, json=SEARCH)

    steam.get(constants.STORESEARCH_URL).mock(side_effect=answer)
    slow = asyncio.create_task(client.get("/steam/search", params={"q": "slow"}))
    await asyncio.sleep(0.05)
    fast = await asyncio.wait_for(client.get("/steam/search", params={"q": "fast"}), 2)
    assert fast.status_code == 200
    assert not slow.done()
    release.set()
    assert (await slow).status_code == 200
