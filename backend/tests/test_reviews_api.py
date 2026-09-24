import asyncio

import httpx
from sqlalchemy import func, select

from src.games.models import Game
from src.reviews import constants
from src.reviews.models import Review
from tests.steam_fakes import mock_steam

APPID = 1145360


async def test_first_request_ingests_then_serves_from_cache(client, engine, steam):
    details, reviews_route = mock_steam(steam, APPID, 250)

    first = await client.get(f"/games/{APPID}/reviews", params={"limit": 3})
    assert first.status_code == 200
    body = first.json()
    assert body["total"] == 250
    assert len(body["items"]) == 3
    calls_after_ingest = (details.call_count, reviews_route.call_count)
    assert details.call_count == 1

    second = await client.get(f"/games/{APPID}/reviews", params={"limit": 3})
    assert second.status_code == 200
    assert second.json() == body
    assert (details.call_count, reviews_route.call_count) == calls_after_ingest

    async with engine.connect() as conn:
        game = (await conn.execute(select(Game))).one()
    assert game.name == "Test Game"
    assert game.last_ingested_at is not None


async def test_pagination_offset(client, steam):
    mock_steam(steam, APPID, 250)
    page1 = (await client.get(f"/games/{APPID}/reviews", params={"limit": 5})).json()
    page2 = (await client.get(f"/games/{APPID}/reviews", params={"limit": 5, "offset": 5})).json()
    ids1 = {r["id"] for r in page1["items"]}
    ids2 = {r["id"] for r in page2["items"]}
    assert len(ids1) == len(ids2) == 5
    assert ids1.isdisjoint(ids2)


async def test_too_few_reviews_returns_422_and_persists_nothing(client, engine, steam):
    mock_steam(steam, APPID, 50)

    resp = await client.get(f"/games/{APPID}/reviews")

    assert resp.status_code == 422
    assert "50" in resp.json()["detail"]
    async with engine.connect() as conn:
        assert await conn.scalar(select(func.count()).select_from(Game)) == 0
        assert await conn.scalar(select(func.count()).select_from(Review)) == 0


async def test_unknown_app_returns_404(client, steam):
    steam.get(constants.APPDETAILS_URL).mock(
        return_value=httpx.Response(200, json={"999": {"success": False}})
    )
    resp = await client.get("/games/999/reviews")
    assert resp.status_code == 404


async def test_steam_down_returns_502(client, steam):
    steam.get(constants.APPDETAILS_URL).mock(return_value=httpx.Response(500))
    resp = await client.get(f"/games/{APPID}/reviews")
    assert resp.status_code == 502


async def test_invalid_appid_is_rejected(client):
    assert (await client.get("/games/0/reviews")).status_code == 422
    assert (await client.get("/games/abc/reviews")).status_code == 422


async def test_concurrent_first_requests_ingest_once(client, engine, steam):
    details, reviews_route = mock_steam(steam, APPID, 250)

    responses = await asyncio.gather(
        *(client.get(f"/games/{APPID}/reviews", params={"limit": 1}) for _ in range(3))
    )

    assert [r.status_code for r in responses] == [200, 200, 200]
    assert details.call_count == 1
    assert reviews_route.call_count == 4  # one ingest: pages at 0, 100, 200, then 250 (empty)
    async with engine.connect() as conn:
        assert await conn.scalar(select(func.count()).select_from(Review)) == 250
