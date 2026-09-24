"""M5: the sentiment + aspect processing pass and the endpoints that read its results.

The ML functions are replaced with keyword fakes, so these tests need neither the trained
sentiment artifact nor the sentence-transformer download.
"""

from datetime import UTC, datetime, timedelta

import httpx
import pytest
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.games.constants import AspectStatus
from src.games.models import Game
from src.ml import aspects, sentiment
from src.reviews import constants, service
from src.reviews.constants import Skipped
from src.reviews.models import Review, ReviewAspect

APPID = 1145360
TEXTS = {
    1: "It crashes a lot. Another crash today. Price is fair.",  # negative: bugs x2, price
    2: "Price is great. Love it.",  # positive: price
    3: "Just vibes. Love it.",  # positive: every unit "none"
    4: "Bad fps. Bad overall.",  # negative: performance
}
KEYWORDS = {"crash": "bugs", "price": "price", "fps": "performance"}


@pytest.fixture
def ml_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def predict(review_text: str) -> dict:
        calls.append(review_text)
        negative = "crash" in review_text or "Bad" in review_text
        return {"label": "negative" if negative else "positive", "confidence": 0.9}

    def assign_aspects(review_text: str) -> list[dict]:
        if review_text.startswith("[img]"):
            return []  # like the real splitter on BBCode-only text
        units = review_text.split(". ")
        tag = lambda unit: next((a for k, a in KEYWORDS.items() if k in unit.lower()), "none")
        return [{"sentence": u, "aspect": tag(u), "similarity": 0.5} for u in units]

    monkeypatch.setattr(sentiment, "predict", predict)
    monkeypatch.setattr(aspects, "assign_aspects", assign_aspects)
    monkeypatch.setattr(sentiment, "ensure_loaded", dict)
    return calls


@pytest.fixture
async def seeded(engine) -> None:
    """An ingested (not processed) game with TEXTS as reviews; review 4 is the newest."""
    base = datetime(2026, 9, 1, tzinfo=UTC)
    async with AsyncSession(engine) as session:
        session.add(Game(appid=APPID, name="Hades", last_ingested_at=base))
        await session.flush()
        session.add_all(
            Review(
                id=rid, appid=APPID, review_text=t, voted_up=True, votes_up=0,
                playtime_forever=0, language="english", created_at=base + timedelta(days=rid),
            )
            for rid, t in TEXTS.items()
        )
        await session.commit()


async def add_review(engine, rid: int, review_text: str) -> None:
    async with AsyncSession(engine) as session:
        session.add(Review(
            id=rid, appid=APPID, review_text=review_text, voted_up=True, votes_up=0,
            playtime_forever=0, language="english", created_at=datetime(2026, 8, 1, tzinfo=UTC),
        ))
        await session.commit()


async def set_status(engine, status: AspectStatus) -> None:
    async with engine.begin() as conn:
        await conn.execute(update(Game).values(aspects_status=status))


async def game_status(engine) -> tuple[str, str | None]:
    async with engine.connect() as conn:
        return (await conn.execute(select(Game.aspects_status, Game.aspects_error))).one()


async def processed_at(engine) -> datetime | None:
    async with engine.connect() as conn:
        return await conn.scalar(select(Game.aspects_processed_at))


async def process(engine, force: bool = False) -> int | Skipped:
    async with AsyncSession(engine, expire_on_commit=False) as session:
        return await service.process_reviews(session, APPID, force=force)


async def unit_count(engine) -> int:
    async with engine.connect() as conn:
        return await conn.scalar(select(func.count()).select_from(ReviewAspect))


async def test_process_stores_units_with_review_sentiment(engine, seeded, ml_calls):
    assert await process(engine) == 4

    async with engine.connect() as conn:
        rows = (await conn.execute(select(ReviewAspect).order_by(ReviewAspect.id))).all()
        sentiments = dict((await conn.execute(select(Review.id, Review.predicted_sentiment))).all())
    assert await game_status(engine) == (AspectStatus.DONE, None)
    assert await processed_at(engine) is not None
    assert len(rows) == 9
    assert [r.aspect_label for r in rows if r.review_id == 1] == ["bugs", "bugs", "price"]
    assert {r.aspect_label for r in rows if r.review_id == 3} == {"none"}
    assert sentiments == {1: "negative", 2: "positive", 3: "positive", 4: "negative"}


async def test_rerun_is_a_noop_and_force_replaces(engine, seeded, ml_calls):
    await process(engine)
    calls_after_first = len(ml_calls)

    assert await process(engine) == Skipped.ALREADY_DONE
    assert len(ml_calls) == calls_after_first
    assert await unit_count(engine) == 9

    assert await process(engine, force=True) == 4
    assert await unit_count(engine) == 9


async def test_process_skips_a_game_another_worker_claimed(engine, seeded, ml_calls):
    await set_status(engine, AspectStatus.PROCESSING)
    assert await process(engine) == Skipped.IN_PROGRESS
    assert ml_calls == []
    assert await unit_count(engine) == 0

    assert await process(engine, force=True) == 4  # takes over a worker that died
    assert (await game_status(engine))[0] == AspectStatus.DONE


async def test_process_skips_unknown_app(engine, ml_calls):
    async with AsyncSession(engine) as session:
        assert await service.process_reviews(session, APPID) == Skipped.NOT_INGESTED


async def test_failure_is_recorded_and_needs_force(engine, seeded, ml_calls, monkeypatch):
    real_predict = sentiment.predict

    def broken(review_text: str) -> dict:
        raise RuntimeError("model exploded")

    monkeypatch.setattr(sentiment, "predict", broken)
    with pytest.raises(RuntimeError):
        await process(engine)
    assert await game_status(engine) == (AspectStatus.FAILED, "RuntimeError: model exploded")
    assert await unit_count(engine) == 0

    monkeypatch.setattr(sentiment, "predict", real_predict)
    assert await process(engine) == Skipped.FAILED  # no silent retry
    assert await process(engine, force=True) == 4
    assert await game_status(engine) == (AspectStatus.DONE, None)


async def test_aspects_unprocessed_returns_202_then_counts(client, engine, seeded, ml_calls):
    first = await client.get(f"/games/{APPID}/aspects")
    assert first.status_code == 202
    assert first.json()["status"] == "processing"
    assert "aspects" not in first.json()

    # The background task ran after the 202 was sent; the next call reads stored results.
    second = await client.get(f"/games/{APPID}/aspects")
    assert second.status_code == 200
    body = second.json()
    assert body["status"] == "ready"
    assert body["overall"] == {"positive": 2, "negative": 2, "total": 4, "positive_pct": 50.0}
    by_aspect = {a["aspect"]: a for a in body["aspects"]}
    assert set(by_aspect) == {"performance", "price", "bugs", "story", "gameplay"}
    # Review 1 has two bugs units but counts once.
    assert (by_aspect["bugs"]["positive"], by_aspect["bugs"]["negative"]) == (0, 1)
    assert (by_aspect["price"]["positive"], by_aspect["price"]["negative"]) == (1, 1)
    assert by_aspect["story"] == {
        "aspect": "story", "positive": 0, "negative": 0, "total": 0, "positive_pct": None
    }

    async with engine.connect() as conn:
        direct = await conn.scalar(
            select(func.count(ReviewAspect.review_id.distinct()))
            .join(Review, Review.id == ReviewAspect.review_id)
            .where(Review.appid == APPID, ReviewAspect.aspect_label == "price",
                   Review.predicted_sentiment == "negative")
        )
    assert by_aspect["price"]["negative"] == direct

    assert len(ml_calls) == 4  # processed once, not per request
    await client.get(f"/games/{APPID}/aspects")
    assert len(ml_calls) == 4


async def test_failed_forced_rerun_keeps_last_success_time(engine, seeded, ml_calls, monkeypatch):
    await process(engine)
    finished = await processed_at(engine)

    def broken(review_text: str) -> dict:
        raise RuntimeError("model exploded")

    monkeypatch.setattr(sentiment, "predict", broken)
    with pytest.raises(RuntimeError):
        await process(engine, force=True)
    assert (await game_status(engine))[0] == AspectStatus.FAILED
    assert await processed_at(engine) == finished


async def test_reviews_filter_by_aspect_and_sentiment(client, engine, seeded, ml_calls):
    await process(engine)

    resp = await client.get(f"/games/{APPID}/reviews", params={"aspect": "bugs", "sentiment": "negative"})
    assert resp.status_code == 200
    assert [r["id"] for r in resp.json()["items"]] == [1]

    price = (await client.get(f"/games/{APPID}/reviews", params={"aspect": "price"})).json()
    assert [r["id"] for r in price["items"]] == [2, 1]

    empty = await client.get(f"/games/{APPID}/reviews", params={"aspect": "bugs", "sentiment": "positive"})
    assert empty.status_code == 200
    assert empty.json()["total"] == 0 and empty.json()["items"] == []


async def test_reviews_filter_on_unprocessed_game_returns_202(client, seeded, ml_calls):
    resp = await client.get(f"/games/{APPID}/reviews", params={"sentiment": "negative"})
    assert resp.status_code == 202
    assert resp.json()["status"] == "processing"

    unfiltered = await client.get(f"/games/{APPID}/reviews")
    assert unfiltered.status_code == 200
    assert unfiltered.json()["total"] == 4


async def test_background_failure_answers_failed_not_processing(
    client, engine, seeded, ml_calls, monkeypatch
):
    def broken(review_text: str) -> dict:
        raise RuntimeError("model exploded")

    monkeypatch.setattr(sentiment, "predict", broken)
    assert (await client.get(f"/games/{APPID}/aspects")).status_code == 202

    for resp in (
        await client.get(f"/games/{APPID}/aspects"),
        await client.get(f"/games/{APPID}/reviews", params={"aspect": "bugs"}),
    ):
        assert resp.status_code == 500
        assert resp.json()["status"] == "failed"
        assert "--force" in resp.json()["detail"]
    assert (await game_status(engine))[0] == AspectStatus.FAILED


async def test_game_being_processed_answers_202_without_a_new_run(client, engine, seeded, ml_calls):
    await set_status(engine, AspectStatus.PROCESSING)
    resp = await client.get(f"/games/{APPID}/aspects")
    assert resp.status_code == 202
    assert ml_calls == []
    assert (await game_status(engine))[0] == AspectStatus.PROCESSING


async def test_missing_sentiment_model_answers_503(client, engine, seeded, monkeypatch, tmp_path):
    real_load = sentiment.load_model
    monkeypatch.setattr(sentiment, "_artifact", None)
    monkeypatch.setattr(sentiment, "load_model", lambda: real_load(tmp_path / "missing.joblib"))

    resp = await client.get(f"/games/{APPID}/aspects")
    assert resp.status_code == 503
    assert "train_sentiment.py" in resp.json()["detail"]
    assert await game_status(engine) == (AspectStatus.PENDING, None)
    assert (await client.get(f"/games/{APPID}/reviews")).status_code == 200  # needs no model


async def test_review_without_units_still_has_sentiment(client, engine, seeded, ml_calls):
    await add_review(engine, 5, "[img]screenshot.png[/img]")
    await process(engine)

    positive = (await client.get(f"/games/{APPID}/reviews", params={"sentiment": "positive"})).json()
    negative = (await client.get(f"/games/{APPID}/reviews", params={"sentiment": "negative"})).json()
    assert 5 in [r["id"] for r in positive["items"]]
    assert positive["total"] + negative["total"] == 5

    overall = (await client.get(f"/games/{APPID}/aspects")).json()["overall"]
    assert overall["total"] == 5


async def test_reviews_rejects_unknown_filter_values(client, seeded):
    assert (await client.get(f"/games/{APPID}/reviews", params={"aspect": "none"})).status_code == 422
    assert (await client.get(f"/games/{APPID}/reviews", params={"sentiment": "meh"})).status_code == 422


async def test_filtered_pagination_boundaries(client, engine, seeded, ml_calls):
    await process(engine)
    url = f"/games/{APPID}/reviews"

    last = (await client.get(url, params={"sentiment": "negative", "limit": 1, "offset": 1})).json()
    assert last["total"] == 2
    assert [r["id"] for r in last["items"]] == [1]

    beyond = await client.get(url, params={"sentiment": "negative", "limit": 1, "offset": 5})
    assert beyond.status_code == 200
    assert beyond.json()["total"] == 2 and beyond.json()["items"] == []


async def test_aspects_unknown_app_matches_reviews_404(client, steam):
    steam.get(constants.APPDETAILS_URL).mock(
        return_value=httpx.Response(200, json={"999": {"success": False}})
    )
    aspects_resp = await client.get("/games/999/aspects")
    reviews_resp = await client.get("/games/999/reviews")
    assert aspects_resp.status_code == reviews_resp.status_code == 404
    assert aspects_resp.json() == reviews_resp.json() == {"detail": "Steam has no app with appid 999"}
