import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

import httpx
from sqlalchemy import delete, exists, func, select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from starlette.background import BackgroundTask
from starlette.responses import JSONResponse

from src.config import settings
from src.games import service as games_service
from src.games.constants import AspectStatus
from src.games.models import Game
from src.ml import aspects, sentiment
from src.ml.anchors import ASPECTS
from src.ml.constants import NEGATIVE, POSITIVE
from src.reviews import constants, ingestion
from src.reviews.exceptions import InsufficientReviews
from src.reviews.models import Review, ReviewAspect

logger = logging.getLogger(__name__)


async def ensure_ingested(session: AsyncSession, http: httpx.AsyncClient, appid: int) -> None:
    """Fetch and cache the app's reviews unless that already happened.

    Concurrent first requests for one appid serialize on a transaction-scoped advisory
    lock, so Steam is hit once and the loser finds the game already ingested. Nothing is
    persisted (and the game is not marked ingested) when the min-review guard fails.
    """
    if games_service.is_ingested(await games_service.get_game(session, appid)):
        return

    await session.execute(
        text("SELECT pg_advisory_xact_lock(:ns, :appid)"),
        {"ns": constants.INGEST_LOCK_NAMESPACE, "appid": appid},
    )
    if games_service.is_ingested(await games_service.get_game(session, appid)):
        return

    logger.info("Ingesting reviews for appid %d", appid)
    name = await ingestion.fetch_app_name(http, appid)
    reviews = await ingestion.fetch_reviews(
        http, appid, settings.max_reviews, settings.steam_request_delay_s
    )
    if len(reviews) < settings.min_review_count:
        raise InsufficientReviews(appid, len(reviews), settings.min_review_count)

    await games_service.save_game(session, appid, name, datetime.now(UTC))
    for i in range(0, len(reviews), constants.INSERT_CHUNK_SIZE):
        chunk = [{**r, "appid": appid} for r in reviews[i : i + constants.INSERT_CHUNK_SIZE]]
        await session.execute(insert(Review).values(chunk).on_conflict_do_nothing())
    await session.commit()
    logger.info("Ingested %d reviews for appid %d", len(reviews), appid)


async def list_reviews(
    session: AsyncSession,
    appid: int,
    limit: int,
    offset: int,
    aspect: str | None = None,
    sentiment_label: str | None = None,
) -> tuple[int, list[Review]]:
    """A page of the app's reviews, newest first.

    `aspect` keeps reviews with at least one unit tagged with it; `sentiment_label` keeps
    reviews predicted that way. Both need the app to be processed (`process_reviews`).
    """
    where = [Review.appid == appid]
    if aspect is not None:
        unit = select(ReviewAspect.id).where(
            ReviewAspect.review_id == Review.id, ReviewAspect.aspect_label == aspect
        )
        where.append(exists(unit))
    if sentiment_label is not None:
        where.append(Review.predicted_sentiment == sentiment_label)

    total = await session.scalar(select(func.count()).select_from(Review).where(*where))
    rows = await session.scalars(
        select(Review)
        .where(*where)
        .order_by(Review.created_at.desc(), Review.id.desc())
        .limit(limit)
        .offset(offset)
    )
    return total or 0, list(rows)


def _analyze(reviews: list[tuple[int, str]]) -> tuple[dict[int, str], list[dict[str, Any]]]:
    """Sentiment for each whole review, and each review's aspect-tagged units."""
    # ponytail: one predict/encode call per review; batching across reviews would be several
    # times faster (known limitation, see DECISIONS.md).
    labels: dict[int, str] = {}
    units: list[dict[str, Any]] = []
    for review_id, review_text in reviews:
        labels[review_id] = sentiment.predict(review_text)["label"]
        units.extend(
            {
                "review_id": review_id,
                "sentence_text": unit["sentence"],
                "aspect_label": unit["aspect"],
                "similarity_score": unit["similarity"],
            }
            for unit in aspects.assign_aspects(review_text)
        )
    return labels, units


def _skip_reason(game: Game | None) -> constants.Skipped:
    if not games_service.is_ingested(game):
        return constants.Skipped.NOT_INGESTED
    return {
        AspectStatus.DONE: constants.Skipped.ALREADY_DONE,
        AspectStatus.PROCESSING: constants.Skipped.IN_PROGRESS,
        AspectStatus.FAILED: constants.Skipped.FAILED,
    }[game.aspects_status]


async def process_reviews(
    session: AsyncSession, appid: int, force: bool = False
) -> int | constants.Skipped:
    """Run sentiment + aspect tagging over the app's cached reviews and store the results.

    Returns the number of reviews processed, or why nothing was done. Only a "pending" game
    is claimed; `force` also claims done, failed and stuck "processing" ones (after tuning
    anchors, or when a worker died mid-run).

    Three short transactions, so no connection is held during the 20-60s of CPU work:
    claim (status -> processing), read the texts, then write every result together with
    status -> done. A failure sets status -> failed with the error and re-raises; callers
    log it. A failed game stays failed until a `force` run.
    """
    claimable = list(AspectStatus) if force else [AspectStatus.PENDING]
    claimed = await session.scalar(
        update(Game)
        .where(
            Game.appid == appid,
            Game.last_ingested_at.is_not(None),
            Game.aspects_status.in_(claimable),
        )
        .values(aspects_status=AspectStatus.PROCESSING, aspects_error=None)
        .returning(Game.appid)
    )
    if claimed is None:
        reason = _skip_reason(await games_service.get_game(session, appid))
        await session.rollback()
        return reason
    await session.commit()

    try:
        result = await session.execute(
            select(Review.id, Review.review_text).where(Review.appid == appid)
        )
        reviews = list(result.tuples())
        await session.commit()  # hand the connection back before the CPU work
        logger.info("Processing %d reviews for appid %d", len(reviews), appid)
        labels, units = await asyncio.to_thread(_analyze, reviews)  # keep the event loop free

        # The row lock orders this write after any concurrent --force run's write, so the
        # delete below sees (and replaces) its rows instead of doubling them.
        await session.execute(select(Game.appid).where(Game.appid == appid).with_for_update())
        app_reviews = select(Review.id).where(Review.appid == appid)
        await session.execute(delete(ReviewAspect).where(ReviewAspect.review_id.in_(app_reviews)))
        if labels:
            await session.execute(
                update(Review),
                [{"id": rid, "predicted_sentiment": label} for rid, label in labels.items()],
            )
        for i in range(0, len(units), constants.INSERT_CHUNK_SIZE):
            chunk = units[i : i + constants.INSERT_CHUNK_SIZE]
            await session.execute(insert(ReviewAspect).values(chunk))
        await session.execute(
            update(Game)
            .where(Game.appid == appid)
            .values(aspects_status=AspectStatus.DONE, aspects_processed_at=datetime.now(UTC))
        )
        await session.commit()
    except Exception as exc:
        await session.rollback()
        await session.execute(
            update(Game)
            .where(Game.appid == appid)
            .values(aspects_status=AspectStatus.FAILED, aspects_error=f"{type(exc).__name__}: {exc}")
        )
        await session.commit()
        raise
    logger.info("Stored %d aspect units for appid %d", len(units), appid)
    return len(reviews)


async def process_in_background(engine: AsyncEngine, appid: int) -> None:
    """Runs after the 202 is sent, in its own session: the request's is closed by then."""
    try:
        async with AsyncSession(engine, expire_on_commit=False) as session:
            await process_reviews(session, appid)
    except Exception:
        # Already recorded as failed on the game (unless the DB itself is what failed).
        logger.exception("Processing reviews for appid %d failed", appid)


def unprocessed_response(engine: AsyncEngine, game: Game) -> JSONResponse:
    """The answer for an app whose results aren't ready (status other than done).

    failed -> 500 with status "failed"; pending -> 202 and processing starts after sending,
    or 503 if no sentiment model is trained yet; processing -> 202, nothing started.
    Returned rather than raised: an exception handler's response would drop the task.
    """
    appid = game.appid
    if game.aspects_status == AspectStatus.FAILED:
        return JSONResponse(
            status_code=500,
            content={
                "appid": appid,
                "status": "failed",
                "detail": f"Analyzing reviews for app {appid} failed; "
                f"retry with scripts/process_reviews.py --force {appid}",
            },
        )
    task = None
    if game.aspects_status == AspectStatus.PENDING:
        sentiment.ensure_loaded()  # 503 now beats a run that is bound to fail
        task = BackgroundTask(process_in_background, engine, appid)
    return JSONResponse(
        status_code=202,
        content={
            "appid": appid,
            "status": "processing",
            "detail": f"Reviews for app {appid} are being analyzed; retry shortly",
        },
        background=task,
    )


async def aspect_summary(session: AsyncSession, appid: int) -> dict[str, Any]:
    """Positive/negative counts per aspect for a processed app.

    Counts are distinct reviews, not units: sentiment is predicted per review, so a review
    with three bugs sentences counts once under bugs. `overall` counts every review, also
    those whose units are all "none" or that have no units at all.
    """
    # ponytail: two queries; one GROUP BY GROUPING SETS would do (known limitation, see
    # DECISIONS.md).
    rows = await session.execute(
        select(
            ReviewAspect.aspect_label,
            Review.predicted_sentiment,
            func.count(ReviewAspect.review_id.distinct()),
        )
        .join(Review, Review.id == ReviewAspect.review_id)
        .where(Review.appid == appid)
        .group_by(ReviewAspect.aspect_label, Review.predicted_sentiment)
    )
    counts = {(label, sent): n for label, sent, n in rows}
    overall = await session.execute(
        select(Review.predicted_sentiment, func.count())
        .where(Review.appid == appid)
        .group_by(Review.predicted_sentiment)
    )
    overall_counts = dict(overall.all())

    def entry(positive: int, negative: int) -> dict[str, Any]:
        total = positive + negative
        return {
            "positive": positive,
            "negative": negative,
            "total": total,
            "positive_pct": round(100 * positive / total, 1) if total else None,
        }

    return {
        "appid": appid,
        "status": "ready",
        "overall": entry(overall_counts.get(POSITIVE, 0), overall_counts.get(NEGATIVE, 0)),
        "aspects": [
            {
                "aspect": aspect,
                **entry(counts.get((aspect, POSITIVE), 0), counts.get((aspect, NEGATIVE), 0)),
            }
            for aspect in ASPECTS  # "none" is not one of them
        ],
    }
