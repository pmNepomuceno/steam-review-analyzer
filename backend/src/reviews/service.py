import logging
from datetime import UTC, datetime

import httpx
from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.games import service as games_service
from src.reviews import constants, ingestion
from src.reviews.exceptions import InsufficientReviews
from src.reviews.models import Review

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
    session: AsyncSession, appid: int, limit: int, offset: int
) -> tuple[int, list[Review]]:
    total = await session.scalar(select(func.count()).select_from(Review).where(Review.appid == appid))
    rows = await session.scalars(
        select(Review)
        .where(Review.appid == appid)
        .order_by(Review.created_at.desc(), Review.id.desc())
        .limit(limit)
        .offset(offset)
    )
    return total or 0, list(rows)
