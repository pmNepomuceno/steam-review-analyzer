import time
from datetime import datetime
from typing import Any

import httpx
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.games import constants
from src.games.constants import AspectStatus
from src.games.models import Game
from src.games.schemas import GameListItem
from src.reviews import ingestion
from src.reviews.models import Review

# /steam/search answers per search term. The calls behind them are paced with every other
# Steam call (`ingestion._get_json`).
_search_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}  # term -> (expiry, results)


async def get_game(session: AsyncSession, appid: int) -> Game | None:
    # populate_existing: callers re-read after waiting on the ingest lock and must
    # see the row another request committed, not a stale identity-map copy.
    stmt = select(Game).where(Game.appid == appid).execution_options(populate_existing=True)
    return await session.scalar(stmt)


async def list_processed(session: AsyncSession) -> list[GameListItem]:
    """Every game whose analysis is done, by name, with its cached review count."""
    review_count = (
        select(func.count()).where(Review.appid == Game.appid).correlate(Game).scalar_subquery()
    )
    result = await session.execute(
        select(Game.appid, Game.name, review_count, Game.aspects_processed_at)
        .where(Game.aspects_status == AspectStatus.DONE)
        .order_by(Game.name)
    )
    return [
        GameListItem(appid=appid, name=name, review_count=count, analyzed_at=analyzed_at)
        for appid, name, count, analyzed_at in result
    ]


def _cached_search(term: str) -> list[dict[str, Any]] | None:
    hit = _search_cache.get(term)
    return hit[1] if hit is not None and hit[0] > time.monotonic() else None


async def search_steam(http: httpx.AsyncClient, term: str) -> list[dict[str, Any]]:
    """Steam store search for `term`, cached for SEARCH_CACHE_TTL_S per (case-folded) term.

    Nothing is held during the call, so a search that hangs on Steam (up to the client's
    timeout and retries) doesn't hold up other visitors' searches. Raises SteamUnavailable
    like ingestion does (502).
    """
    term = term.strip().casefold()
    if not term:
        return []
    if (hit := _cached_search(term)) is not None:
        return hit
    # ponytail: identical searches in flight at once each call Steam (paced); share one call
    # per term if that shows up in the logs.
    results = await ingestion.search_store(http, term)
    # ponytail: dropped wholesale when full rather than LRU; fine for a demo's traffic.
    if len(_search_cache) >= constants.SEARCH_CACHE_MAX_TERMS:
        _search_cache.clear()
    _search_cache[term] = (time.monotonic() + constants.SEARCH_CACHE_TTL_S, results)
    return results


def is_ingested(game: Game | None) -> bool:
    return game is not None and game.last_ingested_at is not None


async def save_game(
    session: AsyncSession, appid: int, name: str, last_ingested_at: datetime
) -> None:
    stmt = insert(Game).values(appid=appid, name=name, last_ingested_at=last_ingested_at)
    stmt = stmt.on_conflict_do_update(
        index_elements=[Game.appid],
        set_={"name": stmt.excluded.name, "last_ingested_at": stmt.excluded.last_ingested_at},
    )
    await session.execute(stmt)
