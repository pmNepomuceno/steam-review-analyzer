from datetime import datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.games.constants import AspectStatus
from src.games.models import Game


async def get_game(session: AsyncSession, appid: int) -> Game | None:
    # populate_existing: callers re-read after waiting on the ingest lock and must
    # see the row another request committed, not a stale identity-map copy.
    stmt = select(Game).where(Game.appid == appid).execution_options(populate_existing=True)
    return await session.scalar(stmt)


async def list_processed(session: AsyncSession) -> list[tuple[int, str]]:
    """(appid, name) of every game whose analysis is done, by name."""
    result = await session.execute(
        select(Game.appid, Game.name)
        .where(Game.aspects_status == AspectStatus.DONE)
        .order_by(Game.name)
    )
    return [(appid, name) for appid, name in result]


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
