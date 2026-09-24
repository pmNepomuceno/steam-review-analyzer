from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base
from src.games.constants import AspectStatus

_STATUSES = ", ".join(f"'{s}'" for s in AspectStatus)


class Game(Base):
    __tablename__ = "games"
    __table_args__ = (
        CheckConstraint(f"aspects_status IN ({_STATUSES})", name="ck_games_aspects_status"),
    )

    appid: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    name: Mapped[str] = mapped_column(Text)
    last_ingested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Set by reviews.service.process_reviews. "done" means review_aspects and
    # reviews.predicted_sentiment hold results for every cached review; ingestion never
    # refreshes a game, so it stays done.
    aspects_status: Mapped[str] = mapped_column(
        String(16), server_default=AspectStatus.PENDING.value
    )
    aspects_error: Mapped[str | None] = mapped_column(Text)  # last failure, when "failed"
    # When a run last finished successfully; kept through later failed or forced runs, so it
    # says how old the stored results are (e.g. for a future refresh policy).
    aspects_processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
