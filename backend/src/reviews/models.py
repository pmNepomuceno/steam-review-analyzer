from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class Review(Base):
    __tablename__ = "reviews"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    appid: Mapped[int] = mapped_column(ForeignKey("games.appid"), index=True)
    review_text: Mapped[str] = mapped_column(Text)
    voted_up: Mapped[bool] = mapped_column(Boolean)
    votes_up: Mapped[int] = mapped_column(Integer)
    playtime_forever: Mapped[int] = mapped_column(Integer)
    language: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    # Predicted on the whole review by the aspect pass; NULL until the game is processed.
    predicted_sentiment: Mapped[str | None] = mapped_column(String(8))


class ReviewAspect(Base):
    """One sentence/clause unit of a review with its aspect."""

    __tablename__ = "review_aspects"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    review_id: Mapped[int] = mapped_column(
        ForeignKey("reviews.id", ondelete="CASCADE"), index=True
    )
    sentence_text: Mapped[str] = mapped_column(Text)
    aspect_label: Mapped[str] = mapped_column(String(16))  # an ml.anchors aspect or "none"
    similarity_score: Mapped[float] = mapped_column(Float)
