from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, Integer, String, Text
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
