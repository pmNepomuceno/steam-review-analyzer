"""Import every domain's models so `Base.metadata` is complete (used by Alembic)."""

from src.database import Base
from src.games.models import Game
from src.reviews.models import Review, ReviewAspect

__all__ = ["Base", "Game", "Review", "ReviewAspect"]
