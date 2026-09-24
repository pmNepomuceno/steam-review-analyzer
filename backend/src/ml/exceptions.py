from pathlib import Path

from src.exceptions import AppError


class SentimentModelNotFound(AppError):
    status_code = 503

    def __init__(self, path: Path):
        super().__init__(
            f"Sentiment model not found at {path}; run backend/scripts/train_sentiment.py"
        )
