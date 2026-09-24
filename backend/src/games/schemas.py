from typing import Literal

from pydantic import BaseModel


class SentimentCounts(BaseModel):
    """Distinct reviews by predicted sentiment."""

    positive: int
    negative: int
    total: int
    positive_pct: float | None  # None when total is 0


class AspectSentiment(SentimentCounts):
    aspect: str


class AspectSummary(BaseModel):
    appid: int
    status: Literal["ready"]
    overall: SentimentCounts  # every processed review, including those with no aspect
    aspects: list[AspectSentiment]


class ProcessingStatus(BaseModel):
    """202 while the app's reviews are being analyzed; 500 once that analysis has failed."""

    appid: int
    status: Literal["processing", "failed"]
    detail: str


# OpenAPI docs for the non-200 answers of endpoints that need processed results.
UNPROCESSED_RESPONSES = {
    202: {"model": ProcessingStatus, "description": "Analysis running; retry shortly"},
    500: {"model": ProcessingStatus, "description": "Analysis failed; needs a --force rerun"},
    503: {"description": "No sentiment model trained yet"},
}
