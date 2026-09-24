from typing import Literal, TypedDict


class SentimentPrediction(TypedDict):
    label: Literal["positive", "negative"]
    confidence: float  # probability of `label`, unrounded, in [0.5, 1]


class AspectAssignment(TypedDict):
    sentence: str
    aspect: str  # one of anchors.ASPECTS or "none"
    similarity: float  # best cosine to any anchor, reported even when aspect is "none"
