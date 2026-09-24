from typing import Literal, TypedDict


class SentimentPrediction(TypedDict):
    label: Literal["positive", "negative"]
    confidence: float  # probability of `label`, unrounded, in [0.5, 1]
