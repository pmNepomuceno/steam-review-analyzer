"""Inference for the TF-IDF + logistic regression sentiment model.

The artifact is produced offline by `backend/scripts/train_sentiment.py`; nothing here
trains. It is loaded once and cached. Loading is lazy so importing this module works
without a trained artifact (fresh clone, tests); the API can call `load_model()` at
startup to load it eagerly.
"""

from pathlib import Path
from typing import Any

import joblib

from src.ml.constants import NEGATIVE, POSITIVE, SENTIMENT_MODEL_PATH
from src.ml.exceptions import SentimentModelNotFound
from src.ml.schemas import SentimentPrediction

_artifact: dict[str, Any] | None = None


def load_model(path: Path = SENTIMENT_MODEL_PATH) -> dict[str, Any]:
    """Load the artifact at `path` and make it the one `predict()` uses."""
    global _artifact
    if not path.is_file():
        raise SentimentModelNotFound(path)
    _artifact = joblib.load(path)
    return _artifact


def predict(text: str) -> SentimentPrediction:
    artifact = _artifact if _artifact is not None else load_model()
    pipeline = artifact["pipeline"]
    probs = pipeline.predict_proba([text])[0]
    best = int(probs.argmax())
    voted_up = bool(pipeline.classes_[best])
    return {"label": POSITIVE if voted_up else NEGATIVE, "confidence": float(probs[best])}
