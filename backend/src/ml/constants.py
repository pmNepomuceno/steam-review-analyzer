from src.config import ROOT_DIR

MODELS_DIR = ROOT_DIR / "backend" / "models_store"
SENTIMENT_MODEL_PATH = MODELS_DIR / "sentiment.joblib"

POSITIVE = "positive"
NEGATIVE = "negative"

# all-MiniLM-L6-v2: 22M params, 384-dim, ~90 MB download, encodes thousands of short
# sentences per second on CPU, and is a strong general-purpose semantic-similarity model.
# paraphrase-MiniLM-L3-v2 is ~2x faster but noticeably worse on similarity; bge-small or
# all-mpnet-base-v2 score a bit higher but are 1.5-5x slower. Inference here is CPU-only
# on a free tier, so speed and size win over the last few points of quality.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Best anchor cosine below which a sentence is tagged "none". Tune in M4 against
# scripts/evaluate_aspects.py; see docs/DECISIONS.md for how the current value was picked.
SIMILARITY_THRESHOLD = 0.40
NONE_ASPECT = "none"

# Sentence splitting. Clause fragments shorter than MIN_CLAUSE_WORDS are merged back into a
# neighbour; segments longer than MAX_SEGMENT_WORDS are cut into word windows to keep units
# sentence-sized. The encoder's 256-wordpiece limit is enforced separately, in tokens, by
# aspects._fit_to_encoder (a word count can't bound text without spaces).
MIN_CLAUSE_WORDS = 3
MAX_SEGMENT_WORDS = 60
