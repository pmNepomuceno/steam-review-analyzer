from src.config import ROOT_DIR

MODELS_DIR = ROOT_DIR / "backend" / "models_store"
SENTIMENT_MODEL_PATH = MODELS_DIR / "sentiment.joblib"

POSITIVE = "positive"
NEGATIVE = "negative"

# all-MiniLM-L6-v2 (run as its ONNX export on onnxruntime, no torch): 22M params, 384-dim,
# ~90 MB download, encodes thousands of short
# sentences per second on CPU, and is a strong general-purpose semantic-similarity model.
# paraphrase-MiniLM-L3-v2 is ~2x faster but noticeably worse on similarity; bge-small or
# all-mpnet-base-v2 score a bit higher but are 1.5-5x slower. Inference here is CPU-only
# on a free tier, so speed and size win over the last few points of quality.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Pinned so the ONNX export can't change under the eval numbers. The full-precision
# onnx/model.onnx is used; the quantized exports in the same repo shift the scores.
EMBEDDING_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
EMBEDDING_MODEL_FILES = ("onnx/model.onnx", "tokenizer.json")
# The model's max_seq_length (its sentence_bert_config.json); longer input is truncated.
MAX_SEQ_LENGTH = 256
# Encoder activation memory scales with batch size x sequence length, and real reviews have
# units at the full 256 tokens. At 32 (sentence-transformers' default) one cached game peaked
# at 640 MB, over Render's 512 MB; at 8 the worst of five games peaked at 411 MB in the
# production image, no slower (docs/DECISIONS.md).
ENCODE_BATCH_SIZE = 8
# onnxruntime's default of one thread per visible core oversubscribes: on a 32-core dev box it
# encoded 3000 units in 22 s against 9 s with 8 threads (4: 14 s, 1: 42 s). Below this cap the
# count follows the container's CPU quota (aspects._encoder_threads), so Render's fractional
# CPU gets 1 thread; the ENCODER_THREADS setting overrides both.
ENCODER_MAX_THREADS = 8

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
