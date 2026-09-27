"""Sentence-level aspect tagging by cosine similarity to anchor phrases.

A review is split into sentence/clause units (rule-based, see `split_sentences`), each unit
is embedded with a sentence-transformer model (its ONNX export, run on onnxruntime without
torch), and it is tagged with the aspect whose anchors (`anchors.ANCHORS`) it is most similar
to, or "none" when even the best match is below `SIMILARITY_THRESHOLD`.

Scoring: an aspect's score is the max cosine over its anchors, and the highest-scoring
aspect wins outright, with no margin rule. So a sentence that fits two aspects about
equally ("the performance issues aren't worth the price") goes to whichever scores higher,
however small the gap; an exact tie goes to the aspect listed first in `ANCHORS`.

The model and anchor embeddings are loaded once per process on first use (or eagerly via
`load()`), never per call, so importing this module stays cheap and works offline.
"""

import logging
import math
import os
import re
import threading
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from src.config import settings
from src.ml.anchors import ANCHORS, ASPECTS
from src.ml.constants import (
    EMBEDDING_MODEL_FILES,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_REVISION,
    ENCODE_BATCH_SIZE,
    ENCODER_MAX_THREADS,
    MAX_SEGMENT_WORDS,
    MAX_SEQ_LENGTH,
    MIN_CLAUSE_WORDS,
    NONE_ASPECT,
    SIMILARITY_THRESHOLD,
)
from src.ml.schemas import AspectAssignment

logger = logging.getLogger(__name__)

# Steam reviews use BBCode. List items, headings and newlines become sentence boundaries;
# the other known tags are dropped and their text kept. Only real Steam tag names match
# (a tag name must be followed by "]", "=value" or key=value attributes), so bracketed
# prose like "[literally unplayable]" or "[Edit: patched]" is left alone.
_TAG_ATTRS = r"(?:=[^\]]*|(?:\s+\w+=[^\s\]]*)+)?"
_BOUNDARY_TAG = re.compile(
    rf"\[(?:\*|/?(?:h[1-6]|list|olist|li|hr|quote|table|tr){_TAG_ATTRS})\]", re.IGNORECASE
)
_INLINE_TAG = re.compile(
    rf"\[/?(?:b|i|u|s|strike|spoiler|noparse|code|url|img|td|th|previewyoutube){_TAG_ATTRS}\]",
    re.IGNORECASE,
)
# Periods after these are not sentence ends. "etc." is left out on purpose: in reviews it
# usually does end the sentence.
_ABBREVIATION = re.compile(
    r"\b(?:e\.g|i\.e|vs|mr|mrs|ms|dr|jr|sr|st|approx|incl|esp|lvl|feat)\.(?=\s)", re.IGNORECASE
)
_ABBREVIATION_DOT = "\ue000"  # private-use placeholder, restored after splitting
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|[\n;]+")
# Clause boundaries inside one sentence. Needed because reviews often skip punctuation
# entirely ("the story is great but it crashes and the price is too high").
_CLAUSE_SPLIT = re.compile(
    r",?\s+(?=(?:but|however|although|though|whereas|and)\b)|,\s+(?=(?:yet|while)\b)",
    re.IGNORECASE,
)
_LEADING_CONNECTOR = re.compile(
    r"^(?:but|however|although|though|whereas|and|yet|while)\b[,\s]*", re.IGNORECASE
)

_load_lock = threading.Lock()
_model: Any = None  # onnxruntime.InferenceSession
_tokenizer: Any = None  # no truncation or padding: `_encode` pads, `_fit_to_encoder` measures
_anchor_emb: np.ndarray | None = None
_anchor_labels: list[str] = []


def _encode(texts: list[str]) -> np.ndarray:
    """L2-normalized sentence embeddings: the model's Transformer -> mean Pooling -> Normalize."""
    # Batch texts of similar length together, as sentence-transformers does, so little of each
    # batch is padding; the rows are put back in input order at the end.
    order = np.argsort([-len(t) for t in texts], kind="stable")
    batches = []
    for i in range(0, len(texts), ENCODE_BATCH_SIZE):
        encodings = _tokenizer.encode_batch([texts[j] for j in order[i : i + ENCODE_BATCH_SIZE]])
        # Units are already cut to fit (`_fit_to_encoder`); the cap is only a backstop.
        width = min(max(len(e.ids) for e in encodings), MAX_SEQ_LENGTH)
        ids = np.zeros((len(encodings), width), dtype=np.int64)  # 0 is [PAD]
        mask = np.zeros_like(ids)
        for row, e in enumerate(encodings):
            n = min(len(e.ids), width)
            ids[row, :n] = e.ids[:n]
            mask[row, :n] = 1
        feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
        hidden = _model.run(None, feed)[0]  # last_hidden_state, (batch, seq, dim)
        weights = mask[..., None].astype(hidden.dtype)
        pooled = (hidden * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
        batches.append(pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None))
    embeddings = np.concatenate(batches)
    embeddings[order] = embeddings.copy()
    return embeddings


def _cpu_quota(cgroup: Path = Path("/sys/fs/cgroup")) -> float | None:
    """CPUs this container may use by its cgroup quota (v2, then v1); None if unlimited/unknown."""
    try:
        quota, period = (cgroup / "cpu.max").read_text().split()
        return None if quota == "max" else int(quota) / int(period)
    except (OSError, ValueError):
        pass
    try:
        quota = int((cgroup / "cpu" / "cpu.cfs_quota_us").read_text())
        period = int((cgroup / "cpu" / "cpu.cfs_period_us").read_text())
        return None if quota < 0 else quota / period
    except (OSError, ValueError):
        return None


def _usable_cpus() -> int:
    """CPUs this process may run on (cpuset/affinity aware, unlike os.cpu_count())."""
    if hasattr(os, "sched_getaffinity"):  # not on macOS
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _encoder_threads() -> int:
    """ENCODER_THREADS if set, else the usable CPUs, lowered to the CPU quota rounded up, capped."""
    if settings.encoder_threads is not None:
        return settings.encoder_threads
    # Affinity sees a --cpuset-cpus limit but not a CFS quota, so check both.
    cpus = _usable_cpus()
    if quota := _cpu_quota():
        cpus = min(cpus, math.ceil(quota))
    return min(cpus, ENCODER_MAX_THREADS)


def load() -> None:
    """Load the encoder and embed every anchor. Safe to call more than once, from any thread."""
    global _model, _tokenizer, _anchor_emb, _anchor_labels
    # Score columns follow ANCHORS order only while every aspect has an anchor; an empty list
    # would shift every later column onto the wrong name in `label()`.
    if empty := [aspect for aspect, phrases in ANCHORS.items() if not phrases]:
        raise ValueError(f"every aspect needs at least one anchor phrase; empty: {empty}")
    # Background processing runs in worker threads; without the lock two cold runs would
    # each load the ~90 MB encoder and race on the globals.
    with _load_lock:
        if _anchor_emb is not None:  # set last, so a failed load is retried
            return
        threads = _encoder_threads()
        # The tokenizers library batches on a Rayon pool sized to every host core, built on
        # first use; give it the same cap as onnxruntime unless set explicitly.
        os.environ.setdefault("RAYON_NUM_THREADS", str(threads))
        import onnxruntime
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        model_path, tokenizer_path = (
            hf_hub_download(EMBEDDING_MODEL_NAME, f, revision=EMBEDDING_MODEL_REVISION)
            for f in EMBEDDING_MODEL_FILES
        )
        _tokenizer = Tokenizer.from_file(tokenizer_path)
        _tokenizer.no_truncation()
        _tokenizer.no_padding()
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = threads
        # Idle threads busy-wait for work by default, which burns a fractional-CPU quota.
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        logger.info("Aspect encoder uses %d thread(s)", options.intra_op_num_threads)
        _model = onnxruntime.InferenceSession(
            model_path, options, providers=["CPUExecutionProvider"]
        )
        labels = [aspect for aspect, phrases in ANCHORS.items() for _ in phrases]
        phrases = [p for ps in ANCHORS.values() for p in ps]
        _anchor_labels = labels
        _anchor_emb = _encode(phrases)


def _content_words(piece: str) -> int:
    """Word count not counting a leading connector, so "and deep mining" counts as 2."""
    return len(_LEADING_CONNECTOR.sub("", piece).split())


def _merge_short(pieces: list[str]) -> list[str]:
    """Glue fragments under MIN_CLAUSE_WORDS onto a neighbour so they keep their context."""
    merged: list[str] = []
    for piece in pieces:
        shortest = min(_content_words(piece), _content_words(merged[-1])) if merged else 0
        if merged and shortest < MIN_CLAUSE_WORDS:
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return merged


def _windows(segment: str) -> list[str]:
    words = segment.split()
    return [
        " ".join(words[i : i + MAX_SEGMENT_WORDS]) for i in range(0, len(words), MAX_SEGMENT_WORDS)
    ]


def _fit_to_encoder(units: list[str]) -> list[str]:
    """Cut any unit longer than the encoder's token limit at token boundaries.

    The word windows in `split_sentences` bound English text, but text without spaces (CJK,
    URLs, keyboard mashing) can be one "word" of hundreds of wordpieces, which the encoder
    would silently truncate. Chunks run from one token start to the next, so no text is lost.
    """
    limit = MAX_SEQ_LENGTH - 2  # room for [CLS] and [SEP]
    fitted: list[str] = []
    for unit in units:
        offsets = _tokenizer.encode(unit, add_special_tokens=False).offsets
        if len(offsets) <= limit:
            fitted.append(unit)
            continue
        starts = [offsets[i][0] for i in range(0, len(offsets), limit)]
        bounds = [0, *starts[1:], len(unit)]
        fitted.extend(unit[a:b].strip() for a, b in pairwise(bounds))
    return fitted


def split_sentences(text: str) -> list[str]:
    """Split review text into sentence/clause units. Every word of the input is kept."""
    if not text or not text.strip():
        return []
    text = _INLINE_TAG.sub(" ", _BOUNDARY_TAG.sub("\n", text))
    text = _ABBREVIATION.sub(lambda m: m.group()[:-1] + _ABBREVIATION_DOT, text)

    units: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        sentence = " ".join(sentence.split()).replace(_ABBREVIATION_DOT, ".")
        if not sentence:
            continue
        clauses = [c.strip() for c in _CLAUSE_SPLIT.split(sentence) if c.strip()]
        for clause in _merge_short(clauses):
            units.extend(_windows(clause))
    return units


def _aspect_scores(
    sentence_emb: np.ndarray, anchor_emb: np.ndarray, anchor_labels: list[str]
) -> tuple[list[str], np.ndarray]:
    """Per-aspect scores (max cosine over that aspect's anchors) for L2-normalized embeddings.

    Returns the aspect names in first-seen order (== `ANCHORS` order) and a
    (n_sentences, n_aspects) matrix whose columns follow that order.
    """
    aspects = list(dict.fromkeys(anchor_labels))
    sims = sentence_emb @ anchor_emb.T  # (n_sentences, n_anchors); dot == cosine
    label_idx = np.array([aspects.index(label) for label in anchor_labels])
    per_aspect = np.stack([sims[:, label_idx == i].max(axis=1) for i in range(len(aspects))], 1)
    return aspects, per_aspect


def label(
    scores: np.ndarray,
    threshold: float = SIMILARITY_THRESHOLD,
    aspects: Sequence[str] = ASPECTS,
) -> tuple[str, float]:
    """The aspect for one row of per-aspect scores, and its score.

    The best aspect wins; below `threshold` the label is "none" but the best score is still
    returned. An exact tie goes to the aspect listed first.
    """
    best = int(scores.argmax())  # argmax returns the first index on exact ties
    score = float(scores[best])
    return (aspects[best] if score >= threshold else NONE_ASPECT), score


def _assign(
    sentences: list[str],
    sentence_emb: np.ndarray,
    anchor_emb: np.ndarray,
    anchor_labels: list[str],
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[AspectAssignment]:
    """Tag each sentence given L2-normalized embeddings (so dot product == cosine)."""
    aspects, per_aspect = _aspect_scores(sentence_emb, anchor_emb, anchor_labels)
    results: list[AspectAssignment] = []
    for sentence, scores in zip(sentences, per_aspect, strict=True):
        aspect, score = label(scores, threshold, aspects)
        results.append({"sentence": sentence, "aspect": aspect, "similarity": score})
    return results


def score_units(units: list[str]) -> np.ndarray:
    """Per-aspect scores for units that are already split, columns in `ASPECTS` order.

    Units are embedded as given; pass them through `split_sentences` and `_fit_to_encoder`
    first if they may be longer than one sentence.
    """
    if not units:
        return np.empty((0, len(ASPECTS)))
    load()
    emb = _encode(units)
    return _aspect_scores(emb, _anchor_emb, _anchor_labels)[1]


def score_aspects(text: str) -> tuple[list[str], np.ndarray]:
    """Split `text` into units and score each against every aspect (see `score_units`)."""
    sentences = split_sentences(text)
    if not sentences:
        return [], np.empty((0, len(ASPECTS)))
    load()
    sentences = _fit_to_encoder(sentences)
    return sentences, score_units(sentences)


def assign_aspects(text: str) -> list[AspectAssignment]:
    """Tag every sentence of `text` with an aspect or "none". Empty text gives []."""
    sentences, scores = score_aspects(text)
    results: list[AspectAssignment] = []
    for sentence, row in zip(sentences, scores, strict=True):
        aspect, score = label(row)
        results.append({"sentence": sentence, "aspect": aspect, "similarity": score})
    return results
