"""Sentence-level aspect tagging by cosine similarity to anchor phrases.

A review is split into sentence/clause units (rule-based, see `split_sentences`), each unit
is embedded with a sentence-transformer, and it is tagged with the aspect whose anchors
(`anchors.ANCHORS`) it is most similar to, or "none" when even the best match is below
`SIMILARITY_THRESHOLD`.

Scoring: an aspect's score is the max cosine over its anchors, and the highest-scoring
aspect wins outright, with no margin rule. So a sentence that fits two aspects about
equally ("the performance issues aren't worth the price") goes to whichever scores higher,
however small the gap; an exact tie goes to the aspect listed first in `ANCHORS`.

The model and anchor embeddings are loaded once per process on first use (or eagerly via
`load()`), never per call, so importing this module stays cheap and works offline.
"""

import re
from itertools import pairwise
from typing import Any

import numpy as np

from src.ml.anchors import ANCHORS
from src.ml.constants import (
    EMBEDDING_MODEL_NAME,
    MAX_SEGMENT_WORDS,
    MIN_CLAUSE_WORDS,
    NONE_ASPECT,
    SIMILARITY_THRESHOLD,
)
from src.ml.schemas import AspectAssignment

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

_model: Any = None
_anchor_emb: np.ndarray | None = None
_anchor_labels: list[str] = []


def load() -> None:
    """Load the encoder and embed every anchor. Safe to call more than once."""
    global _model, _anchor_emb, _anchor_labels
    if _model is not None:
        return
    from sentence_transformers import SentenceTransformer  # heavy import (torch)

    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    labels = [aspect for aspect, phrases in ANCHORS.items() for _ in phrases]
    phrases = [p for ps in ANCHORS.values() for p in ps]
    _anchor_emb = model.encode(phrases, normalize_embeddings=True, convert_to_numpy=True)
    _anchor_labels = labels
    _model = model


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
    limit = _model.max_seq_length - 2  # room for [CLS] and [SEP]
    fitted: list[str] = []
    for unit in units:
        offsets = _model.tokenizer(
            unit, add_special_tokens=False, return_offsets_mapping=True
        )["offset_mapping"]
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


def _assign(
    sentences: list[str],
    sentence_emb: np.ndarray,
    anchor_emb: np.ndarray,
    anchor_labels: list[str],
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[AspectAssignment]:
    """Tag each sentence given L2-normalized embeddings (so dot product == cosine)."""
    aspects = list(dict.fromkeys(anchor_labels))  # first-seen order == ANCHORS order
    sims = sentence_emb @ anchor_emb.T  # (n_sentences, n_anchors)
    label_idx = np.array([aspects.index(label) for label in anchor_labels])
    per_aspect = np.stack([sims[:, label_idx == i].max(axis=1) for i in range(len(aspects))], 1)

    results: list[AspectAssignment] = []
    for sentence, scores in zip(sentences, per_aspect, strict=True):
        best = int(scores.argmax())  # argmax returns the first index on exact ties
        score = float(scores[best])
        aspect = aspects[best] if score >= threshold else NONE_ASPECT
        results.append({"sentence": sentence, "aspect": aspect, "similarity": score})
    return results


def assign_aspects(text: str) -> list[AspectAssignment]:
    """Tag every sentence of `text` with an aspect or "none". Empty text gives []."""
    sentences = split_sentences(text)
    if not sentences:
        return []
    load()
    sentences = _fit_to_encoder(sentences)
    emb = _model.encode(sentences, normalize_embeddings=True, convert_to_numpy=True)
    return _assign(sentences, emb, _anchor_emb, _anchor_labels)
