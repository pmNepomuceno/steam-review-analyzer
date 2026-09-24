import threading
import time

import numpy as np
import pytest

from src.ml import aspects
from src.ml.anchors import ANCHORS, ASPECTS
from src.ml.constants import MAX_SEGMENT_WORDS, NONE_ASPECT, SIMILARITY_THRESHOLD

TARGET = (
    "The story is incredible but it crashes every 20 minutes and the price is way too "
    "high for what you get"
)


# --- sentence splitting (no model) ---


@pytest.mark.parametrize("text", ["", "   ", "\n\t \n"])
def test_split_empty_gives_nothing(text):
    assert aspects.split_sentences(text) == []


def test_split_unpunctuated_one_liner():
    assert aspects.split_sentences("great game") == ["great game"]


def test_split_sentences_and_newlines():
    text = "Loved it. Runs great!\nWould buy again?  Yes"
    assert aspects.split_sentences(text) == ["Loved it.", "Runs great!", "Would buy again?", "Yes"]


def test_split_strips_bbcode_and_breaks_on_list_items():
    text = "[h1]Pros[/h1][list][*]great [b]combat[/b][*]nice music[/list]"
    assert aspects.split_sentences(text) == ["Pros", "great combat", "nice music"]


def test_split_keeps_non_bbcode_brackets():
    assert aspects.split_sentences("[10/10] would play again") == ["[10/10] would play again"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("[literally unplayable] crashes", ["[literally unplayable] crashes"]),
        ("[tried it on steam deck] runs fine", ["[tried it on steam deck] runs fine"]),
        ("[Edit: after the patch it runs fine] ok", ["[Edit: after the patch it runs fine] ok"]),
        ("[i think] it is fine", ["[i think] it is fine"]),
        ("[list of cons] too short", ["[list of cons] too short"]),
    ],
)
def test_split_keeps_bracketed_prose(text, expected):
    assert aspects.split_sentences(text) == expected


def test_split_strips_tags_with_attributes():
    url = "https://store.steampowered.com/app/620/Portal_2/?utm_source=review&curator=12345678"
    text = f"[quote=Bob]runs fine[/quote][url={url}]nice link[/url] [table noborder=1][tr][td]x"
    assert aspects.split_sentences(text) == ["runs fine", "nice link", "x"]


def test_split_ignores_abbreviation_periods():
    assert aspects.split_sentences("Great game e.g. the combat. Mr. Todd is great") == [
        "Great game e.g. the combat.",
        "Mr. Todd is great",
    ]


def test_split_clauses_on_conjunctions():
    assert aspects.split_sentences(TARGET) == [
        "The story is incredible",
        "but it crashes every 20 minutes",
        "and the price is way too high for what you get",
    ]


def test_split_merges_short_fragments():
    # "and gameplay" / "and deep mining" are too short to stand alone as clauses.
    assert aspects.split_sentences("the story and gameplay are great") == [
        "the story and gameplay are great"
    ]
    assert aspects.split_sentences("you get exploration, crafting and deep mining") == [
        "you get exploration, crafting and deep mining"
    ]


def test_split_run_on_keeps_every_word():
    words = [f"word{i}" for i in range(500)]
    units = aspects.split_sentences(" ".join(words))
    assert len(units) > 1
    assert all(len(u.split()) <= MAX_SEGMENT_WORDS for u in units)
    assert " ".join(units).split() == words


# --- assignment logic with fake embeddings (no model) ---

LABELS = ["a", "a", "b"]
ANCHOR_EMB = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])


def test_assign_best_aspect_uses_max_over_anchors():
    [r] = aspects._assign(["s"], np.array([[0.0, 1.0]]), ANCHOR_EMB, LABELS, threshold=0.5)
    assert r == {"sentence": "s", "aspect": "a", "similarity": 1.0}
    assert type(r["similarity"]) is float


def test_assign_below_threshold_is_none_and_keeps_score():
    [r] = aspects._assign(["s"], np.array([[0.6, 0.8]]), ANCHOR_EMB, ["a", "a", "b"], 1.01)
    assert r["aspect"] == NONE_ASPECT
    assert r["similarity"] == pytest.approx(1.0)


def test_assign_exact_tie_goes_to_first_aspect():
    anchor_emb = np.array([[1.0, 0.0], [1.0, 0.0]])
    for labels, expected in ((["a", "b"], "a"), (["b", "a"], "b")):
        [r] = aspects._assign(["s"], np.array([[1.0, 0.0]]), anchor_emb, labels, 0.5)
        assert r["aspect"] == expected


def test_load_rejects_an_aspect_without_anchors(monkeypatch):
    monkeypatch.setattr(aspects, "ANCHORS", {**ANCHORS, "price": []})
    with pytest.raises(ValueError, match="price"):
        aspects.load()


def test_concurrent_load_builds_the_encoder_once(monkeypatch):
    import sentence_transformers

    built = []

    class SlowEncoder:
        def __init__(self, *args, **kwargs):
            built.append(self)
            time.sleep(0.2)  # wide window for a second thread to slip in without the lock

        def encode(self, phrases, **kwargs):
            return np.zeros((len(phrases), 2))

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", SlowEncoder)
    monkeypatch.setattr(aspects, "_model", None)
    monkeypatch.setattr(aspects, "_anchor_emb", None)
    monkeypatch.setattr(aspects, "_anchor_labels", [])

    threads = [threading.Thread(target=aspects.load) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(built) == 1


def test_anchor_config_shape():
    assert ASPECTS == ("performance", "price", "bugs", "story", "gameplay")
    # Anchor counts grow from evaluation findings (M4 added three gameplay anchors), not a
    # fixed authoring rule; the upper bound only catches a runaway list.
    assert all(4 <= len(phrases) <= 12 for phrases in ANCHORS.values())
    assert 0.0 < SIMILARITY_THRESHOLD < 1.0


# --- real model (downloads all-MiniLM-L6-v2 into the HF cache on first run) ---


@pytest.fixture(scope="module")
def model():
    aspects.load()


def assert_valid(results):
    for r in results:
        assert set(r) == {"sentence", "aspect", "similarity"}
        assert r["aspect"] in (*ASPECTS, NONE_ASPECT)
        assert type(r["similarity"]) is float
        assert -1.0 <= r["similarity"] <= 1.0


def test_multi_topic_review_is_tagged_per_sentence(model):
    results = aspects.assign_aspects(TARGET)
    assert_valid(results)
    assert [r["aspect"] for r in results] == ["story", "bugs", "price"]
    assert all(r["similarity"] >= SIMILARITY_THRESHOLD for r in results)


@pytest.mark.parametrize(
    "text, aspect",
    [
        ("the frame rate is awful even on low settings", "performance"),
        ("not worth $70, wait for a sale", "price"),
        ("my save got corrupted twice", "bugs"),
        ("the characters are so likeable", "story"),
        ("the combat feels amazing", "gameplay"),
    ],
)
def test_clear_sentences_get_their_aspect(model, text, aspect):
    [r] = aspects.assign_aspects(text)
    assert r["aspect"] == aspect


@pytest.mark.parametrize(
    "text", ["I bought this after watching a friend play it", "lol", "meh", "10/10", "Thanks Todd"]
)
def test_off_topic_and_tiny_sentences_are_none(model, text):
    [r] = aspects.assign_aspects(text)
    assert r["aspect"] == NONE_ASPECT
    assert r["similarity"] < SIMILARITY_THRESHOLD


@pytest.mark.parametrize("text", ["", "   ", "\n"])
def test_empty_text_gives_empty_list(model, text):
    assert aspects.assign_aspects(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "great game",
        "🔥🔥🔥 👍",
        "このゲームは最高",
        "Ça marche très bien",
        "The story is great pero el juego se cierra cada rato",
        "g̷̢l̶i̵t̸c̷h̶ \x00\x07\u200b",
        "a" * 20_000,
        "long run on " * 2_000,
    ],
)
def test_assign_never_crashes(model, text):
    results = aspects.assign_aspects(text)
    assert results
    assert_valid(results)


def test_long_unspaced_text_is_cut_to_encoder_limit(model):
    text = "このゲームは最高です。" * 40  # one "word", ~440 wordpieces
    units = aspects._fit_to_encoder(aspects.split_sentences(text))
    limit = aspects._model.max_seq_length - 2
    assert len(units) > 1
    assert all(len(aspects._model.tokenizer.tokenize(u)) <= limit for u in units)
    assert "".join(units) == text
    assert len(aspects.assign_aspects(text)) == len(units)


def test_near_tie_goes_to_one_aspect(model):
    # Plausibly performance or price; best match wins, no split or "ambiguous" label.
    [r] = aspects.assign_aspects("the performance issues aren't worth the price")
    assert r["aspect"] in {"performance", "price"}


def test_assign_aspects_matches_label_over_scores(model):
    sentences, scores = aspects.score_aspects(TARGET)
    assert scores.shape == (len(sentences), len(ASPECTS))
    expected = [aspects.label(row) for row in scores]
    got = [(r["aspect"], r["similarity"]) for r in aspects.assign_aspects(TARGET)]
    assert got == [(a, pytest.approx(s)) for a, s in expected]
    np.testing.assert_allclose(aspects.score_units(sentences), scores, atol=1e-6)
