import csv

import numpy as np
import pytest

from scripts import build_eval_sample as bes
from scripts import evaluate_aspects as ea
from src.ml.anchors import ASPECTS

THRESHOLD = 0.40


def make_unit(game: str, i: int, sim: float, aspect: int) -> bes.Unit:
    scores = np.full(len(ASPECTS), sim - 0.2)
    scores[aspect] = sim
    return bes.Unit(appid=1, game=game, review_id=i, sentence=f"{game} sentence {i}",
                    scores=scores)


@pytest.fixture
def sample(tmp_path):
    """A written sample of 20 units over two games plus its paths; labels still blank."""
    units = [make_unit(g, i, sim, i % len(ASPECTS))
             for g in ("a", "b") for i, sim in enumerate([0.35, 0.45, 0.7, 0.1, 0.2] * 4)]
    drawn = bes.draw_sample(units, 20, seed=0, threshold=THRESHOLD)
    paths = {"labels": tmp_path / "labels.csv", "key": tmp_path / "key.csv",
             "meta": tmp_path / "meta.json"}
    bes.write_sample(drawn, units, 0, paths["labels"], paths["key"], paths["meta"], THRESHOLD)
    return drawn, paths


def fill(paths, gold_for, delimiter=",", **extra):
    """Fill gold_aspect with gold_for(key_row); extra maps column -> fn(key_row)."""
    with paths["key"].open(newline="", encoding=bes.CSV_ENCODING) as f:
        key = {r["id"]: r for r in csv.DictReader(f)}
    with paths["labels"].open(newline="", encoding=bes.CSV_ENCODING) as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["gold_aspect"] = gold_for(key[row["id"]])
        for column, fn in extra.items():
            row[column] = fn(key[row["id"]])
    with paths["labels"].open("w", newline="", encoding=bes.CSV_ENCODING) as f:
        writer = csv.DictWriter(f, bes.LABEL_COLUMNS, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def run(paths, threshold=THRESHOLD):
    rows = ea.load_rows(paths["labels"], paths["key"])
    meta = ea.read_meta(paths["meta"])
    scores = ea.stored_scores(rows)
    return ea.evaluate(rows, scores, threshold, meta, "test")


def test_perfect_labels_score_one_and_report_every_section(sample, tmp_path):
    _, paths = sample
    fill(paths, lambda k: k["predicted_aspect"])
    ev = run(paths)
    assert ev.model["accuracy"] == ev.unambiguous["accuracy"] == 1.0
    assert ev.model["macro_f1"] == 1.0
    assert ev.weighted["accuracy"] == pytest.approx(1.0)
    report = ea.format_report(ev)
    for section in ("Summary", "Per aspect", "Confusion matrix", "by stratum", "by game",
                    "Threshold sweep", "Misses (0;"):
        assert section in report
    ea.write_report(report, tmp_path / "report.txt")
    assert (tmp_path / "report.txt").read_text().startswith("Aspect tagging report")


def test_misses_baseline_and_ambiguous_split(sample):
    _, paths = sample
    # Gold says "none" everywhere; the ambiguous flag goes on rows the model tagged.
    fill(paths, lambda k: "none",
         ambiguous=lambda k: "y" if k["predicted_aspect"] != "none" else "")
    ev = run(paths)
    n_none = sum(p == "none" for p in ev.pred)
    assert ev.majority == "none"
    assert ev.baseline["accuracy"] == 1.0
    assert ev.model["correct"] == n_none
    assert ev.unambiguous["n"] == n_none and ev.unambiguous["accuracy"] == 1.0
    assert f"Misses ({20 - n_none};" in ea.format_report(ev)


def test_semicolon_csv_with_messy_labels_is_read(sample):
    # Excel in pt-BR/European locales saves ";"-separated; labelers type " Story ", "Y".
    _, paths = sample
    fill(paths, lambda k: " Story " if k["id"] == "1" else "NONE", delimiter=";",
         ambiguous=lambda k: "Y" if k["id"] == "2" else "")
    rows = {r.id: r for r in ea.load_rows(paths["labels"], paths["key"])}
    assert rows[1].gold == "story" and rows[3].gold == "none"
    assert rows[2].ambiguous and not rows[1].ambiguous
    assert rows[1].sentence  # taken from the key


def test_every_problem_is_reported_at_once(sample):
    _, paths = sample
    fill(paths, lambda k: {"1": "", "2": "graphics", "3": ""}.get(k["id"], "none"),
         ambiguous=lambda k: "maybe" if k["id"] == "4" else "")
    with pytest.raises(ea.LabelError) as exc:
        ea.load_rows(paths["labels"], paths["key"])
    message = str(exc.value)
    assert "2 rows have no gold_aspect yet: ids 1, 3" in message
    assert "'graphics'" in message
    assert "'maybe'" in message


def test_blank_sample_refuses(sample):
    _, paths = sample
    with pytest.raises(ea.LabelError, match="20 rows have no gold_aspect"):
        ea.load_rows(paths["labels"], paths["key"])


def test_ids_must_match_the_key(sample):
    _, paths = sample
    fill(paths, lambda k: "none")
    lines = paths["labels"].read_text(encoding=bes.CSV_ENCODING).splitlines()
    lines[1] = lines[1].replace("1,", "99,", 1)
    paths["labels"].write_text("\n".join(lines) + "\n", encoding=bes.CSV_ENCODING)
    with pytest.raises(ea.LabelError) as exc:
        ea.load_rows(paths["labels"], paths["key"])
    assert "ids missing from labels.csv: 1" in str(exc.value)
    assert "ids not in key.csv: 99" in str(exc.value)


def test_missing_gold_column_is_explained(sample):
    _, paths = sample
    text = paths["labels"].read_text(encoding=bes.CSV_ENCODING)
    paths["labels"].write_text(text.replace("gold_aspect", "gold"), encoding=bes.CSV_ENCODING)
    with pytest.raises(ea.LabelError, match="no gold_aspect column"):
        ea.load_rows(paths["labels"], paths["key"])


def test_non_utf8_file_is_explained(sample):
    _, paths = sample
    paths["labels"].write_bytes("id,gold_aspect\n1,caf\xe9\n".encode("cp1252"))
    with pytest.raises(ea.LabelError, match="CSV UTF-8"):
        ea.load_rows(paths["labels"], paths["key"])


def test_sweep_relabels_the_same_scores():
    scores = np.array([[0.45, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.35, 0.1]])
    assert ea.predict(scores, 0.40) == [ASPECTS[0], "none"]
    assert ea.predict(scores, 0.30) == [ASPECTS[0], ASPECTS[3]]
    assert ea.predict(scores, 0.50) == ["none", "none"]


def test_metrics_macro_f1_ignores_classes_absent_from_gold():
    m = ea.metrics(["price", "price", "none"], ["price", "none", "none"])
    assert m["accuracy"] == pytest.approx(2 / 3)
    assert m["classes"]["price"] == {"precision": 1.0, "recall": 0.5, "f1": pytest.approx(2 / 3),
                                     "support": 2, "predicted": 1}
    assert m["macro_f1"] == pytest.approx((2 / 3 + 2 / 3) / 2)


def test_population_weights_stand_for_their_stratum():
    rows = [ea.Row(i, "g", s, "x", "none", False, "", None)
            for i, s in enumerate(["near_below", "near_below", "other"])]
    meta = {"population_strata": {"g": {"near_below": 10, "other": 90}}}
    np.testing.assert_allclose(ea.population_weights(rows, meta), [5, 5, 90])
    assert ea.population_weights(rows, {}) is None


def test_wilson_interval():
    low, high = ea.wilson(7, 10)
    assert low == pytest.approx(0.3968, abs=1e-3) and high == pytest.approx(0.8922, abs=1e-3)
    assert ea.wilson(0, 0) == (0.0, 0.0)
    assert ea.wilson(10, 10)[1] == 1.0
