import csv
from collections import Counter

import numpy as np
import pytest

from scripts import build_eval_sample as bes
from scripts.build_eval_sample import Unit
from src.ml.anchors import ASPECTS

THRESHOLD = 0.40


def unit(game: str, i: int, sim: float, aspect: int = 0, words: int = 5) -> Unit:
    """A unit whose best score is `sim` on ASPECTS[aspect]; the others trail by 0.2."""
    scores = np.full(len(ASPECTS), sim - 0.2)
    scores[aspect] = sim
    sentence = " ".join([f"{game}-{i}"] + ["w"] * (words - 1))
    return Unit(appid=hash(game) % 1000, game=game, review_id=i, sentence=sentence,
                scores=scores)


def game_units(game: str, n_per_kind: int, start: int = 0) -> list[Unit]:
    """Plenty of every stratum and every predicted class."""
    units, i = [], start
    for _ in range(n_per_kind):
        for sim in (0.33, 0.37, 0.42, 0.48):  # band
            units.append(unit(game, i, sim, aspect=i % len(ASPECTS)))
            i += 1
        units.append(unit(game, i, 0.2, words=30))  # long, clear none
        i += 1
        units.append(unit(game, i, 0.1))  # clear none
        i += 1
        for a in range(len(ASPECTS)):  # clear aspect
            units.append(unit(game, i, 0.7, aspect=a))
            i += 1
    return units


def test_stratum_bounds_follow_threshold():
    assert bes.band_bounds(0.40) == (0.30, 0.50)
    cases = [(0.2999, "other"), (0.30, "near_below"), (0.3999, "near_below"),
             (0.40, "near_above"), (0.50, "near_above"), (0.5001, "other")]
    for sim, expected in cases:
        assert bes.stratum_of(unit("g", 0, sim), THRESHOLD) == expected
    assert bes.stratum_of(unit("g", 0, 0.60, words=25), THRESHOLD) == "long"
    assert bes.stratum_of(unit("g", 0, 0.45, words=40), THRESHOLD) == "near_above"
    assert bes.stratum_of(unit("g", 0, 0.45), 0.50) == "near_below"


def test_quotas_add_up():
    assert bes.quotas(28) == {"near_below": 7, "near_above": 7, "long": 4, "other": 10}
    for per_game in range(1, 60):
        assert sum(bes.quotas(per_game).values()) == per_game


def test_each_game_gets_equal_share_despite_size_imbalance():
    units = game_units("big", 50) + game_units("small", 5, start=10_000)
    sample = bes.draw_sample(units, 56, seed=1, threshold=THRESHOLD)
    assert Counter(u.game for u in sample.units) == {"big": 28, "small": 28}
    for game in ("big", "small"):
        strata = Counter(s for u, s in zip(sample.units, sample.strata, strict=True)
                         if u.game == game)
        assert strata == {"near_below": 7, "near_above": 7, "long": 4, "other": 10}
    assert sample.shortfalls == {}


def test_other_stratum_covers_every_predicted_class():
    sample = bes.draw_sample(game_units("g", 20), 28, seed=3, threshold=THRESHOLD)
    other = [u for u, s in zip(sample.units, sample.strata, strict=True) if s == "other"]
    assert {u.predicted(THRESHOLD) for u in other} == {*ASPECTS, "none"}


def test_same_seed_same_sample_and_different_seed_differs():
    units = game_units("a", 10) + game_units("b", 10, start=5000)
    first = bes.draw_sample(units, 40, seed=7, threshold=THRESHOLD)
    again = bes.draw_sample(units, 40, seed=7, threshold=THRESHOLD)
    other = bes.draw_sample(units, 40, seed=8, threshold=THRESHOLD)
    sentences = [u.sentence for u in first.units]
    assert sentences == [u.sentence for u in again.units]
    assert sentences != [u.sentence for u in other.units]


def test_duplicates_are_sampled_once():
    dupes = [Unit(1, g, i, "10/10  Would Play", np.full(len(ASPECTS), 0.35))
             for i, g in enumerate(["a", "a", "b"])]
    sample = bes.draw_sample(dupes, 10, seed=0, threshold=THRESHOLD)
    assert len(sample.units) == 1


def test_short_stratum_is_backfilled_and_reported():
    # No near-threshold units at all: band slots are backfilled from the rest.
    units = [unit("g", i, 0.1 if i % 2 else 0.7, aspect=i % 5) for i in range(100)]
    sample = bes.draw_sample(units, 28, seed=0, threshold=THRESHOLD)
    assert len(sample.units) == 28
    assert sample.shortfalls == {"g": {"near_below": 7, "near_above": 7, "long": 4}}
    assert set(sample.strata) == {"other"}  # backfilled units keep their true stratum


def test_game_smaller_than_quota_gives_what_it_has():
    sample = bes.draw_sample([unit("g", i, 0.35) for i in range(5)], 28, 0, THRESHOLD)
    assert len(sample.units) == 5


def write(tmp_path, units=None):
    units = units or game_units("a", 5) + game_units("b", 5, start=5000)
    sample = bes.draw_sample(units, 20, seed=0, threshold=THRESHOLD)
    paths = {"labels": tmp_path / "labels.csv", "key": tmp_path / "key.csv",
             "meta": tmp_path / "meta.json"}
    bes.write_sample(sample, units, 0, paths["labels"], paths["key"], paths["meta"], THRESHOLD)
    return sample, paths


def read(path):
    with path.open(newline="", encoding=bes.CSV_ENCODING) as f:
        return list(csv.DictReader(f))


def rewrite(path, rows, delimiter=","):
    with path.open("w", newline="", encoding=bes.CSV_ENCODING) as f:
        writer = csv.DictWriter(f, bes.LABEL_COLUMNS, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def test_labels_file_is_blind_and_joins_to_key(tmp_path):
    sample, paths = write(tmp_path)
    labels, key = read(paths["labels"]), read(paths["key"])
    assert list(labels[0]) == bes.LABEL_COLUMNS
    assert not {"predicted_aspect", "predicted_similarity", "stratum"} & set(labels[0])
    assert all(r["gold_aspect"] == "" for r in labels)
    assert [r["id"] for r in labels] == [r["id"] for r in key] == [
        str(i) for i in range(1, 21)]
    assert [r["sentence"] for r in labels] == [r["sentence"] for r in key]
    for row, u in zip(key, sample.units, strict=True):
        assert row["predicted_aspect"] == u.predicted(THRESHOLD)
        assert float(row["margin"]) == pytest.approx(0.2, abs=1e-3)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["key.csv", "labels.csv", "meta.json"]


def test_key_scores_relabel_exactly_at_the_threshold(tmp_path):
    # 0.39996 printed to 4 decimals is 0.4000, which would re-label as an aspect.
    units = [unit("g", 0, 0.39996, aspect=2), unit("g", 1, np.float32(0.40001), aspect=1)]
    sample, paths = write(tmp_path, units)
    for row, u in zip(read(paths["key"]), sample.units, strict=True):
        scores = np.array([float(row[f"score_{a}"]) for a in ASPECTS])
        np.testing.assert_array_equal(scores, u.scores)
        assert bes.label(scores, THRESHOLD)[0] == row["predicted_aspect"]
        assert float(row["predicted_similarity"]) == u.similarity
    assert {r["predicted_aspect"] for r in read(paths["key"])} == {"none", ASPECTS[1]}


def test_fresh_labels_file_is_not_edited(tmp_path):
    _, paths = write(tmp_path)
    assert not bes.labels_edited(paths["labels"], paths["meta"])
    assert not bes.labels_edited(tmp_path / "missing.csv", paths["meta"])


def test_gold_label_counts_as_edited(tmp_path):
    _, paths = write(tmp_path)
    rows = read(paths["labels"])
    rows[3]["gold_aspect"] = "story"
    rewrite(paths["labels"], rows)
    assert bes.labels_edited(paths["labels"], paths["meta"])


def test_labels_saved_with_other_delimiter_count_as_edited(tmp_path):
    # Excel in pt-BR/European locales saves ";"-separated CSV, so a gold_aspect column
    # can't be found; the guard must not take that as "no labels".
    _, paths = write(tmp_path)
    rows = read(paths["labels"])
    rows[0]["gold_aspect"] = "bugs"
    rewrite(paths["labels"], rows, delimiter=";")
    assert bes.labels_edited(paths["labels"], paths["meta"])


def test_labels_without_recorded_hash_count_as_edited(tmp_path):
    _, paths = write(tmp_path)
    paths["meta"].unlink()
    assert bes.labels_edited(paths["labels"], paths["meta"])
    paths["meta"].write_text("not json", encoding="utf-8")
    assert bes.labels_edited(paths["labels"], paths["meta"])


def test_failed_write_leaves_existing_files_untouched(tmp_path, monkeypatch):
    _, paths = write(tmp_path)
    before = {name: path.read_bytes() for name, path in paths.items()}

    def boom(self):
        raise RuntimeError("disk full")

    monkeypatch.setattr(Unit, "runner_up", boom)
    with pytest.raises(RuntimeError):
        write(tmp_path, game_units("c", 5))
    monkeypatch.undo()
    monkeypatch.setattr(bes.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("full")))
    with pytest.raises(OSError):
        write(tmp_path, game_units("c", 5))

    assert {name: path.read_bytes() for name, path in paths.items()} == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["key.csv", "labels.csv", "meta.json"]


def test_back_up_copies_existing_files(tmp_path):
    _, paths = write(tmp_path)
    backups = bes.back_up([paths["labels"], paths["key"], tmp_path / "missing.csv"])
    assert len(backups) == 2
    assert backups[0].name.startswith("labels.bak-") and backups[0].suffix == ".csv"
    assert backups[0].read_bytes() == paths["labels"].read_bytes()


def test_runner_up_and_margin():
    u = Unit(1, "g", 1, "s", np.array([0.1, 0.5, 0.45, 0.2, 0.3]))
    assert u.runner_up() == (ASPECTS[2], pytest.approx(0.05))
    assert u.predicted(0.40) == ASPECTS[1]
    assert u.predicted(0.60) == "none"
