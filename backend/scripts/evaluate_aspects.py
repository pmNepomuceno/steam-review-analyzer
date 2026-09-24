"""Score the aspect tagging against the hand-labeled sample (M4).

Joins the gold labels in `docs/eval/aspect_labels.csv` (filled in by hand, rules in
`docs/eval/ASPECT_LABELING.md`) to the model side in `aspect_sample_key.csv` on id, and
reports:

- per-aspect precision / recall / F1 and accuracy next to a majority-class baseline, for all
  rows and without the rows flagged ambiguous;
- a population-weighted estimate: the sample oversamples the hard near-threshold band, so
  each unit is weighted by how many units of its game and stratum it stands for;
- a confusion matrix, accuracy per stratum and per game, what the model does with the `none`
  topics tagged in notes (graphics, audio, ...), a threshold sweep, and every miss.

By default the key's sentences are re-scored with the current `ANCHORS` and model, so editing
anchors and re-running shows the effect directly. `--from-key` uses the scores stored when
the sample was drawn instead (no model needed). The report is written to
`docs/eval/aspect_report.txt`.

Run from backend/: `python scripts/evaluate_aspects.py [--threshold 0.40] [--from-key]`
"""

import argparse
import csv
import io
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from scripts.build_eval_sample import (
    CLASSES,
    CSV_ENCODING,
    EVAL_DIR,
    KEY_PATH,
    LABELS_PATH,
    META_PATH,
    STRATA,
    anchors_hash,
)
from src.ml.anchors import ASPECTS
from src.ml.aspects import label
from src.ml.constants import NONE_ASPECT, SIMILARITY_THRESHOLD

REPORT_PATH = EVAL_DIR / "aspect_report.txt"  # tracked in git
# thresholds tried: 0.30-0.50 in steps of 0.02, plus 0.35 and 0.45 (the +/-0.05 around 0.40)
SWEEP = tuple(sorted({round(0.30 + 0.02 * i, 2) for i in range(11)} | {0.35, 0.45}))
LOW_SUPPORT_WARNING = 10  # gold examples below which a class's metrics are very noisy
SENTENCE_CHARS = 100  # sentences in the list of misses are cut to this length
YES = {"y", "yes", "x", "true", "1"}
NO = {"", "n", "no", "false", "0"}


class LabelError(ValueError):
    """The labels or key file can't be evaluated as it stands."""


@dataclass
class Row:
    id: int
    game: str
    stratum: str  # at sampling time, for the threshold in the meta file
    sentence: str
    gold: str
    ambiguous: bool
    note: str
    key_scores: np.ndarray | None  # scores stored when the sample was drawn (ASPECTS order)


def read_table(path: Path) -> list[dict[str, str]]:
    """Rows of a CSV with lowercase headers and stripped values, blank rows dropped.

    The delimiter is read off the header line, since spreadsheets in many locales save
    "CSV" with ";" (the column names are plain words, so the header has no other
    punctuation).
    """
    try:
        text = path.read_text(encoding=CSV_ENCODING)
    except FileNotFoundError:
        raise LabelError(f"{path} not found") from None
    except UnicodeDecodeError:
        raise LabelError(f'{path.name} is not UTF-8; save it again as "CSV UTF-8"') from None
    delimiter = max((",", ";", "\t"), key=text.split("\n", 1)[0].count)
    rows = []
    for raw in csv.DictReader(io.StringIO(text, newline=""), delimiter=delimiter):
        row = {k.strip().lower(): (v or "").strip() for k, v in raw.items() if k is not None}
        if any(row.values()):
            rows.append(row)
    return rows


def _ambiguous(value: str) -> bool | None:
    value = value.lower()
    return True if value in YES else False if value in NO else None


def _ids(ids: list, limit: int = 15) -> str:
    more = f" (+{len(ids) - limit} more)" if len(ids) > limit else ""
    return ", ".join(map(str, ids[:limit])) + more


def _key_scores(row: dict[str, str]) -> np.ndarray | None:
    try:
        return np.array([float(row[f"score_{a}"]) for a in ASPECTS])
    except (KeyError, ValueError):
        return None


def load_rows(labels_path: Path = LABELS_PATH, key_path: Path = KEY_PATH) -> list[Row]:
    """Join the gold labels to the key on id. Raises LabelError listing every problem.

    Only id, gold_aspect, ambiguous and notes are read from the labels file; the sentence
    comes from the key, so a spreadsheet mangling the text doesn't matter.
    """
    labels, key = read_table(labels_path), read_table(key_path)
    for name, rows, needed in ((labels_path.name, labels, {"id", "gold_aspect"}),
                               (key_path.name, key, {"id", "game", "stratum", "sentence"})):
        if missing := needed - set(rows[0] if rows else ()):
            raise LabelError(f"{name} has no {', '.join(sorted(missing))} column")

    problems: list[str] = []
    gold_by_id: dict[int, dict[str, str]] = {}
    for line, row in enumerate(labels, 2):  # line 1 is the header
        try:
            row_id = int(row["id"])
        except ValueError:
            problems.append(f"line {line}: id {row['id']!r} is not a number")
            continue
        if row_id in gold_by_id:
            problems.append(f"id {row_id} appears more than once")
        gold_by_id[row_id] = row
    key_by_id = {int(row["id"]): row for row in key}
    if missing := sorted(key_by_id.keys() - gold_by_id.keys()):
        problems.append(f"ids missing from {labels_path.name}: {_ids(missing)}")
    if extra := sorted(gold_by_id.keys() - key_by_id.keys()):
        problems.append(f"ids not in {key_path.name}: {_ids(extra)}")

    blank, bad_gold, bad_ambiguous = [], [], []
    for row_id in sorted(gold_by_id.keys() & key_by_id.keys()):
        row = gold_by_id[row_id]
        gold = row["gold_aspect"].lower()
        if not gold:
            blank.append(row_id)
        elif gold not in CLASSES:
            bad_gold.append(f"{row_id} ({row['gold_aspect']!r})")
        if _ambiguous(row.get("ambiguous", "")) is None:
            bad_ambiguous.append(f"{row_id} ({row['ambiguous']!r})")
    if blank:
        problems.append(f"{len(blank)} rows have no gold_aspect yet: ids {_ids(blank)}")
    if bad_gold:
        problems.append(f"gold_aspect must be one of {', '.join(CLASSES)}; got "
                        f"{_ids(bad_gold)}")
    if bad_ambiguous:
        problems.append(f"ambiguous must be y or blank; got {_ids(bad_ambiguous)}")
    if problems:
        raise LabelError("\n".join(problems))

    return [
        Row(id=row_id, game=key_by_id[row_id]["game"], stratum=key_by_id[row_id]["stratum"],
            sentence=key_by_id[row_id]["sentence"], gold=gold["gold_aspect"].lower(),
            ambiguous=bool(_ambiguous(gold.get("ambiguous", ""))),
            note=gold.get("notes", "").lower(), key_scores=_key_scores(key_by_id[row_id]))
        for row_id, gold in sorted(gold_by_id.items())
    ]


def read_meta(path: Path = META_PATH) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def stored_scores(rows: list[Row]) -> np.ndarray:
    if any(row.key_scores is None for row in rows):
        raise LabelError("the key has no score column for some aspect in ANCHORS (were "
                         "aspects added since the sample was drawn?); re-score instead of "
                         "using --from-key")
    return np.stack([row.key_scores for row in rows])


def rescore(rows: list[Row]) -> np.ndarray:
    """Score the key's units with the current anchors (loads the model on first use)."""
    from src.ml import aspects

    return aspects.score_units([row.sentence for row in rows])


def predict(scores: np.ndarray, threshold: float) -> list[str]:
    return [label(row, threshold)[0] for row in scores]


def metrics(gold: list[str], pred: list[str], weights: np.ndarray | None = None) -> dict:
    """Accuracy, per-class precision/recall/F1, and macro F1 over the classes in gold."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold, pred, labels=list(CLASSES), zero_division=0, sample_weight=weights
    )
    support, predicted = Counter(gold), Counter(pred)
    present = [i for i, c in enumerate(CLASSES) if support[c]]
    return {
        "n": len(gold),
        "correct": sum(g == p for g, p in zip(gold, pred, strict=True)),
        "accuracy": float(accuracy_score(gold, pred, sample_weight=weights)) if gold else 0.0,
        "macro_f1": float(np.mean(f1[present])) if present else 0.0,
        "classes": {
            c: {"precision": float(precision[i]), "recall": float(recall[i]),
                "f1": float(f1[i]), "support": support[c], "predicted": predicted[c]}
            for i, c in enumerate(CLASSES)
        },
    }


def wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion; sane at small n and near 0 or 1."""
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def population_weights(rows: list[Row], meta: dict) -> np.ndarray | None:
    """How many population units each sampled unit stands for: its game and stratum's
    population count over the number sampled from it. None if the meta file doesn't cover
    the sample."""
    population = meta.get("population_strata") or {}
    sampled = Counter((row.game, row.stratum) for row in rows)
    try:
        return np.array([population[row.game][row.stratum] / sampled[row.game, row.stratum]
                         for row in rows])
    except (KeyError, TypeError):
        return None


@dataclass
class Evaluation:
    rows: list[Row]
    scores: np.ndarray
    pred: list[str]
    threshold: float
    majority: str  # most frequent gold class, what the baseline always predicts
    model: dict
    baseline: dict
    unambiguous: dict
    baseline_unambiguous: dict
    weighted: dict | None
    baseline_weighted: dict | None
    covered_units: float | None  # population units the weights stand for (sum of weights)
    population_units: int | None  # deduped population size, to check the weights cover it
    sweep: list[dict]
    source: str  # where the scores came from


def evaluate(rows: list[Row], scores: np.ndarray, threshold: float, meta: dict,
             source: str) -> Evaluation:
    gold = [row.gold for row in rows]
    pred = predict(scores, threshold)
    majority = Counter(gold).most_common(1)[0][0]
    base = [majority] * len(rows)
    clear = [i for i, row in enumerate(rows) if not row.ambiguous]
    weights = population_weights(rows, meta)

    def pick(values: list[str]) -> list[str]:
        return [values[i] for i in clear]

    sweep = []
    for t in sorted({*SWEEP, threshold}):
        m = metrics(gold, predict(scores, t))
        none = m["classes"][NONE_ASPECT]
        sweep.append({"threshold": t, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                      "none_precision": none["precision"], "none_recall": none["recall"],
                      "none_share": none["predicted"] / len(rows)})

    return Evaluation(
        rows=rows, scores=scores, pred=pred, threshold=threshold, majority=majority,
        model=metrics(gold, pred), baseline=metrics(gold, base),
        unambiguous=metrics(pick(gold), pick(pred)),
        baseline_unambiguous=metrics(pick(gold), pick(base)),
        weighted=None if weights is None else metrics(gold, pred, weights),
        baseline_weighted=None if weights is None else metrics(gold, base, weights),
        covered_units=None if weights is None else float(weights.sum()),
        population_units=meta.get("population_units_deduped"),
        sweep=sweep, source=source,
    )


def score_source(rows: list[Row], scores: np.ndarray, meta: dict, rescored: bool) -> str:
    """One line saying where the scores came from and whether they drifted from the key."""
    if not rescored:
        return "scores stored in the key when the sample was drawn (--from-key)"
    changed = "changed" if meta.get("anchors_sha256") != anchors_hash() else "unchanged"
    text = f"re-scored with the current anchors ({changed} since the sample was drawn)"
    if all(row.key_scores is not None for row in rows) and "threshold" in meta:
        t = meta["threshold"]
        old = predict(stored_scores(rows), t)
        flipped = sum(a != b for a, b in zip(old, predict(scores, t), strict=True))
        text += f"; {flipped} predictions at {t:.2f} differ from the key"
    return text


def _short(name: str) -> str:
    return name[:5]


def _accuracy_by(rows: list[Row], pred: list[str], groups: list[str],
                 order: list[str]) -> list[str]:
    lines = []
    for group in order:
        idx = [i for i, g in enumerate(groups) if g == group]
        correct = sum(rows[i].gold == pred[i] for i in idx)
        low, high = wilson(correct, len(idx))
        lines.append(f"  {group:<22} {len(idx):>4} units  accuracy {correct / len(idx):.3f} "
                     f"({low:.3f}-{high:.3f})")
    return lines


def format_report(ev: Evaluation) -> str:
    lines: list[str] = []
    add = lines.append
    rows, pred = ev.rows, ev.pred
    gold = [row.gold for row in rows]

    add("=== Data ===")
    add(f"  {len(rows)} labeled units from {len({r.game for r in rows})} games, "
        f"{sum(r.ambiguous for r in rows)} flagged ambiguous. Threshold {ev.threshold:.2f}.")
    add(f"  Scores: {ev.source}.")
    counts = Counter(gold)
    add("  Gold labels: " + ", ".join(f"{c} {counts[c]}" for c in CLASSES))

    add("")
    add(f'=== Summary: model vs. majority-class baseline (always "{ev.majority}") ===')
    add(f"  {'rows':<20} {'n':>4} {'accuracy':>9} {'95% CI':>12} {'macro F1':>9} "
        f"{'base acc':>9} {'base macro F1':>14}")
    for name, m, b in (("all", ev.model, ev.baseline),
                       ("unambiguous", ev.unambiguous, ev.baseline_unambiguous)):
        low, high = wilson(m["correct"], m["n"])
        add(f"  {name:<20} {m['n']:>4} {m['accuracy']:>9.3f} {f'{low:.3f}-{high:.3f}':>12} "
            f"{m['macro_f1']:>9.3f} {b['accuracy']:>9.3f} {b['macro_f1']:>14.3f}")
    if ev.weighted and ev.baseline_weighted:
        w, b = ev.weighted, ev.baseline_weighted
        add(f"  {'population-weighted':<20} {w['n']:>4} {w['accuracy']:>9.3f} {'':>12} "
            f"{w['macro_f1']:>9.3f} {b['accuracy']:>9.3f} {b['macro_f1']:>14.3f}")
        add("  population-weighted: each unit counts for the units of its game and stratum it")
        add("  stands for, so the oversampled hard band stops dominating. An estimate for all")
        add("  cached sentences.")

    add("")
    add("=== Per aspect (all rows unless noted) ===")
    add(f"  {'aspect':<12} {'gold':>5} {'pred':>5} {'precision':>10} {'recall':>7} "
        f"{'F1':>6} {'F1 unamb':>9} {'F1 pop-wtd':>11}")
    for c in CLASSES:
        m = ev.model["classes"][c]
        wtd = f"{ev.weighted['classes'][c]['f1']:.3f}" if ev.weighted else "-"
        add(f"  {c:<12} {m['support']:>5} {m['predicted']:>5} {m['precision']:>10.3f} "
            f"{m['recall']:>7.3f} {m['f1']:>6.3f} "
            f"{ev.unambiguous['classes'][c]['f1']:>9.3f} {wtd:>11}")

    add("")
    add("=== Confusion matrix (rows: gold, columns: predicted) ===")
    matrix = confusion_matrix(gold, pred, labels=list(CLASSES))
    add(f"  {'':<12}" + "".join(f"{_short(c):>7}" for c in CLASSES))
    for c, counts_row in zip(CLASSES, matrix, strict=True):
        add(f"  {c:<12}" + "".join(f"{n:>7d}" for n in counts_row))

    add("")
    add("=== Accuracy by stratum (strata from sampling time) ===")
    strata = [s for s in STRATA if any(r.stratum == s for r in rows)]
    lines.extend(_accuracy_by(rows, pred, [r.stratum for r in rows], strata))
    add("")
    add("=== Accuracy by game ===")
    games = [r.game for r in rows]
    lines.extend(_accuracy_by(rows, pred, games, sorted(set(games))))

    tagged = [(i, r.note) for i, r in enumerate(rows) if r.gold == NONE_ASPECT and r.note]
    if tagged:
        add("")
        add("=== Gold none units by topic in notes (a missing aspect shows up here) ===")
        for note, n in Counter(note for _, note in tagged).most_common():
            got = Counter(pred[i] for i, t in tagged if t == note)
            add(f"  {note:<20} {n:>3}  predicted: "
                + ", ".join(f"{c} {k}" for c, k in got.most_common()))

    add("")
    add("=== Threshold sweep (same scores; * = this run) ===")
    add(f"  {'threshold':>9} {'accuracy':>9} {'macro F1':>9} {'none prec':>10} "
        f"{'none rec':>9} {'pred none':>10}")
    best = max(ev.sweep, key=lambda s: (s["macro_f1"], -abs(s["threshold"] - ev.threshold)))
    for s in ev.sweep:
        mark = "*" if s["threshold"] == ev.threshold else " "
        tail = "  <- best macro F1" if s is best else ""
        add(f" {mark}{s['threshold']:>9.2f} {s['accuracy']:>9.3f} {s['macro_f1']:>9.3f} "
            f"{s['none_precision']:>10.3f} {s['none_recall']:>9.3f} {s['none_share']:>10.1%}"
            f"{tail}")
    add(f"  Picking the threshold here tunes on these {len(rows)} units, so the numbers at "
        "the picked value are optimistic.")

    misses = sorted((i for i in range(len(rows)) if gold[i] != pred[i]),
                    key=lambda i: (gold[i], pred[i], rows[i].id))
    add("")
    add(f"=== Misses ({len(misses)}; ~ = ambiguous) ===")
    add(f"  {'id':>4}  {'gold -> predicted':<24} {'score':>6}  {'stratum':<10}  sentence")
    for i in misses:
        r = rows[i]
        text = r.sentence if len(r.sentence) <= SENTENCE_CHARS else (
            r.sentence[: SENTENCE_CHARS - 3] + "...")
        add(f"  {r.id:>4}{'~' if r.ambiguous else ' '} {f'{r.gold} -> {pred[i]}':<24} "
            f"{float(ev.scores[i].max()):>6.3f}  {r.stratum:<10}  {text}")

    warnings = []
    thin = [c for c in CLASSES if counts[c] < LOW_SUPPORT_WARNING]
    if thin:
        warnings.append(f"WARNING: fewer than {LOW_SUPPORT_WARNING} gold units for "
                        f"{', '.join(thin)}; their precision/recall swing a lot per unit.")
    if ev.weighted is None:
        warnings.append("WARNING: the meta file doesn't cover this sample, so there is no "
                        "population-weighted estimate.")
    elif ev.population_units and round(ev.covered_units or 0) != ev.population_units:
        warnings.append(f"WARNING: the weights stand for {ev.covered_units:.0f} of "
                        f"{ev.population_units} population units (a stratum with units but "
                        "none sampled), so the weighted estimate misses some.")
    if warnings:
        add("")
        lines.extend(warnings)
    return "\n".join(lines)


def write_report(report: str, path: Path) -> None:
    header = (f"Aspect tagging report, generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC by "
              "backend/scripts/evaluate_aspects.py\n\n")
    path.write_text(header + report + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD)
    parser.add_argument("--from-key", action="store_true",
                        help="use the scores stored in the key instead of re-scoring")
    args = parser.parse_args()

    meta = read_meta()
    try:
        rows = load_rows()
        scores = stored_scores(rows) if args.from_key else rescore(rows)
    except LabelError as exc:
        print(f"Cannot evaluate yet:\n{exc}")
        return 1
    source = score_source(rows, scores, meta, rescored=not args.from_key)
    report = format_report(evaluate(rows, scores, args.threshold, meta, source))
    print(report)
    write_report(report, REPORT_PATH)
    print(f"\nSaved report to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
