"""Build the hand-labeling sample for the aspect-tagging evaluation (M4).

Every cached English review is split and scored with `ml.aspects`, and a stratified sample
of sentence units is drawn:

- the same number of units from each game, however many sentences each game has;
- within a game, half the units come from the band within `BAND` of `SIMILARITY_THRESHOLD`
  (split evenly between just below and just above it), because that is where tagging is
  least reliable; the rest are long units (where glued clauses dilute the score) and clear
  cases spread round-robin over every predicted class, "none" included.

Two files are written so labeling stays blind to the model:

- `docs/eval/aspect_labels.csv`: id, game, sentence and empty gold_aspect / ambiguous /
  notes columns. This is the file to label (rules in `docs/eval/ASPECT_LABELING.md`).
- `docs/eval/aspect_sample_key.csv`: the model's side (prediction, scores, stratum),
  joined to the labels on id by `evaluate_aspects.py`.

plus `aspect_sample_meta.json` with the settings, population counts per stratum and the
hash of the labels file as written.

A rebuild refuses to run once the labels file differs from that hash (it may hold gold
labels); `--force` rebuilds anyway after copying the current three files to `*.bak-<time>`.

Run from backend/: `python scripts/build_eval_sample.py [--n 140] [--seed 42] [--force]`
"""

import argparse
import asyncio
import csv
import hashlib
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import numpy as np

from src.ml.anchors import ANCHORS, ASPECTS
from src.ml.aspects import label
from src.ml.constants import EMBEDDING_MODEL_NAME, NONE_ASPECT, SIMILARITY_THRESHOLD

EVAL_DIR = BACKEND_DIR.parent / "docs" / "eval"  # tracked in git: the gold labels live here
LABELS_PATH = EVAL_DIR / "aspect_labels.csv"
KEY_PATH = EVAL_DIR / "aspect_sample_key.csv"
META_PATH = EVAL_DIR / "aspect_sample_meta.json"

DEFAULT_N = 140
DEFAULT_SEED = 42
BAND = 0.10  # half-width of the near-threshold band that is oversampled
LONG_UNIT_WORDS = 25  # units this long often glue several topics together

STRATA = ("near_below", "near_above", "long", "other")
CLASSES = (*ASPECTS, NONE_ASPECT)  # every label a unit can get, in report order
LABEL_COLUMNS = ["id", "game", "sentence", "gold_aspect", "ambiguous", "notes"]
KEY_COLUMNS = [
    "id", "appid", "game", "review_id", "stratum", "sentence", "word_count",
    "predicted_aspect", "predicted_similarity", "runner_up_aspect", "margin",
    *(f"score_{a}" for a in ASPECTS),
]
# Excel and LibreOffice read a BOM-less UTF-8 CSV as a legacy codepage and mangle accents
# and emoji; reading with utf-8-sig accepts files with or without the BOM.
CSV_ENCODING = "utf-8-sig"


@dataclass
class Unit:
    """One sentence unit with its per-aspect scores (columns in `ASPECTS` order)."""

    appid: int
    game: str
    review_id: int
    sentence: str
    scores: np.ndarray

    @property
    def word_count(self) -> int:
        return len(self.sentence.split())

    @property
    def similarity(self) -> float:
        return float(self.scores.max())

    def predicted(self, threshold: float = SIMILARITY_THRESHOLD) -> str:
        return label(self.scores, threshold)[0]

    def runner_up(self) -> tuple[str, float]:
        """Second-best aspect and how far the best aspect's score is ahead of it."""
        order = np.argsort(-self.scores, kind="stable")
        return ASPECTS[order[1]], float(self.scores[order[0]] - self.scores[order[1]])


def band_bounds(threshold: float, band: float = BAND) -> tuple[float, float]:
    # Rounded so 0.40 - 0.10 is 0.30, not 0.30000000000000004.
    return round(threshold - band, 6), round(threshold + band, 6)


def stratum_of(unit: Unit, threshold: float = SIMILARITY_THRESHOLD) -> str:
    low, high = band_bounds(threshold)
    sim = unit.similarity
    if low <= sim < threshold:
        return "near_below"
    if threshold <= sim <= high:
        return "near_above"
    if unit.word_count >= LONG_UNIT_WORDS:
        return "long"
    return "other"


def quotas(per_game: int) -> dict[str, int]:
    """Per-game stratum quotas: 28 -> 7 just below, 7 just above, 4 long, 10 other."""
    near = per_game // 4
    long = per_game // 7
    return {"near_below": near, "near_above": near, "long": long,
            "other": per_game - 2 * near - long}


def normalize(sentence: str) -> str:
    return " ".join(sentence.lower().split())


def dedupe(units: list[Unit]) -> list[Unit]:
    """Keep the first unit for each normalized text, so repeats like "10/10" appear once."""
    seen: set[str] = set()
    kept = []
    for unit in units:
        key = normalize(unit.sentence)
        if key not in seen:
            seen.add(key)
            kept.append(unit)
    return kept


def _round_robin(units: list[Unit], k: int, rng: random.Random, threshold: float) -> list[Unit]:
    """Take k units cycling over predicted classes, so rare aspects still show up."""
    buckets: dict[str, list[Unit]] = defaultdict(list)
    for unit in units:
        buckets[unit.predicted(threshold)].append(unit)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    order = [c for c in CLASSES if buckets[c]]
    picked: list[Unit] = []
    while len(picked) < k and order:
        for cls in list(order):
            if len(picked) == k:
                break
            picked.append(buckets[cls].pop())
            if not buckets[cls]:
                order.remove(cls)
    return picked


@dataclass
class Sample:
    units: list[Unit]  # shuffled; ids are 1-based positions in this list
    strata: list[str]  # the true stratum of each unit (backfilled units keep theirs)
    shortfalls: dict[str, dict[str, int]]  # game -> stratum -> slots backfilled


def draw_sample(
    units: list[Unit], n: int, seed: int, threshold: float = SIMILARITY_THRESHOLD
) -> Sample:
    """Stratified sample: n // games units per game (remainder to the first games)."""
    rng = random.Random(seed)
    by_game: dict[str, list[Unit]] = defaultdict(list)
    for unit in dedupe(units):
        by_game[unit.game].append(unit)
    games = sorted(by_game)
    if not games:
        return Sample([], [], {})

    picked: list[Unit] = []
    shortfalls: dict[str, dict[str, int]] = {}
    for i, game in enumerate(games):
        per_game = n // len(games) + (1 if i < n % len(games) else 0)
        pools: dict[str, list[Unit]] = defaultdict(list)
        for unit in by_game[game]:
            pools[stratum_of(unit, threshold)].append(unit)

        chosen: list[Unit] = []
        short: dict[str, int] = {}
        for stratum, k in quotas(per_game).items():
            pool = pools[stratum]
            if stratum == "other":
                take = _round_robin(pool, k, rng, threshold)
            else:
                take = rng.sample(pool, min(k, len(pool)))
            if len(take) < k:
                short[stratum] = k - len(take)
            chosen.extend(take)

        missing = per_game - len(chosen)
        if missing:
            taken = {id(u) for u in chosen}
            leftover = [u for u in by_game[game] if id(u) not in taken]
            chosen.extend(rng.sample(leftover, min(missing, len(leftover))))
        if short:
            shortfalls[game] = short
        picked.extend(chosen)

    rng.shuffle(picked)
    return Sample(picked, [stratum_of(u, threshold) for u in picked], shortfalls)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def labels_edited(labels_path: Path = LABELS_PATH, meta_path: Path = META_PATH) -> bool:
    """Whether the labels file may hold hand edits that a rebuild would destroy.

    It counts as untouched only if it is byte-identical to the file this script last wrote
    (hash kept in the meta file). Looking for a filled gold_aspect column instead would miss
    labels saved with another delimiter (Excel in many locales uses ";") or renamed columns.
    """
    if not labels_path.exists():
        return False
    try:
        recorded = json.loads(meta_path.read_text(encoding="utf-8")).get("labels_sha256")
    except (OSError, ValueError, AttributeError):
        recorded = None
    return recorded != file_sha256(labels_path)


def back_up(paths: list[Path]) -> list[Path]:
    """Copy each existing file to `<stem>.bak-<UTC time><suffix>` next to it."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backups = []
    for path in paths:
        if path.exists():
            backup = path.with_name(f"{path.stem}.bak-{stamp}{path.suffix}")
            shutil.copy2(path, backup)
            backups.append(backup)
    return backups


def _num(x: float) -> str:
    """Full precision (repr round-trips), so re-labeling from the key at any threshold gives
    the same answer as the model; 4 decimals would turn 0.39996 into 0.4000."""
    return repr(float(x))


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding=CSV_ENCODING) as f:
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        writer.writerows(rows)


def anchors_hash() -> str:
    return hashlib.sha256(json.dumps(ANCHORS, sort_keys=True).encode()).hexdigest()[:16]


def write_sample(
    sample: Sample,
    population: list[Unit],
    seed: int,
    labels_path: Path = LABELS_PATH,
    key_path: Path = KEY_PATH,
    meta_path: Path = META_PATH,
    threshold: float = SIMILARITY_THRESHOLD,
) -> None:
    """Write the labels, key and meta files.

    Everything goes to `*.tmp` files first and replaces the real files only once all three
    are written, so a failure part-way leaves the old labels/key pair intact rather than a
    new labels file next to an old key.
    """
    label_rows = [{"id": i, "game": unit.game, "sentence": unit.sentence,
                   "gold_aspect": "", "ambiguous": "", "notes": ""}
                  for i, unit in enumerate(sample.units, 1)]
    key_rows = []
    for i, (unit, stratum) in enumerate(zip(sample.units, sample.strata, strict=True), 1):
        runner_up, margin = unit.runner_up()
        key_rows.append({
            "id": i, "appid": unit.appid, "game": unit.game, "review_id": unit.review_id,
            "stratum": stratum, "sentence": unit.sentence, "word_count": unit.word_count,
            "predicted_aspect": unit.predicted(threshold),
            "predicted_similarity": _num(unit.similarity),
            "runner_up_aspect": runner_up, "margin": _num(margin),
            **{f"score_{a}": _num(s) for a, s in zip(ASPECTS, unit.scores, strict=True)},
        })

    targets = [key_path, labels_path, meta_path]  # replaced in this order, meta last
    tmps = {path: path.with_name(f"{path.name}.tmp") for path in targets}
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _write_csv(tmps[labels_path], LABEL_COLUMNS, label_rows)
        _write_csv(tmps[key_path], KEY_COLUMNS, key_rows)
        meta = _meta(sample, population, seed, threshold, file_sha256(tmps[labels_path]))
        tmps[meta_path].write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        for path in targets:
            os.replace(tmps[path], path)
    finally:
        for tmp in tmps.values():
            tmp.unlink(missing_ok=True)


def _meta(
    sample: Sample, population: list[Unit], seed: int, threshold: float, labels_sha256: str
) -> dict:
    deduped = dedupe(population)
    return {
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": seed,
        "n": len(sample.units),
        "threshold": threshold,
        "band": list(band_bounds(threshold)),
        "long_unit_words": LONG_UNIT_WORDS,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "anchors_sha256": anchors_hash(),
        "population_units": len(population),
        "population_units_deduped": len(deduped),
        "population_strata": _counts_by_game(deduped, [stratum_of(u, threshold) for u in deduped]),
        "sample_strata": _counts_by_game(sample.units, sample.strata),
        "shortfalls": sample.shortfalls,
        "labels_sha256": labels_sha256,
    }


def _counts_by_game(units: list[Unit], strata: list[str]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for unit, stratum in zip(units, strata, strict=True):
        counts[unit.game][stratum] += 1
    return {g: {s: counts[g][s] for s in STRATA} for g in sorted(counts)}


def format_summary(sample: Sample, population: list[Unit], threshold: float) -> str:
    deduped = dedupe(population)
    pop = _counts_by_game(deduped, [stratum_of(u, threshold) for u in deduped])
    got = _counts_by_game(sample.units, sample.strata)
    low, high = band_bounds(threshold)
    lines = [
        (f"Population: {len(population)} units, {len(deduped)} after dedupe. "
         f"Threshold {threshold:.2f}, band [{low:.2f}, {high:.2f}]."),
        f"  {'game':<22} " + " ".join(f"{s:>16}" for s in STRATA) + f" {'total':>6}",
    ]
    for game in sorted(pop):
        cells = [f"{got.get(game, {}).get(s, 0):>5} / {pop[game][s]:<8}" for s in STRATA]
        lines.append(f"  {game:<22} " + " ".join(f"{c:>16}" for c in cells)
                     + f" {sum(got.get(game, {}).values()):>6}")
    lines.append("  (cells: sampled / population)")
    predicted = Counter(u.predicted(threshold) for u in sample.units)
    lines.append("Sample by predicted class: "
                 + ", ".join(f"{c} {predicted[c]}" for c in CLASSES))
    for game, short in sample.shortfalls.items():
        lines.append(f"NOTE: {game} ran short in {short}; backfilled from its other units.")
    return "\n".join(lines)


async def load_reviews() -> list[tuple[int, str, int, str]]:
    """(appid, game name, review id, text) for every cached, non-empty English review."""
    from sqlalchemy import func, select

    from src.database import SessionLocal, engine
    from src.games.models import Game
    from src.reviews.constants import STEAM_LANGUAGE
    from src.reviews.models import Review

    async with SessionLocal() as session:
        rows = await session.execute(
            select(Game.appid, Game.name, Review.id, Review.review_text)
            .join(Game, Game.appid == Review.appid)
            .where(Review.language == STEAM_LANGUAGE)
            .where(func.trim(Review.review_text) != "")
            .order_by(Review.id)
        )
        reviews = [tuple(row) for row in rows]
    await engine.dispose()
    return reviews


def score_population(reviews: list[tuple[int, str, int, str]]) -> list[Unit]:
    from src.ml import aspects

    units: list[Unit] = []
    done: Counter = Counter()
    for appid, game, review_id, text in reviews:
        sentences, scores = aspects.score_aspects(text)
        units.extend(Unit(appid, game, review_id, s, row)
                     for s, row in zip(sentences, scores, strict=True))
        done[game] += 1
        if done[game] % 250 == 0:
            print(f"  scored {done[game]} reviews of {game}", flush=True)
    return units


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="sample size")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true",
                        help="rebuild even though the labels file was edited (backs it up)")
    args = parser.parse_args()

    edited = labels_edited()
    if edited and not args.force:
        print(f"Refusing to overwrite {LABELS_PATH}: it has changed since this script wrote "
              "it, so it may hold gold labels. Pass --force to rebuild anyway (the current "
              "files are backed up first).")
        return 1

    reviews = asyncio.run(load_reviews())
    if not reviews:
        print("No cached English reviews found. Ingest a game first (GET /games/{appid}/reviews).")
        return 1
    print(f"Scoring {len(reviews)} reviews...", flush=True)
    population = score_population(reviews)
    sample = draw_sample(population, args.n, args.seed)
    if edited:
        for backup in back_up([LABELS_PATH, KEY_PATH, META_PATH]):
            print(f"Backed up to {backup}")
    write_sample(sample, population, args.seed)

    print(format_summary(sample, population, SIMILARITY_THRESHOLD))
    print(f"\nLabel this file: {LABELS_PATH}")
    print(f"Model key (don't open while labeling): {KEY_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
