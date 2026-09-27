"""Report the API process's peak memory once both ML models are loaded and used.

Loads what the running API loads, the same way: imports `src.main`, loads the sentiment
artifact and the sentence encoder with the functions the request path uses, then runs the
background processing step (`reviews.service._analyze`). Prints peak RSS after each stage,
so the larger contributor is visible, and compares the total with Render's 512 MB free-tier
limit. Needs a trained sentiment model.

Two inputs:
- no argument: a synthetic worst case. Every review fills one encoder batch
  (`ENCODE_BATCH_SIZE` units) with units at the full `MAX_SEQ_LENGTH` tokens. Encoder
  activations scale with batch size x sequence length, so this is the largest batch the
  encoder can see. Short synthetic sentences understate the peak: real reviews contain
  units of up to 256 tokens (see docs/DECISIONS.md). Needs `DATABASE_URL` set (the engine
  is created on import) but never connects.
- `APPID`: that game's cached reviews, read from the database, as a production run
  processes them. Run it once per game; peak RSS is per process.

Run from `backend/`: `python scripts/check_memory.py [APPID]`
"""

import asyncio
import resource
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

LIMIT_MB = 512  # Render free tier
# Text without spaces is cut at token boundaries (aspects._fit_to_encoder), so each segment
# yields one unit at the full token limit plus a short remainder. At ~10 wordpieces per
# repeat, 26 repeats exceed the 254 content tokens a unit can hold.
FULL_SEGMENT = "このゲームは最高です" * 26


def peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KiB on Linux


async def cached_reviews(appid: int) -> list[tuple[int, str]]:
    from sqlalchemy import select

    from src.database import SessionLocal, engine
    from src.reviews.models import Review

    async with SessionLocal() as session:
        result = await session.execute(
            select(Review.id, Review.review_text).where(Review.appid == appid)
        )
        reviews = list(result.all())
    await engine.dispose()
    return reviews


def worst_case_reviews(n_reviews: int) -> list[tuple[int, str]]:
    from src.ml import aspects
    from src.ml.constants import ENCODE_BATCH_SIZE, MAX_SEQ_LENGTH

    review = ". ".join([FULL_SEGMENT] * ENCODE_BATCH_SIZE)
    units = aspects._fit_to_encoder(aspects.split_sentences(review))
    full = sum(len(aspects._fit_tokenizer.encode(u).ids) == MAX_SEQ_LENGTH for u in units)
    assert full >= ENCODE_BATCH_SIZE, f"only {full} full-length units per review"
    return [(i, review) for i in range(n_reviews)]


def main() -> None:
    stages: list[tuple[str, float]] = [("python start", peak_mb())]

    import src.main  # noqa: F401  the app, its routers and the DB engine
    from src.config import settings

    stages.append(("import src.main", peak_mb()))

    from src.ml import aspects, sentiment
    from src.reviews.service import _analyze

    sentiment.ensure_loaded()
    stages.append(("sentiment model (TF-IDF + LogReg)", peak_mb()))
    aspects.load()
    stages.append(("sentence encoder (onnxruntime + MiniLM)", peak_mb()))

    if len(sys.argv) > 1:
        appid = int(sys.argv[1])
        reviews = asyncio.run(cached_reviews(appid))
        if not reviews:
            sys.exit(f"no cached reviews for appid {appid}")
        label = f"processing {len(reviews)} reviews of {appid}"
    else:
        reviews = worst_case_reviews(settings.max_reviews)  # what one production run processes
        label = f"processing {len(reviews)} worst-case reviews"
    _analyze(reviews)
    stages.append((label, peak_mb()))

    print(f"{'stage':44} {'peak RSS':>10} {'delta':>9}")
    previous = 0.0
    for name, mb in stages:
        print(f"{name:44} {mb:7.0f} MB {mb - previous:+6.0f} MB")
        previous = mb
    total = stages[-1][1]
    verdict = "OVER" if total > LIMIT_MB else "under"
    print(f"\nPeak {total:.0f} MB: {verdict} the {LIMIT_MB} MB limit ({total / LIMIT_MB:.0%}).")
    largest = max(zip(stages[1:], stages), key=lambda p: p[0][1] - p[1][1])[0][0]
    print(f"Largest contributor: {largest}.")


if __name__ == "__main__":
    main()
