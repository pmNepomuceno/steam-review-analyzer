"""Run sentiment + aspect tagging over cached reviews; store in `review_aspects` and `reviews`.

The API does the same thing in the background for an app it finds unprocessed; this
script is for doing it ahead of time, and for redoing it after anchors or the sentiment
model change (`--force`). Already-processed apps are skipped, so reruns are cheap; each
skip says why. Failed apps stay failed until `--force`. One app failing doesn't stop the
rest, but makes the exit status 1.

Run from `backend/`: `python scripts/process_reviews.py [APPID ...] [--force]`
With no appids, every ingested game is processed.
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


async def run(appids: list[int], force: bool) -> int:
    """Process each app in its own session; returns how many failed."""
    from sqlalchemy import select

    from src.database import SessionLocal, engine
    from src.games.models import Game
    from src.reviews.service import process_reviews

    if not appids:
        async with SessionLocal() as session:
            appids = list(
                await session.scalars(
                    select(Game.appid).where(Game.last_ingested_at.is_not(None)).order_by(Game.appid)
                )
            )
    failures = 0
    for appid in appids:
        start = time.perf_counter()
        try:
            async with SessionLocal() as session:
                outcome = await process_reviews(session, appid, force=force)
        except Exception:
            logging.getLogger(__name__).exception("Processing appid %d failed", appid)
            failures += 1
            continue
        elapsed = time.perf_counter() - start
        status = f"processed {outcome} reviews" if isinstance(outcome, int) else f"skipped: {outcome}"
        print(f"{appid}: {status} in {elapsed:.1f}s")
    await engine.dispose()
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("appids", nargs="*", type=int, help="default: every ingested game")
    parser.add_argument(
        "--force",
        action="store_true",
        help="also redo processed games, retry failed ones and take over stuck ones",
    )
    args = parser.parse_args()
    logging.basicConfig()

    from src.ml import aspects, sentiment

    sentiment.load_model()
    aspects.load()
    return 1 if asyncio.run(run(args.appids, args.force)) else 0


if __name__ == "__main__":
    sys.exit(main())
