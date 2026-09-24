# Steam Review Sentiment + Aspect Analyzer

Pulls Steam reviews for any game and (in later milestones) breaks sentiment down by aspect. See `docs/PROJECT_BRIEF.md` for scope and milestones.

**Status:** backend Milestones 1-5 done. `GET /games/{appid}/reviews` fetches and caches a game's English reviews in Postgres on first request; `GET /games/{appid}/aspects` reports sentiment per aspect once the reviews have been analyzed.

## Setup

Requires Docker and [uv](https://docs.astral.sh/uv/).

```bash
cp .env.example .env
docker compose up -d db            # Postgres on localhost:5433

cd backend
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements/dev.txt
alembic upgrade head
uvicorn src.main:app --reload
```

### First-time order: ingest, train, serve

The sentiment model is trained on cached reviews, and only the running API caches them, so a fresh setup (models are not committed) goes in this order. It is a manual step:

1. **Ingest.** With the API running (no model needed yet), request a few games' reviews, e.g. `curl "localhost:8000/games/1145360/reviews?limit=1"`. Include games with mixed ratings so the model sees enough negative reviews.
2. **Train.** `python scripts/train_sentiment.py` writes `models_store/sentiment.joblib`.
3. **Serve.** `/aspects` and filtered `/reviews` now work. The API loads the model on first use, so no restart is needed. Until then they answer 503. After retraining, restart the API to pick up the new model, and run `python scripts/process_reviews.py --force` to redo stored results.

## Try it

```bash
curl "localhost:8000/games/1145360/reviews?limit=3"   # Hades: first call ingests (~10s), later calls are instant
```

| Status | Meaning |
|---|---|
| 404 | Steam has no app with that appid |
| 422 | Fewer than `MIN_REVIEW_COUNT` (200) usable English reviews, or invalid appid |
| 502 | Steam unavailable or rate-limiting |

`/aspects`, and `/reviews` with an `aspect` or `sentiment` filter, also answer:

| Status | Meaning |
|---|---|
| 202 `{status: "processing"}` | Reviews are being analyzed (20-60s on first request); retry shortly |
| 500 `{status: "failed"}` | Analysis failed; the reason is in `games.aspects_error` and the server log. Retry with `python scripts/process_reviews.py --force APPID` |
| 503 | No sentiment model trained yet (see the setup order above) |

## Tests and lint

```bash
cd backend && source .venv/bin/activate
pytest -q        # needs the db container running; Steam is mocked
ruff check .
```

Settings live in `.env` (`DATABASE_URL`, `MIN_REVIEW_COUNT`, `MAX_REVIEWS`, `STEAM_REQUEST_DELAY_S`).
