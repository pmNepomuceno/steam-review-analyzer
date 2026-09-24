# Steam Review Sentiment + Aspect Analyzer

Pulls Steam reviews for any game and (in later milestones) breaks sentiment down by aspect. See `docs/PROJECT_BRIEF.md` for scope and milestones.

**Status:** Milestone 1 (on-demand ingestion). `GET /games/{appid}/reviews` fetches and caches a game's English reviews in Postgres on first request, then serves them from the cache.

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

## Try it

```bash
curl "localhost:8000/games/1145360/reviews?limit=3"   # Hades: first call ingests (~10s), later calls are instant
```

| Status | Meaning |
|---|---|
| 404 | Steam has no app with that appid |
| 422 | Fewer than `MIN_REVIEW_COUNT` (200) usable English reviews, or invalid appid |
| 502 | Steam unavailable or rate-limiting |

## Tests and lint

```bash
cd backend && source .venv/bin/activate
pytest -q        # needs the db container running; Steam is mocked
ruff check .
```

Settings live in `.env` (`DATABASE_URL`, `MIN_REVIEW_COUNT`, `MAX_REVIEWS`, `STEAM_REQUEST_DELAY_S`).
