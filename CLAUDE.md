# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

Milestones 1 (on-demand ingestion), 2 (sentiment baseline: `scripts/train_sentiment.py` + `src/ml/sentiment.py`) and 3 (aspect tagging: `src/ml/anchors.py` + `src/ml/aspects.py::assign_aspects`, standalone and not yet persisted to `review_aspects`) are implemented in the backend. `scripts/evaluate_aspects.py` and the whole `frontend/` are still empty scaffolding for later milestones. `docs/PROJECT_BRIEF.md` is the source of truth for scope, data model, endpoints and milestones; `docs/DECISIONS.md` logs choices made while building.

## Commands

Backend commands run from `backend/` inside the uv venv (Python 3.12, not the system 3.14):

```bash
docker compose up -d db                  # from repo root; Postgres on host port 5433 (5432 is often taken)
cp .env.example .env                     # once; settings are read from the repo-root .env
cd backend && source .venv/bin/activate  # create with: uv venv --python 3.12 .venv && uv pip install -r requirements/dev.txt
alembic upgrade head
uvicorn src.main:app --reload            # API on :8000
pytest -q                                # needs the db container up; Steam is mocked with respx
pytest tests/test_reviews_api.py::test_pagination_offset   # single test
ruff check .
python scripts/train_sentiment.py        # retrain from cached reviews -> models_store/sentiment.joblib + docs/eval/sentiment_report.txt
alembic revision --autogenerate -m "..." # after changing models
```

Tests create and use a separate `steam_reviews_test` database on the same server (see `tests/conftest.py`).

## What the project is

A web app that takes any Steam appid, fetches and caches its reviews in Postgres, and reports sentiment broken down by aspect (performance, price, bugs, story, gameplay, or none). It is a portfolio piece meant to show a trained classifier and an embedding-based pipeline, so **do not replace the ML components with LLM API calls**.

## Architecture (planned, per the brief)

- **Backend:** FastAPI, domain-driven layout following zhanymkanov/fastapi-best-practices. Each domain package under `backend/src/` (`games/`, `reviews/`, `ml/`) has its own `router.py`, `schemas.py`, `models.py`, `service.py`, `dependencies.py`, `constants.py` and `exceptions.py`. Global concerns live at the top of `backend/src/` (`main.py`, `config.py`, `database.py`, `models.py`, `exceptions.py`). Keep new code inside its domain rather than in type-based folders.
- **ML pipeline (`backend/src/ml/`):**
  - `sentiment.py`: TF-IDF + logistic regression trained on Steam's own `voted_up` flag as a weak-supervision label. Trained by `backend/scripts/train_sentiment.py`. Artifacts go in `backend/models_store/`, which is gitignored (only `.gitkeep` is tracked), so models must be regenerated rather than committed.
  - `aspects.py` and `anchors.py`: reviews are split into sentence/clause units (regex), embedded with `all-MiniLM-L6-v2` on CPU, and assigned to an aspect by max cosine similarity to hand-written anchor phrases (or to "none" below a threshold). Anchor-based assignment is deliberately chosen over BERTopic or a fine-tuned classifier. The two tunables live in one place each: `ANCHORS` in `anchors.py` and `SIMILARITY_THRESHOLD` in `ml/constants.py`. The model loads lazily on first call (first run downloads ~90 MB to the HF cache, which `tests/test_aspects.py` also needs).
  - `backend/scripts/evaluate_aspects.py` scores the aspect assignment against 100–150 hand-labeled sentences. This is the mitigation for the project's riskiest assumption, so changes to anchors or thresholds should be checked against it.
- **Ingestion:** lazy and on-demand, split across two files. `reviews/ingestion.py` is the Steam HTTP client (cursor pagination, retry on 429/5xx). `reviews/service.py::ensure_ingested` orchestrates: it takes a Postgres advisory lock per appid so concurrent first requests fetch once, fetches, applies the min-review guard (`MIN_REVIEW_COUNT`, default 200; a failing game is not persisted or marked ingested), then commits in one transaction. `games.last_ingested_at` is the "already cached" marker. Domain errors subclass `src.exceptions.AppError` (with a `status_code`) and are rendered by one handler in `main.py`.
- **Data model:** `games` (appid, name, last_ingested_at), `reviews` (id, appid FK, review_text, voted_up, votes_up, playtime_forever, language, created_at), `review_aspects` (review_id FK, sentence_text, aspect_label, similarity_score, predicted_sentiment).
- **API:** `GET /games/{appid}/aspects` (aggregated per-aspect sentiment) and `GET /games/{appid}/reviews` (filterable by aspect and sentiment).
- **Frontend:** Next.js app router. `app/page.tsx` is the game selector and `app/games/[appid]/page.tsx` is the per-game dashboard (aspect chart, sentiment trend, filterable review list).

## Scope constraints

English-only reviews, no auth or user accounts, no streaming ingestion, no review-bomb filtering. Transformer fine-tuning is a stretch goal, not committed scope. Hosting targets are free tiers only (Neon/Supabase, Render/Fly, Vercel).

## Docs

`docs/PROJECT_BRIEF.md` has a decisions log (section 10). `docs/DECISIONS.md` exists but is empty; record new architectural decisions there or in the brief's log with the date and reason.
