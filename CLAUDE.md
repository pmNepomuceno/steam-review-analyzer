# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

Milestones 1-7 are done. Backend: on-demand ingestion (M1), the sentiment baseline (M2), aspect tagging (M3), the aspect eval against the user's 140 hand labels (M4; the anchors were tuned on that same set, so quote its numbers as optimistic) and aggregation + API (M5). M5 stores per-unit results in `review_aspects` and each review's sentiment in `reviews.predicted_sentiment` through `reviews/service.py::process_reviews`, tracked by `games.aspects_status` (pending/processing/done/failed, reason in `aspects_error`). That function is run by `scripts/process_reviews.py` or by a background task when `/aspects` (or a filtered `/reviews`) finds a game pending, which answers 202 `{status: "processing"}`; a failed game answers 500 `{status: "failed"}` and is only retried by `--force`; with no trained model they answer 503. Gold labels in `docs/eval/aspect_labels.csv` are written by the user by hand, never generated, filled in or "fixed" by Claude. After changing `ANCHORS`, `SIMILARITY_THRESHOLD` or the sentiment model, rerun `python scripts/evaluate_aspects.py` and `python scripts/process_reviews.py --force`. M6 (dashboard) is done: `frontend/` is a Next.js app that polls `/aspects` until ready, with one screen per backend state, and renders the aspect chart, the daily trend (`/aspects` `trend`) and the filterable review list (`/reviews` items carry `predicted_sentiment` and `aspects`). The review sample is deliberately the ~1000 most recent reviews, labelled in the UI. M7 (deploy prep) is done: `render.yaml` (API as a Docker web service from `backend/`, `DATABASE_URL` and `DATABASE_URL_DIRECT` entered in Render's dashboard, never committed), production Dockerfiles for both apps, the committed sentiment model, `scripts/check_memory.py` (peak RSS measured in the production image: 354-411 MB across the 5 cached games, 80% of Render's 512 MB free tier at worst, and 442 MB (86%) for the synthetic worst case of full 256-token batches; this needs the aspect encoder on ONNX Runtime instead of torch and `ENCODE_BATCH_SIZE = 8`. Earlier figures of 358/351 MB came from short synthetic sentences and understated real reviews, so measure memory with real review lengths), and README.md as the portfolio write-up with the measured eval numbers and known limitations.

The deployed API runs with `ALLOW_ON_DEMAND_PROCESSING=false` (`render.yaml`): at Render's 0.1 CPU one game takes 14-19 min, so the demo serves only games already `done` and answers 403 with the list of them for any other appid. Locally the setting defaults to `true`. Encoder batching across reviews is logged in DECISIONS.md as not attempted; don't start it unasked.

**Deployed:** the user deployed by hand. The API runs on Render (Singapore), the dashboard on Vercel (https://steam-review-analyzer-wheat.vercel.app/) and the data on Neon, loaded from the five local games with the README's `pg_dump --data-only` procedure (the real history of the first load, including a failed first attempt, is in DECISIONS.md, 2026-10-01). README.md's live URL is filled in; the demo clip link is still a placeholder the user fills in after recording it. Never deploy or push from here. `docs/PROJECT_BRIEF.md` is the source of truth for scope, data model, endpoints and milestones; `docs/DECISIONS.md` logs choices made while building.

## Commands

Backend commands run from `backend/` inside the uv venv (Python 3.12, not the system 3.14):

A trained sentiment model is committed, so a fresh clone serves `/aspects` without training. To train from scratch the order is ingest, train, serve, done by hand: start the API, which runs without a sentiment model (it loads lazily on first use), request `/games/{appid}/reviews` for a few games to cache them, run `python scripts/train_sentiment.py`, and then `/aspects` works without a restart (503 before that). Retraining needs an API restart to pick up the new model.

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
python scripts/build_eval_sample.py      # M4 aspect eval sample -> docs/eval/aspect_labels.csv (hand-label, blind) + aspect_sample_key.csv; refuses to run once the labels file has been edited, unless --force (which backs up the old files first)
python scripts/evaluate_aspects.py      # score gold labels -> docs/eval/aspect_report.txt; re-scores with current ANCHORS; --threshold 0.36 to try another value, --from-key to use stored scores
python scripts/process_reviews.py        # sentiment + aspect tags -> review_aspects + reviews.predicted_sentiment for every ingested game (or pass appids); skips done/failed/in-progress games and says why, --force redoes them
python scripts/check_memory.py [APPID]   # peak RSS with the real app + both models, vs Render's 512 MB: synthetic worst case (full 256-token batches), or one game's cached reviews
alembic revision --autogenerate -m "..." # after changing models
```

Frontend commands run from `frontend/` (Node 22+; the API must be running for `dev`):

```bash
npm install
npm run dev          # dashboard on :3000; /api/* is proxied to API_URL (default http://localhost:8000)
npm test             # vitest, state logic in lib/api.ts
npm run lint         # eslint .
npm run typecheck    # tsc --noEmit
npm run build
```

Tests create and use a separate `steam_reviews_test` database on the same server (see `tests/conftest.py`).

## What the project is

A web app that takes any Steam appid, fetches and caches its reviews in Postgres, and reports sentiment broken down by aspect (performance, price, bugs, story, gameplay, or none). It is a portfolio piece meant to show a trained classifier and an embedding-based pipeline, so **do not replace the ML components with LLM API calls**.

## Architecture

- **Backend:** FastAPI, domain-driven layout following zhanymkanov/fastapi-best-practices. Each domain package under `backend/src/` (`games/`, `reviews/`, `ml/`) has its own `router.py`, `schemas.py`, `models.py`, `service.py`, `dependencies.py`, `constants.py` and `exceptions.py`. Global concerns live at the top of `backend/src/` (`main.py`, `config.py`, `database.py`, `models.py`, `exceptions.py`). Keep new code inside its domain rather than in type-based folders.
- **ML pipeline (`backend/src/ml/`):**
  - `sentiment.py`: TF-IDF + logistic regression trained on Steam's own `voted_up` flag as a weak-supervision label. Trained by `backend/scripts/train_sentiment.py`. The artifact is `backend/models_store/sentiment.joblib` and is committed (Render's free tier can't train or keep one), so retraining changes a tracked file. `scikit-learn` is pinned in `requirements/base.txt` to the version that pickled it; bump both together.
  - `aspects.py` and `anchors.py`: reviews are split into sentence/clause units (regex), embedded with `all-MiniLM-L6-v2` on CPU (its ONNX export on onnxruntime + `tokenizers`, no torch; `_encode` does the mean pooling and normalization, and the revision is pinned in `ml/constants.py`), and assigned to an aspect by max cosine similarity to hand-written anchor phrases (or to "none" below a threshold). Anchor-based assignment is deliberately chosen over BERTopic or a fine-tuned classifier. The two tunables live in one place each: `ANCHORS` in `anchors.py` and `SIMILARITY_THRESHOLD` in `ml/constants.py`. The model loads lazily on first call (first run downloads the ~90 MB ONNX file to the HF cache, which `tests/test_aspects.py` also needs).
  - `backend/scripts/evaluate_aspects.py` scores the aspect assignment against 100–150 hand-labeled sentences. This is the mitigation for the project's riskiest assumption, so changes to anchors or thresholds should be checked against it.
- **Ingestion:** lazy and on-demand, split across two files. `reviews/ingestion.py` is the Steam HTTP client (cursor pagination, retry on 429/5xx). `reviews/service.py::ensure_ingested` orchestrates: it takes a Postgres advisory lock per appid so concurrent first requests fetch once, fetches, applies the min-review guard (`MIN_REVIEW_COUNT`, default 200; a failing game is not persisted or marked ingested), then commits in one transaction. `games.last_ingested_at` is the "already cached" marker. Domain errors subclass `src.exceptions.AppError` (with a `status_code`) and are rendered by one handler in `main.py`.
- **Data model:** `games` (appid, name, last_ingested_at, aspects_status, aspects_error, aspects_processed_at), `reviews` (id, appid FK, review_text, voted_up, votes_up, playtime_forever, language, created_at, predicted_sentiment), `review_aspects` (review_id FK, sentence_text, aspect_label, similarity_score).
- **API:** `GET /games/{appid}/aspects` (aggregated per-aspect sentiment) and `GET /games/{appid}/reviews` (filterable by aspect and sentiment).
- **Frontend:** Next.js app router. `app/page.tsx` is the game selector and `app/games/[appid]/page.tsx` is the per-game dashboard: `Dashboard.tsx` (polling and state screens), `charts.tsx` (Recharts aspect and trend charts), `ReviewList.tsx` (filters and page in the URL query). `lib/api.ts` holds the response types and the HTTP-status-to-UI-state mapping (`fetchState`, `poll`), which is what `lib/api.test.ts` tests.

- **Deploy:** `backend/Dockerfile` runs `alembic upgrade head` (on `DATABASE_URL_DIRECT` when set) and then uvicorn on `${PORT:-8000}`. Its `WORKDIR` must stay `/app/backend`, because `src/config.py` finds the repo root two levels above `src/`. The MiniLM ONNX file is baked into the image; the Dockerfile reads name, revision and files from `EMBEDDING_MODEL_*` in `src/ml/constants.py`, so change them only there. On startup (best effort: a DB error there is logged, not fatal) and whenever a request finds a game `processing`, the API marks it `failed` (`fail_interrupted_runs`), but only if its run lock (a Postgres advisory lock `process_reviews` holds for the whole run) is free, so a CLI run against the same database is left alone. `_analyze` runs are serialized per process to keep the measured memory bound, and API background runs queue on `_background_lock` before taking any connection (a queued game stays `pending`). Encoder threads follow the CPU affinity and the container's cgroup CPU quota, whichever is lower (at most 8, `ENCODER_THREADS` overrides and must be >= 1), with onnxruntime spinning off; the tokenizer's Rayon pool gets the same cap through `RAYON_NUM_THREADS`, and the Dockerfile sets `OPENBLAS_NUM_THREADS=1` for numpy. `frontend/Dockerfile` needs `API_URL` as a build arg and fails without it, since Next bakes rewrites at build time.

## Scope constraints

English-only reviews, no auth or user accounts, no streaming ingestion, no review-bomb filtering. Transformer fine-tuning is a stretch goal, not committed scope. Hosting targets are free tiers only (Neon/Supabase, Render/Fly, Vercel).

## Docs

`docs/PROJECT_BRIEF.md` has a decisions log (section 10). `docs/DECISIONS.md` continues that log; record new decisions there with the date and reason.
