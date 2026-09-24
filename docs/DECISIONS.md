# Decisions

Extends the decisions log in `PROJECT_BRIEF.md` section 10.

| Date | Decision | Reason |
|---|---|---|
| 2026-09-24 | Dev Postgres runs in Docker Compose on host port 5433 | Port 5432 is commonly taken by other local Postgres containers |
| 2026-09-24 | Backend pinned to Python 3.12 (uv venv, `python:3.12-slim` image) | System Python is 3.14, and ML wheels needed in M2-M3 may not support it yet |
| 2026-09-24 | Minimum 200 usable English reviews before serving a game (`MIN_REVIEW_COUNT`) | Enough for meaningful per-aspect breakdowns without excluding most indie games. A game that fails the guard is not marked ingested, so it can succeed on a later request |
| 2026-09-24 | Cap ingestion at 1000 reviews per game (`MAX_REVIEWS`) | Bounds first-request latency (about 10s for 1000 reviews) |
| 2026-09-24 | Serialize concurrent first requests with `pg_advisory_xact_lock(1, appid)` | Prevents duplicate Steam fetches without adding a job queue |
| 2026-09-24 | Ingest inside the request (blocking first call), no background jobs | Brief scopes v1 to fetch-and-cache; revisit if latency hurts the dashboard |
| 2026-09-24 | Added Starfield, Cities: Skylines II and PAYDAY 3 (Mixed ratings) to the dev data alongside Portal 2 and Hades | Portal 2 + Hades had only 47 negatives in 2000 reviews, too few to train or evaluate the minority class; the five games give 1097 negatives in 5000 |
| 2026-09-24 | Sentiment model trains on all cached games pooled, with `class_weight="balanced"` | One model serves any appid; balancing stops LogReg from winning by predicting "positive" everywhere |
| 2026-09-24 | Vectorizer + classifier saved as one sklearn `Pipeline` in `models_store/sentiment.joblib` | One artifact can't drift out of sync between vectorizer and model |
| 2026-09-24 | `ml/sentiment.py` loads the artifact lazily on first `predict()` and caches it; the API should call `load_model()` in its lifespan once M5 uses it | Importing the module must not fail on a fresh clone or in tests, where no trained artifact exists |
| 2026-09-24 | Sentiment eval is judged on minority-class (negative) F1 vs. a majority-class baseline, with per-game breakdown | Steam reviews skew positive, so accuracy alone rewards a do-nothing model. First run: negative F1 0.718 vs. baseline 0.000; accuracy 0.860 vs. 0.781. On the ~98%-positive games (Hades, Portal 2) model accuracy is *below* the baseline's |
| 2026-09-24 | Ingestion fetches the most recent reviews (`filter=recent`), up to `MAX_REVIEWS`, and never refreshes a cached game. Chosen during M1 but not logged until now | Simplest cursor walk. Consequence: all sentiment/aspect results describe *recent* opinion, not all-time. With the 1000 cap the cached window is only ~1-3 months per game (e.g. Portal 2: 2026-08-25 to 2026-09-24), which also limits the M6 sentiment-trend chart. Revisit before M6 (see brief section 9) |
