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
