# Project: Steam Review Sentiment + Aspect Analyzer

**One-liner:** A web app that pulls Steam reviews for any game and breaks
sentiment down by aspect (performance, price, bugs, story, gameplay) — a
view Steam's own UI never shows.

## 1. Problem and audience
- **Who is it for?** Indie devs triaging post-launch feedback, publishers
  weighing a price change, players comparing two similarly-rated games.
- **What problem does it solve?** Steam only shows one aggregate sentiment
  number. Finding out *why* players feel that way means reading hundreds
  of reviews by hand.
- **How do people solve it today?** Manually, or not at all.
- **Why build this?** Portfolio piece for AI Developer / AI Engineer roles —
  demonstrates a trained classifier + embedding-based pipeline + full-stack
  delivery, not just an LLM API call.

## 2. Core features (MVP only)
1. On-demand ingestion: given any Steam appid, fetch and cache its reviews
   in Postgres (first request triggers the fetch, later ones are instant).
2. Sentiment classification: TF-IDF + logistic regression trained on
   Steam's own `voted_up` label (weak supervision, no manual labeling).
3. Aspect extraction: split reviews into sentences, assign each to an
   aspect (performance, price, bugs, story, gameplay, or none) by cosine
   similarity to hand-written anchor phrases.
4. Aggregation: per-game, per-aspect positive/negative breakdown, served
   via API.
5. Dashboard: game selector, aspect breakdown chart, sentiment trend,
   filterable review list.

## 3. Out of scope (for now)
- Non-English reviews (English-only for v1)
- Real-time/streaming ingestion — fetch-and-cache only
- User accounts or auth
- Transformer fine-tuning — stretch goal, not committed scope
- Review-bomb detection/filtering

## 4. Tech stack
| Layer | Choice | Why |
|---|---|---|
| Frontend | Next.js (React) | Dashboard UI, deploys free on Vercel |
| Backend | FastAPI, domain-driven structure | Scales cleanly across games/reviews/ml domains |
| Database | PostgreSQL (Neon/Supabase free tier) | Review cache + labeled data |
| Auth | None (v1) | Out of scope |
| Hosting | Render/Fly (backend) + Vercel (frontend) | $0 at this traffic level |
| Other | scikit-learn (sentiment), sentence-transformers (aspect embeddings), Docker | Real trained/embedding-based ML, not LLM calls |

## 5. Data model (rough)
- `games`: appid, name, last_ingested_at
- `reviews`: id, appid (FK), review_text, voted_up, votes_up, playtime_forever, language, created_at
- `review_aspects`: review_id (FK), sentence_text, aspect_label, similarity_score, predicted_sentiment

## 6. Key screens / API endpoints
- `GET /games/{appid}/aspects` — aggregated per-aspect sentiment
- `GET /games/{appid}/reviews` — filterable review list (by aspect, sentiment)
- Ingestion triggers lazily on first `GET` for an unseen appid
- Frontend: game selector → per-game dashboard (aspect chart, trend, filtered review list)

## 7. Milestones
| # | Milestone | Done when... |
|---|---|---|
| 1 | Repo scaffold + on-demand ingestion | Given any appid, reviews are fetched, cached in Postgres, min-review-count guard in place. **Done 2026-09-24** |
| 2 | Sentiment baseline | TF-IDF + LogReg trained on `voted_up`, reports accuracy/F1 vs. majority-class baseline. **Done 2026-09-24** (negative-class F1 0.718 vs. baseline 0.000) |
| 3 | Aspect anchors + embedding assignment | Sentences tagged with an aspect (or none) via cosine similarity. **Done 2026-09-24** (standalone `assign_aspects()`; persistence deferred to M5; threshold 0.40 pending M4) |
| 4 | Aspect evaluation | 100–150 hand-labeled sentences scored; per-aspect precision/recall documented. **Code done 2026-09-24** (140-unit sample + `evaluate_aspects.py`); waiting on hand labels |
| 5 | Aggregation + API | `/games/{appid}/aspects` and `/games/{appid}/reviews` return real data |
| 6 | Dashboard | Selector, aspect chart, trend chart, filterable review list against live API |
| 7 | Polish, deploy, stretch goal | Live URL works, README has eval results; fine-tune DistilBERT if time allows |

## 8. Constraints and risks
- **Time:** 1 week
- **Budget:** $0 (free tiers only)
- **Riskiest assumption:** anchor-based aspect assignment is "good enough" without labeled training data — mitigated by the M4 evaluation step
- **Things I don't know how to do yet:** tuning similarity thresholds well; fine-tuning a transformer on a time budget

## 9. Open questions
- Final aspect list — confirm: performance, price, bugs, story, gameplay, other
- ~~Minimum review count before serving results for a new appid~~ — resolved: 200 usable English reviews (see `DECISIONS.md`)
- Which 2-3 games to develop/evaluate against before wiring up "any appid"
- Review sample: ingestion currently takes only the ~1000 most recent reviews and never refreshes, so results reflect recent opinion over a 1-3 month window. Decide before M6 whether that's the intended scope (and label it in the UI), or whether to sample across the game's lifetime and/or refresh stale caches (see `DECISIONS.md`)

## 10. Decisions log
| Date | Decision | Reason |
|---|---|---|
| 2026-09-24 | Use `voted_up` as weak-supervision sentiment label | Free, real signal from users, no manual labeling or LLM calls needed |
| 2026-09-24 | Anchor-based embedding similarity for aspects (not BERTopic or a fine-tuned classifier) | Fastest, most explainable option that fits a 1-week scope |
| 2026-09-24 | Support any appid via on-demand fetch-and-cache, not a fixed game list | Ships a real tool, not a one-off analysis |
| 2026-09-24 | Domain-driven backend structure (games/reviews/ml packages), per zhanymkanov/fastapi-best-practices | Scales better than type-based folders as the project grows |