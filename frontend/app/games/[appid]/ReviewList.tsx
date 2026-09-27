"use client";

import { useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";

import { fetchState, pageInfo, type Loaded, type ReviewPage } from "../../../lib/api";

const PAGE_SIZE = 20;

export default function ReviewList({ appid, aspects }: { appid: number; aspects: string[] }) {
  // Filters and page live in the URL (?aspect=bugs&sentiment=negative&offset=20), so a
  // filtered view can be linked to and survives a reload.
  // Unknown values (a stale or hand-edited link) are ignored rather than sent on for a 422.
  const search = useSearchParams();
  const rawAspect = search.get("aspect") ?? "";
  const aspect = aspects.includes(rawAspect) ? rawAspect : "";
  const rawSentiment = search.get("sentiment") ?? "";
  const sentiment = ["positive", "negative"].includes(rawSentiment) ? rawSentiment : "";
  const rawOffset = Number(search.get("offset"));
  const offset = Number.isSafeInteger(rawOffset) && rawOffset > 0 ? rawOffset : 0;
  const navigate = (changes: Record<string, string | number>) => {
    const next = new URLSearchParams(search);
    for (const [key, value] of Object.entries(changes)) {
      if (value) next.set(key, String(value));
      else next.delete(key);
    }
    window.history.replaceState(null, "", next.size ? `?${next}` : window.location.pathname);
  };
  // The answer remembers which query it belongs to; a mismatch means a newer one is loading.
  const [answer, setAnswer] = useState<{ query: string; result: Loaded<ReviewPage> } | null>(null);

  const params = new URLSearchParams({ limit: String(PAGE_SIZE), offset: String(offset) });
  if (aspect) params.set("aspect", aspect);
  if (sentiment) params.set("sentiment", sentiment);
  const query = `/api/games/${appid}/reviews?${params}`;
  const result = answer?.result ?? null;
  const loading = answer?.query !== query;

  useEffect(() => {
    const controller = new AbortController();
    fetchState<ReviewPage>(query, controller.signal)
      .then((r) => setAnswer({ query, result: r }))
      .catch(() => {}); // aborted by a newer filter/page or unmount
    return () => controller.abort();
  }, [query]);

  const page = result?.kind === "ready" ? result.data : null;
  const info = page && pageInfo(page.total, PAGE_SIZE, page.offset);

  return (
    <>
      <div className="filters">
        <label>
          Aspect
          <select
            value={aspect}
            onChange={(e) => navigate({ aspect: e.target.value, offset: 0 })}
          >
            <option value="">All reviews</option>
            {aspects.map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
        </label>
        <label>
          Sentiment
          <select
            value={sentiment}
            onChange={(e) => navigate({ sentiment: e.target.value, offset: 0 })}
          >
            <option value="">Any</option>
            <option value="positive">Positive</option>
            <option value="negative">Negative</option>
          </select>
        </label>
      </div>

      {result && result.kind !== "ready" && (
        <p role="alert">
          {result.kind === "processing" || result.kind === "timeout"
            ? "Reviews are still being analyzed; reload shortly."
            : "message" in result
              ? result.message
              : "Could not load reviews."}
        </p>
      )}
      {!result && <p className="muted">Loading reviews…</p>}

      {page && info && (
        <>
          <p className="muted" aria-live="polite">
            {page.total === 0
              ? "No reviews match these filters."
              : info.to === 0
                ? "No reviews on this page."
                : `Showing ${info.from}–${info.to} of ${page.total}`}
          </p>
          <ul className={`reviews${loading ? " stale" : ""}`} aria-busy={loading}>
            {page.items.map((r) => (
              <li key={r.id}>
                <div className="review-meta">
                  {r.predicted_sentiment && (
                    <span className={`badge ${r.predicted_sentiment}`}>
                      {r.predicted_sentiment}
                    </span>
                  )}
                  {r.aspects.map((a) => (
                    <span key={a} className="tag">
                      {a}
                    </span>
                  ))}
                  <span>
                    · {r.created_at.slice(0, 10)} · {Math.round(r.playtime_forever / 60)} h played
                    · {r.voted_up ? "Recommended" : "Not recommended"} on Steam
                  </span>
                </div>
                <p className="review-text">{r.review_text}</p>
              </li>
            ))}
          </ul>
          {(info.hasPrev || info.hasNext) && (
            <div className="pager">
              <button
                disabled={!info.hasPrev || loading}
                onClick={() => navigate({ offset: info.prevOffset })}
              >
                ← Newer
              </button>
              <button
                disabled={!info.hasNext || loading}
                onClick={() => navigate({ offset: page.offset + PAGE_SIZE })}
              >
                Older →
              </button>
            </div>
          )}
        </>
      )}
    </>
  );
}
