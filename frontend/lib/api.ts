// Types for the FastAPI responses and the one place that turns an HTTP answer into a UI state.

export type Sentiment = "positive" | "negative";

export type SentimentCounts = {
  positive: number;
  negative: number;
  total: number;
  positive_pct: number | null;
};

export type AspectSummary = {
  appid: number;
  name: string;
  status: "ready";
  overall: SentimentCounts;
  aspects: (SentimentCounts & { aspect: string })[];
  trend: { date: string; positive: number; negative: number }[];
};

export type Review = {
  id: number;
  review_text: string;
  voted_up: boolean;
  votes_up: number;
  playtime_forever: number; // minutes
  language: string;
  created_at: string;
  predicted_sentiment: Sentiment | null;
  aspects: string[];
};

export type ReviewPage = {
  appid: number;
  total: number;
  limit: number;
  offset: number;
  items: Review[];
};

export type AvailableGame = { appid: number; name: string };

export type Loaded<T> =
  | { kind: "ready"; data: T }
  | { kind: "processing" } // 202: analysis running
  // 500 status "failed": needs a --force rerun by the owner. `interrupted`: the run's process
  // died, and `message` says so; otherwise `message` is a generic line.
  | { kind: "failed"; message: string; interrupted: boolean }
  | { kind: "not_found"; message: string } // 404: Steam has no such app
  // 403 status "unavailable": this deployment only serves games processed ahead of time
  | { kind: "unavailable"; message: string; games: AvailableGame[] }
  | { kind: "timeout" } // still processing after maxWaitMs
  | { kind: "error"; status: number | null; message: string }; // 422, 502, 503, network, ...

export async function fetchState<T>(path: string, signal?: AbortSignal): Promise<Loaded<T>> {
  let res: Response;
  try {
    res = await fetch(path, { signal, cache: "no-store" });
  } catch (err) {
    if (signal?.aborted) throw err;
    return { kind: "error", status: null, message: "Could not reach the API." };
  }
  const body = await res.json().catch(() => null);
  if (signal?.aborted) throw signal.reason; // aborted while the body was streaming
  if (res.status === 200) {
    if (body === null) return { kind: "error", status: 200, message: "The API sent an unreadable answer." };
    return { kind: "ready", data: body as T };
  }
  if (res.status === 202) return { kind: "processing" };
  const message = typeof body?.detail === "string" ? body.detail : `Request failed (${res.status})`;
  if (res.status === 404) return { kind: "not_found", message };
  if (res.status === 403 && body?.status === "unavailable") {
    const games = Array.isArray(body.available) ? (body.available as AvailableGame[]) : [];
    return { kind: "unavailable", message, games };
  }
  if (res.status === 500 && body?.status === "failed") {
    return { kind: "failed", message, interrupted: body.interrupted === true };
  }
  return { kind: "error", status: res.status, message };
}

export type PollOptions = {
  intervalMs?: number;
  maxWaitMs?: number;
  signal?: AbortSignal;
  onUpdate?: (state: Loaded<unknown>) => void;
};

/** Fetch `path` until it stops answering 202; every answer goes to `onUpdate`. */
export async function poll<T>(
  path: string,
  { intervalMs = 3000, maxWaitMs = 5 * 60_000, signal, onUpdate }: PollOptions = {},
): Promise<Loaded<T>> {
  const deadline = Date.now() + maxWaitMs;
  for (;;) {
    const state = await fetchState<T>(path, signal);
    if (state.kind !== "processing") {
      onUpdate?.(state);
      return state;
    }
    // The API fails a game whose run died, but a run can legitimately outlast any wait here.
    if (Date.now() + intervalMs > deadline) {
      onUpdate?.({ kind: "timeout" });
      return { kind: "timeout" };
    }
    onUpdate?.(state);
    await sleep(intervalMs, signal);
  }
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(signal.reason);
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      clearTimeout(timer);
      reject(signal.reason);
    }, { once: true });
  });
}

export function pageInfo(total: number, limit: number, offset: number) {
  const past = offset >= total; // empty result, or an offset beyond the last page (hand-edited URL)
  return {
    from: past ? 0 : offset + 1,
    to: past ? 0 : Math.min(offset + limit, total),
    hasPrev: offset > 0,
    // From past the end, jump back to the real last page instead of stepping through empty ones.
    prevOffset: past ? Math.max(0, Math.floor((total - 1) / limit) * limit) : Math.max(0, offset - limit),
    hasNext: offset + limit < total,
  };
}
