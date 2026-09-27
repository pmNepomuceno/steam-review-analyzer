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

export type Loaded<T> =
  | { kind: "ready"; data: T }
  | { kind: "processing" } // 202: analysis running
  | { kind: "failed"; message: string } // 500 status "failed": analysis crashed, needs --force
  | { kind: "not_found"; message: string } // 404: Steam has no such app
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
  if (res.status === 500 && body?.status === "failed") return { kind: "failed", message };
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
    // A worker killed mid-run leaves the game "processing" for good (see DECISIONS.md).
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
