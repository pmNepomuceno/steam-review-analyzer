"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { poll, type AspectSummary, type Loaded } from "../../../lib/api";
import AppidForm from "../../AppidForm";
import { AspectChart, TrendChart } from "./charts";
import ReviewList from "./ReviewList";

type State = Loaded<AspectSummary> | { kind: "loading" };

const POLL_MS = 3000;

export default function Dashboard({ appid }: { appid: number }) {
  const [state, setState] = useState<State>({ kind: "loading" });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    poll<AspectSummary>(`/api/games/${appid}/aspects`, {
      intervalMs: POLL_MS,
      signal: controller.signal,
      onUpdate: (s) => setState(s as Loaded<AspectSummary>),
    }).catch(() => {}); // only rejects when aborted on unmount
    return () => controller.abort();
  }, [appid, attempt]);

  const retry = () => {
    setState({ kind: "loading" });
    setAttempt((n) => n + 1);
  };

  switch (state.kind) {
    case "loading":
      return (
        <Status title="Loading…" spinner>
          A game seen for the first time is fetched from Steam first, which takes 10–20 seconds.
        </Status>
      );
    case "processing":
      return (
        <Status title="Analyzing reviews…" spinner>
          Sentiment and aspect tagging runs once per game and takes up to a minute. Checking again
          every {POLL_MS / 1000} seconds; results appear here when ready.
        </Status>
      );
    case "timeout":
      return (
        <Status title="Still analyzing" error>
          This is taking longer than expected; the analysis may have stalled.
          <p>
            <button onClick={retry}>Check again</button>
          </p>
        </Status>
      );
    case "failed":
      return (
        <Status title="Analysis failed" error>
          {/* No retry button: a failed game stays failed until the owner reruns it with
              --force, which no visitor can do. */}
          {state.message}
        </Status>
      );
    case "not_found":
      return (
        <Status title="Game not found" error>
          Steam has no app with appid {appid}. Check the number in the game&apos;s store URL.
          <div style={{ display: "flex", justifyContent: "center", marginTop: 16 }}>
            <AppidForm />
          </div>
        </Status>
      );
    case "unavailable":
      return (
        <Status title="Not in this demo">
          {state.message}
          {state.games.length > 0 && (
            <>
              <p>Available games:</p>
              <ul>
                {state.games.map((g) => (
                  <li key={g.appid}>
                    <Link href={`/games/${g.appid}`}>{g.name}</Link>
                  </li>
                ))}
              </ul>
            </>
          )}
        </Status>
      );
    case "error":
      return (
        <Status title={state.status === 422 ? "Not enough reviews" : "Something went wrong"} error>
          {state.message}
          <p>
            <button onClick={retry}>Try again</button> or <Link href="/">pick another game</Link>
          </p>
        </Status>
      );
    case "ready":
      return <Ready data={state.data} />;
  }
}

function Status({
  title,
  spinner,
  error,
  children,
}: {
  title: string;
  spinner?: boolean;
  error?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className={`card state${error ? " error" : ""}`} role={error ? "alert" : "status"}>
      {spinner && <span className="spinner" aria-hidden />}
      <h2>{title}</h2>
      <div className="muted">{children}</div>
    </div>
  );
}

function Ready({ data }: { data: AspectSummary }) {
  const { overall, trend } = data;
  const first = trend[0]?.date;
  const last = trend.at(-1)?.date;
  return (
    <>
      <p>
        <Link href="/">← Another game</Link>
      </p>
      <h1>{data.name}</h1>
      <p className="muted">
        Based on the {overall.total} most recent English reviews
        {first && ` (${first} to ${last})`}: {overall.positive_pct ?? 0}% predicted positive.
        Older reviews are not included.
      </p>
      <section className="card">
        <h2>Sentiment by aspect</h2>
        <AspectChart aspects={data.aspects} />
      </section>
      <section className="card">
        <h2>Reviews per day</h2>
        <TrendChart trend={trend} />
      </section>
      <section className="card">
        <h2>Reviews</h2>
        <ReviewList appid={data.appid} aspects={data.aspects.map((a) => a.aspect)} />
      </section>
    </>
  );
}
