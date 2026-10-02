"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import {
  headerImage,
  poll,
  type AspectStats,
  type AspectSummary,
  type Loaded,
  type SteamRating,
} from "../../../lib/api";
import {
  comparison,
  copyFor,
  formatDate,
  MIN_MENTIONS,
  pct,
  quoteText,
  spike,
  storePct,
  strength,
  verdict,
} from "../../../lib/insights";
import AppidForm from "../../AppidForm";
import { TrendChart } from "./charts";
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
          {/* Shown only if the answer is slow, so a cached game doesn't flash it. */}
          <span className="later">
            A game seen for the first time is fetched from Steam first, which takes 10–20 seconds.
          </span>
        </Status>
      );
    case "processing":
      return (
        <Status title="Analyzing reviews…" spinner>
          Sentiment and topic tagging runs once per game and takes up to a minute. Checking again
          every {POLL_MS / 1000} seconds; results appear here when ready.
        </Status>
      );
    case "timeout":
      return (
        <Status title="Still analyzing">
          <p>This is taking longer than expected; the analysis may have stalled.</p>
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
          <p>{state.message}</p>
        </Status>
      );
    case "not_found":
      return (
        <Status title="Game not found" error>
          <p>Steam has no app with appid {appid}. Search for the game by name instead:</p>
          <div className="state-search">
            <AppidForm />
          </div>
        </Status>
      );
    case "unavailable":
      return (
        <Status title="Not in this demo">
          <p>{state.message}</p>
          {state.games.length > 0 && (
            <>
              <p>These games are ready to open:</p>
              <ul>
                {state.games.map((g) => (
                  <li key={g.appid}>
                    <Link href={`/games/${g.appid}`}>{g.name}</Link>
                  </li>
                ))}
              </ul>
            </>
          )}
          <p>
            <Link href="/">Back to the homepage</Link>
          </p>
        </Status>
      );
    case "error":
      return (
        <Status title={state.status === 422 ? "Not enough reviews" : "Something went wrong"} error>
          <p>{state.message}</p>
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
    <div className={`panel state${error ? " error" : ""}`} role={error ? "alert" : "status"}>
      {spinner && <span className="spinner" aria-hidden />}
      <h1>{title}</h1>
      <div className="muted">{children}</div>
    </div>
  );
}

function Ready({ data }: { data: AspectSummary }) {
  const { overall, trend, steam_rating: rating } = data;
  const first = trend[0]?.date;
  const last = trend.at(-1)?.date;
  const peak = spike(trend);
  return (
    <>
      <Link href="/" className="back">
        ← All games
      </Link>
      <div className="game-head">
        <div>
          <h1>{data.name}</h1>
          <p className="sample">
            The {overall.total.toLocaleString("en-US")} most recent English reviews
            {first && last && `, ${formatDate(first)} to ${formatDate(last)}`}. Older reviews are
            not included.
          </p>
        </div>
        <div className="art">
          {/* eslint-disable-next-line @next/next/no-img-element -- Steam CDN banner */}
          <img
            src={headerImage(data.appid)}
            alt=""
            width={460}
            height={215}
            onError={(e) => (e.currentTarget.style.visibility = "hidden")}
          />
        </div>
      </div>

      <Headline data={data} rating={rating} />

      <section aria-labelledby="topics-heading">
        <div className="section-head">
          <h2 id="topics-heading">By topic</h2>
          <p className="muted">
            Each review is split into sentences and every sentence is matched to a topic. The
            share is how many of the reviews that bring a topic up read as positive.
          </p>
        </div>
        <ul className="aspects">
          {data.aspects.map((a) => (
            <AspectCard key={a.aspect} appid={data.appid} aspect={a} />
          ))}
        </ul>
      </section>

      <section className="panel" aria-labelledby="trend-heading">
        <h2 id="trend-heading">Reviews per day</h2>
        {peak && <p className="callout">{peak.text}</p>}
        <TrendChart trend={trend} spike={peak?.date} />
      </section>
      <section className="panel" id="reviews" aria-labelledby="reviews-heading">
        <h2 id="reviews-heading">Reviews</h2>
        <ReviewList appid={data.appid} aspects={data.aspects.map((a) => a.aspect)} />
      </section>
    </>
  );
}

function RatingLabel({ rating }: { rating: SteamRating }) {
  const tone = rating.score_desc.includes("Positive")
    ? "positive"
    : rating.score_desc.includes("Negative")
      ? "negative"
      : "mixed";
  return (
    <>
      <span className={`rating-${tone}`}>{rating.score_desc}</span>
      {rating.positive_pct !== null &&
        `, ${storePct(rating)}% of ${rating.total.toLocaleString("en-US")} reviews`}
    </>
  );
}

/** The page's one claim: Steam's thumbs against the weakest topic, then the method behind it. */
function Headline({ data, rating }: { data: AspectSummary; rating: SteamRating | null }) {
  const result = comparison(data);
  const summary = verdict(data);
  const support = result && strength(data);
  const sample = data.steam_sample.total.toLocaleString("en-US");
  return (
    <section className="headline panel" aria-labelledby="headline">
      <h2 id="headline" className="verdict">
        {result?.text ?? summary}
      </h2>
      {result && (
        <div className="figures">
          <div className="figure">
            <span className="num">{result.steamPct}%</span>
            <span className="label">Steam thumbs up</span>
          </div>
          <div className="figure">
            <span className={`num${result.weaker ? " weak" : ""}`}>{result.aspectPct}%</span>
            <span className="label">{copyFor(result.aspect).label}</span>
          </div>
        </div>
      )}
      {support && <p className="support">{support}</p>}
      <details className="about">
        <summary>About these numbers</summary>
        <ul>
          {result && (
            <li>
              Both figures come from the same {sample} reviews. Steam&apos;s is the reviewers&apos;
              own thumbs up; the other is the share of reviews mentioning{" "}
              {copyFor(result.aspect).phrase} that the sentiment model reads as positive.
            </li>
          )}
          {rating && (
            <li>
              Steam&apos;s store rating counts every English review, not only these:{" "}
              <RatingLabel rating={rating} />.
            </li>
          )}
          {/* Hand-copied from the eval reports; update them after retraining or retuning.
              Sentiment: negative precision 0.643 / recall 0.813 (docs/eval/sentiment_report.txt),
              not accuracy, which a model answering "positive" every time already gets to 78%.
              Tagging: accuracy 0.621 vs baseline 0.329 (docs/eval/aspect_report.txt). */}
          <li>
            The sentiment model reads {pct(data.overall.positive_pct) ?? 0}% of these {sample}{" "}
            reviews as positive. Tested on 1,000 reviews it had not seen, when it called a review
            negative it was right about 2 times in 3, and it caught about 4 in 5 of the negative
            reviews. It is weakest on games nearly everyone likes, where negative reviews are rare.
          </li>
          <li>
            Topic tagging matched 62% of 140 hand-labelled sentences, against 33% for always
            guessing &ldquo;no topic&rdquo;. The topic phrases were tuned on those same sentences,
            so that figure is optimistic. Expect some quotes under the wrong topic.
          </li>
          <li>
            <a href={METHOD_URL}>How it works and how it was measured</a>
          </li>
        </ul>
      </details>
    </section>
  );
}

const METHOD_URL = "https://github.com/pmNepomuceno/steam-review-analyzer#readme";

function AspectCard({ appid, aspect: a }: { appid: number; aspect: AspectStats }) {
  const { label, description } = copyFor(a.aspect);
  const share = pct(a.positive_pct);
  // A topic most reviews complain about links to the complaints; any other to all its reviews.
  const negative = a.negative > a.positive;
  const query = negative ? `aspect=${a.aspect}&sentiment=negative` : `aspect=${a.aspect}`;
  const count = negative ? a.negative : a.total;
  return (
    <li className="aspect">
      <div>
        <h3>{label}</h3>
        <p className="desc">{description}</p>
      </div>
      {share === null ? (
        <p className="muted">No review in this sample mentions it.</p>
      ) : (
        <>
          <p className="stat">
            <strong>{share}%</strong>
            <span>
              positive · {a.total.toLocaleString("en-US")} review{a.total === 1 ? "" : "s"}
              {a.total < MIN_MENTIONS && ", too few to judge"}
            </span>
          </p>
          <div
            className="split"
            role="img"
            aria-label={`${a.positive} positive and ${a.negative} negative reviews`}
          >
            {a.positive > 0 && <span className="pos" style={{ flex: a.positive }} />}
            {a.negative > 0 && <span className="neg" style={{ flex: a.negative }} />}
          </div>
        </>
      )}
      {a.quotes.length > 0 && (
        <ul className="quotes" aria-label={`What reviews say about ${label}`}>
          {a.quotes.map((q) => (
            <li key={q.text} className={q.sentiment}>
              <Thumb up={q.sentiment === "positive"} />
              <blockquote>&ldquo;{quoteText(q.text)}&rdquo;</blockquote>
            </li>
          ))}
        </ul>
      )}
      {a.total > 0 && (
        <Link className="more" href={`/games/${appid}?${query}#reviews`}>
          {negative
            ? `Read the ${count === 1 ? "negative review" : `${count.toLocaleString("en-US")} negative reviews`}`
            : `Read ${count === 1 ? "the review" : `all ${count.toLocaleString("en-US")} reviews`}`}
          <span className="sr-only"> about {label.toLowerCase()}</span> →
        </Link>
      )}
    </li>
  );
}

function Thumb({ up }: { up: boolean }) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      role="img"
      aria-label={up ? "From a positive review" : "From a negative review"}
    >
      {up ? (
        <>
          <path d="M7 10v12" />
          <path d="M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2a3.13 3.13 0 0 1 3 3.88Z" />
        </>
      ) : (
        <>
          <path d="M17 14V2" />
          <path d="M9 18.12 10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.76a2 2 0 0 0-1.79 1.11L12 22a3.13 3.13 0 0 1-3-3.88Z" />
        </>
      )}
    </svg>
  );
}
