"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { fetchState, headerImage, type GameListItem, type Loaded } from "../lib/api";
import AppidForm from "./AppidForm";

const DATE = new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", year: "numeric" });

/** The homepage's search box and the cards of every game whose analysis is ready. */
export default function GameBrowser() {
  const [games, setGames] = useState<Loaded<GameListItem[]> | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetchState<GameListItem[]>("/api/games", controller.signal)
      .then(setGames)
      .catch(() => {}); // aborted on unmount
    return () => controller.abort();
  }, []);

  const list = games?.kind === "ready" ? games.data : null;
  const analyzed = list ? new Set(list.map((g) => g.appid)) : undefined;

  return (
    <>
      <AppidForm analyzed={analyzed} />

      <section aria-labelledby="games-heading">
        <div className="section-head">
          <h2 id="games-heading">Analyzed games</h2>
          <p className="muted">Ready to open, each built from its most recent English reviews on Steam.</p>
        </div>
        {!games && (
          <ul className="games" aria-busy="true" aria-label="Loading games">
            {[0, 1, 2].map((i) => (
              <li key={i} className="skeleton" />
            ))}
          </ul>
        )}
        {games && !list && (
          <p role="alert" className="panel">
            Could not load the game list. {"message" in games ? games.message : ""} The search
            above still works.
          </p>
        )}
        {list?.length === 0 && (
          <p className="panel muted">
            No game has been analyzed yet. Search for one above to fetch and analyze it.
          </p>
        )}
        {list && list.length > 0 && (
          <ul className="games">
            {list.map((g) => (
              <li key={g.appid}>
                <Link href={`/games/${g.appid}`} className="game-card">
                  <div className="art">
                    {/* eslint-disable-next-line @next/next/no-img-element -- Steam CDN banner */}
                    <img
                      src={headerImage(g.appid)}
                      alt=""
                      width={460}
                      height={215}
                      loading="lazy"
                      onError={(e) => (e.currentTarget.style.visibility = "hidden")}
                    />
                  </div>
                  <div className="body">
                    <h3>{g.name}</h3>
                    <p className="meta">
                      {g.review_count.toLocaleString("en-US")} reviews
                      {g.analyzed_at && ` · Analyzed ${DATE.format(new Date(g.analyzed_at))}`}
                    </p>
                  </div>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </section>
    </>
  );
}
