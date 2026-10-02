// Plain-language reading of an /aspects answer: labels for the aspects and the templated
// sentences above the charts. Templates on purpose, no LLM: every sentence is traceable to
// the numbers shown below it.

import type { AspectStats, AspectSummary } from "./api";

type AspectCopy = { label: string; phrase: string; description: string };

// `phrase` is the label as it reads inside a sentence.
export const ASPECT_COPY: Record<string, AspectCopy> = {
  performance: {
    label: "Performance",
    phrase: "performance",
    description: "Frame rate, stutter, loading times and how well it runs",
  },
  price: {
    label: "Price & value",
    phrase: "price and value",
    description: "Whether it is worth what it costs",
  },
  bugs: {
    label: "Bugs & stability",
    phrase: "bugs and stability",
    description: "Crashes, glitches and broken features",
  },
  story: {
    label: "Story & writing",
    phrase: "story and writing",
    description: "Plot, characters, dialogue and voice acting",
  },
  gameplay: {
    label: "Gameplay",
    phrase: "gameplay",
    description: "Combat, mechanics, content and the core loop",
  },
};

export const copyFor = (aspect: string): AspectCopy =>
  ASPECT_COPY[aspect] ?? { label: aspect, phrase: aspect, description: "" };

// Fewer mentions than this and an aspect's percentage is too noisy to call a strength or a
// weakness (Hades has 12 performance reviews out of 1000; one review moves it 8 points).
export const MIN_MENTIONS = 20;
const FLAT_SPREAD = 10; // best and worst within this many points: no standout either way
const GAP = 5; // Steam's share must beat the weakest aspect by this much for a "but"

export const pct = (value: number | null) => (value === null ? null : Math.round(value));

// Steam's store page truncates its percentage (171,921 of 173,897 is 98.86%, shown as 98%), so
// the stored rating is shown the same way to match what visitors see there.
export const storePct = ({ positive, total }: { positive: number; total: number }) =>
  Math.floor((100 * positive) / total);

/** Aspects with enough mentions to judge, weakest first. */
export function judged(aspects: AspectStats[]): AspectStats[] {
  return aspects
    .filter((a) => a.total >= MIN_MENTIONS && a.positive_pct !== null)
    .sort((a, b) => (a.positive_pct ?? 0) - (b.positive_pct ?? 0));
}

export function verdict(summary: Pick<AspectSummary, "aspects">): string {
  const ranked = judged(summary.aspects);
  if (ranked.length === 0) {
    return "Too few reviews mention specific topics to point out strengths or weaknesses.";
  }
  const worst = ranked[0];
  const best = ranked[ranked.length - 1];
  const bestPct = pct(best.positive_pct)!;
  const worstPct = pct(worst.positive_pct)!;
  if (ranked.length === 1) {
    return `Only ${copyFor(best.aspect).phrase} comes up often enough to judge: ${bestPct}% positive.`;
  }
  if (bestPct - worstPct < FLAT_SPREAD) {
    const mean = Math.round(ranked.reduce((sum, a) => sum + (a.positive_pct ?? 0), 0) / ranked.length);
    return `Players feel much the same about every topic they bring up: around ${mean}% positive.`;
  }
  const weakness =
    worstPct < 40
      ? `${copyFor(worst.aspect).phrase} is the sore point at ${worstPct}%`
      : worstPct < 60
        ? `they are split on ${copyFor(worst.aspect).phrase} (${worstPct}%)`
        : `${copyFor(worst.aspect).phrase} trails at ${worstPct}%`;
  // "Most positive about", not "love": the best aspect can be bugs, where it means few complaints.
  return `Players are most positive about ${copyFor(best.aspect).phrase} (${bestPct}%), but ${weakness}.`;
}

export type Comparison = {
  steamPct: number;
  aspect: string;
  aspectPct: number;
  weaker: boolean; // the aspect is clearly below Steam's share (the "but" sentence)
  text: string;
};

/** Steam's thumbs-up share of the same reviews against the weakest judged aspect. */
export function comparison(
  summary: Pick<AspectSummary, "aspects" | "steam_sample">,
): Comparison | null {
  const steamPct = pct(summary.steam_sample.positive_pct);
  const [worst] = judged(summary.aspects);
  if (steamPct === null || !worst) return null;
  const aspectPct = pct(worst.positive_pct)!;
  const phrase = copyFor(worst.aspect).phrase;
  const weaker = aspectPct <= steamPct - GAP;
  const text = weaker
      ? `Steam says ${steamPct}% positive, but satisfaction on ${phrase} is only ${aspectPct}%.`
      : `Steam says ${steamPct}% positive, and even ${phrase}, the lowest-rated topic, holds ${aspectPct}%.`;
  return { steamPct, aspect: worst.aspect, aspectPct, weaker, text };
}
