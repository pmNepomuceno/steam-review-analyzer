import { describe, expect, it } from "vitest";

import type { AspectStats } from "./api";
import { comparison, storePct, verdict } from "./insights";

const aspect = (name: string, positive_pct: number | null, total = 100): AspectStats => ({
  aspect: name,
  positive: 0,
  negative: 0,
  total,
  positive_pct,
  quotes: [],
});
const sample = (positive_pct: number | null) => ({
  positive: 0,
  negative: 0,
  total: 1000,
  positive_pct,
});

// Cities: Skylines II's real numbers (story at 41 mentions still counts).
const CITIES = [
  aspect("performance", 49.2, 122),
  aspect("price", 46.9, 81),
  aspect("bugs", 30.6, 170),
  aspect("story", 43.9, 41),
  aspect("gameplay", 68.9, 244),
];

describe("verdict", () => {
  it("names the best and the worst judged aspect", () => {
    expect(verdict({ aspects: CITIES })).toBe(
      "Players are most positive about gameplay (69%), but bugs and stability is the sore point at 31%.",
    );
  });

  it("calls a narrow spread flat instead of inventing a weakness", () => {
    const hades = [aspect("price", 95.2), aspect("story", 95.4), aspect("gameplay", 94.0)];
    expect(verdict({ aspects: hades })).toBe(
      "Players feel much the same about every topic they bring up: around 95% positive.",
    );
  });

  it("ignores aspects with too few mentions", () => {
    const few = [aspect("bugs", 0, 3), aspect("story", 98.6, 139)];
    expect(verdict({ aspects: few })).toBe(
      "Only story and writing comes up often enough to judge: 99% positive.",
    );
    expect(verdict({ aspects: [aspect("bugs", 0, 3), aspect("story", null, 0)] })).toBe(
      "Too few reviews mention specific topics to point out strengths or weaknesses.",
    );
  });

  it("uses the split and trails wording for middling weak spots", () => {
    expect(verdict({ aspects: [aspect("gameplay", 90), aspect("bugs", 55)] })).toBe(
      "Players are most positive about gameplay (90%), but they are split on bugs and stability (55%).",
    );
    expect(verdict({ aspects: [aspect("gameplay", 90), aspect("price", 70)] })).toBe(
      "Players are most positive about gameplay (90%), but price and value trails at 70%.",
    );
  });
});

describe("comparison", () => {
  it("contrasts Steam's share with a clearly weaker aspect", () => {
    expect(comparison({ aspects: CITIES, steam_sample: sample(65.6) })).toEqual({
      steamPct: 66,
      aspect: "bugs",
      aspectPct: 31,
      weaker: true,
      text: "Steam says 66% positive, but satisfaction on bugs and stability is only 31%.",
    });
  });

  it("says so when even the weakest aspect keeps up", () => {
    const portal = [aspect("story", 98.6), aspect("gameplay", 98.4)];
    expect(comparison({ aspects: portal, steam_sample: sample(98.3) })?.text).toBe(
      "Steam says 98% positive, and even gameplay, the lowest-rated topic, holds 98%.",
    );
  });

  it("is absent without a judged aspect or a Steam share", () => {
    expect(comparison({ aspects: [aspect("bugs", 0, 3)], steam_sample: sample(90) })).toBeNull();
    expect(comparison({ aspects: CITIES, steam_sample: sample(null) })).toBeNull();
  });
});

describe("storePct", () => {
  it("truncates like Steam's store page", () => {
    // Stored ratings checked against the live store pages on 2026-10-02.
    expect(storePct({ positive: 171921, total: 173897 })).toBe(98); // Portal 2, 98.86%
    expect(storePct({ positive: 20019, total: 34580 })).toBe(57); // Cities: Skylines II, 57.89%
    expect(storePct({ positive: 14370, total: 31932 })).toBe(45); // PAYDAY 3, 45.00%
    expect(storePct({ positive: 29, total: 100 })).toBe(29); // no float underflow below 29
  });
});
