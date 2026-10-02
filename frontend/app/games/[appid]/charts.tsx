"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { AspectSummary } from "../../../lib/api";
import { copyFor } from "../../../lib/insights";

// Colors are CSS variables from globals.css, so light/dark mode needs nothing here.
const AXIS = { stroke: "var(--border)", tick: { fill: "var(--text-2)", fontSize: 13 } };
const TOOLTIP = {
  contentStyle: {
    background: "var(--surface-2)",
    border: 0,
    borderRadius: 3,
    boxShadow: "0 4px 16px rgb(0 0 0 / 0.35)",
    color: "var(--text)",
  },
  labelStyle: { color: "var(--text-strong)", fontWeight: 500 },
  cursor: { fill: "var(--field)", fillOpacity: 0.5 },
};
const label = (aspect: string) => copyFor(aspect).label;
const LEGEND = {
  itemSorter: null, // keep series order (Positive first), not alphabetical
  formatter: (v: string) => <span style={{ color: "var(--text)" }}>{v}</span>,
};
// Data arrives once, so the grow-in animation only delays reading the chart.
const STATIC = { isAnimationActive: false };

export function AspectChart({ aspects }: { aspects: AspectSummary["aspects"] }) {
  const rows = aspects.map((a) => ({
    topic: label(a.aspect),
    positive: a.positive,
    negative: a.negative,
    total: a.total,
  }));
  return (
    <>
      <div className="chart">
        <ResponsiveContainer>
          <BarChart data={rows} layout="vertical" barGap={2} margin={{ left: 4, right: 16 }}>
            <CartesianGrid horizontal={false} stroke="var(--border)" />
            <XAxis type="number" allowDecimals={false} {...AXIS} />
            <YAxis type="category" dataKey="topic" width={118} {...AXIS} />
            <Tooltip {...TOOLTIP} />
            <Legend {...LEGEND} />
            <Bar {...STATIC} dataKey="positive" name="Positive" fill="var(--positive)" radius={[0, 4, 4, 0]} />
            <Bar {...STATIC} dataKey="negative" name="Negative" fill="var(--negative)" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="chart-note">
        Reviews with at least one sentence about the topic, by the review&apos;s predicted
        sentiment. A review counts once per topic however many sentences it has about it, and
        reviews about none of these topics are left out.
      </p>
      <DataTable
        caption="Show as a table"
        columns={["topic", "positive", "negative", "total"]}
        rows={rows}
      />
    </>
  );
}

export function TrendChart({ trend }: { trend: AspectSummary["trend"] }) {
  return (
    <>
      <div className="chart">
        <ResponsiveContainer>
          <BarChart data={trend} margin={{ right: 16 }}>
            <CartesianGrid vertical={false} stroke="var(--border)" />
            <XAxis dataKey="date" minTickGap={24} {...AXIS} />
            <YAxis allowDecimals={false} width={40} {...AXIS} />
            <Tooltip {...TOOLTIP} />
            <Legend {...LEGEND} />
            {/* A 1px surface-colored stroke on each segment makes a 2px gap between them. */}
            <Bar
              {...STATIC}
              dataKey="positive"
              name="Positive"
              stackId="day"
              fill="var(--positive)"
              stroke="var(--surface)"
              strokeWidth={1}
            />
            <Bar
              {...STATIC}
              dataKey="negative"
              name="Negative"
              stackId="day"
              fill="var(--negative)"
              stroke="var(--surface)"
              strokeWidth={1}
              radius={[4, 4, 0, 0]}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <DataTable
        caption="Show as a table"
        columns={["date", "positive", "negative"]}
        rows={trend}
      />
    </>
  );
}

function DataTable<T extends Record<string, string | number | null>>({
  caption,
  columns,
  rows,
}: {
  caption: string;
  columns: (keyof T & string)[];
  rows: T[];
}) {
  return (
    <details>
      <summary>{caption}</summary>
      <table>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {columns.map((c) => (
                <td key={c}>{row[c]}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </details>
  );
}
