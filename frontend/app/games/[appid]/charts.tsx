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

// Colors are CSS variables from globals.css, so light/dark mode needs nothing here.
const AXIS = { stroke: "var(--border)", tick: { fill: "var(--text-2)", fontSize: 12 } };
const TOOLTIP = {
  contentStyle: {
    background: "var(--surface)",
    border: "1px solid var(--border)",
    borderRadius: 6,
    color: "var(--text)",
  },
  cursor: { fill: "var(--border)", fillOpacity: 0.4 },
};
const LEGEND = {
  itemSorter: null, // keep series order (Positive first), not alphabetical
  formatter: (v: string) => <span style={{ color: "var(--text)" }}>{v}</span>,
};
// Data arrives once, so the grow-in animation only delays reading the chart.
const STATIC = { isAnimationActive: false };

export function AspectChart({ aspects }: { aspects: AspectSummary["aspects"] }) {
  return (
    <>
      <div className="chart">
        <ResponsiveContainer>
          <BarChart data={aspects} layout="vertical" barGap={2} margin={{ left: 8, right: 16 }}>
            <CartesianGrid horizontal={false} stroke="var(--border)" />
            <XAxis type="number" allowDecimals={false} {...AXIS} />
            <YAxis type="category" dataKey="aspect" width={88} {...AXIS} />
            <Tooltip {...TOOLTIP} />
            <Legend {...LEGEND} />
            <Bar {...STATIC} dataKey="positive" name="Positive" fill="var(--positive)" radius={[0, 4, 4, 0]} />
            <Bar {...STATIC} dataKey="negative" name="Negative" fill="var(--negative)" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="muted">
        Distinct reviews with at least one sentence about the aspect, by the review&apos;s
        predicted sentiment. Reviews about none of these count only in the total.
      </p>
      <DataTable
        caption="Aspect counts as a table"
        columns={["aspect", "positive", "negative", "total"]}
        rows={aspects}
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
              stroke="var(--surface-2)"
              strokeWidth={1}
            />
            <Bar
              {...STATIC}
              dataKey="negative"
              name="Negative"
              stackId="day"
              fill="var(--negative)"
              stroke="var(--surface-2)"
              strokeWidth={1}
              radius={[4, 4, 0, 0]}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <DataTable
        caption="Reviews per day as a table"
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
