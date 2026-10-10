"use client";

import { Bar, BarChart, CartesianGrid, LabelList, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { MetricSummary } from "@/lib/types";

// Rate metrics only (all on a 0-1 scale) so a single axis is valid.
export const CHART_METRICS: [string, string][] = [
  ["accuracy", "Accuracy"],
  ["selective_accuracy", "Selective acc."],
  ["correct_abstention_rate", "Correct abstention"],
  ["wrong_abstention_rate", "Wrong abstention"],
  ["citation_gold_rate", "Cites gold doc"],
  ["evidence_recall", "Evidence recall"],
];

const SERIES: Record<string, { label: string; color: string }> = {
  standard: { label: "Standard RAG", color: "var(--series-1)" },
  adaptive: { label: "Adaptive RAG", color: "var(--series-2)" },
};

export function MetricsChart({ summary, methods }: { summary: Record<string, MetricSummary>; methods: string[] }) {
  const data = CHART_METRICS.map(([key, label]) => {
    const row: Record<string, string | number | null> = { metric: label };
    methods.forEach((m) => { row[m] = summary[m]?.[key] ?? null; });
    return row;
  }).filter((row) => methods.some((m) => row[m] != null));

  const pct = (v: unknown) => (typeof v === "number" ? `${Math.round(v * 100)}%` : "n/a");

  return (
    <div className="h-80 w-full" role="img" aria-label="Grouped bar chart of rate metrics per method; the same values are listed in the table below.">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 20, right: 8, left: -12, bottom: 0 }} barGap={2} barCategoryGap="22%">
          <CartesianGrid vertical={false} stroke="var(--grid)" />
          <XAxis dataKey="metric" tick={{ fill: "var(--muted)", fontSize: 11 }} tickLine={false} axisLine={{ stroke: "var(--border)" }} interval={0} />
          <YAxis domain={[0, 1]} tickFormatter={pct} tick={{ fill: "var(--muted)", fontSize: 11 }} tickLine={false} axisLine={false} width={48} />
          <Tooltip
            cursor={{ fill: "var(--surface-2)" }}
            formatter={(v, name) => [pct(v), SERIES[String(name)]?.label ?? String(name)]}
            contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 6, color: "var(--text)", fontSize: 12 }}
            labelStyle={{ color: "var(--text)", fontWeight: 600 }}
          />
          <Legend formatter={(v) => <span style={{ color: "var(--muted)", fontSize: 12 }}>{SERIES[v]?.label ?? v}</span>} />
          {methods.map((m) => (
            <Bar key={m} dataKey={m} fill={SERIES[m]?.color} radius={[4, 4, 0, 0]} maxBarSize={36} stroke="var(--surface)" strokeWidth={1}>
              <LabelList dataKey={m} position="top" formatter={pct} style={{ fill: "var(--muted)", fontSize: 10 }} />
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
