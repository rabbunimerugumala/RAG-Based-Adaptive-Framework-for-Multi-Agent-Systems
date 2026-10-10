"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { StatusBadge, TraceTimeline } from "@/components/result-views";
import { Badge, Card, CardHeader, EmptyState, Notice, Spinner, Stat } from "@/components/ui";
import { api, fmt } from "@/lib/api";
import type { QueryResponse, RunSummary } from "@/lib/types";

export default function TraceView() {
  const runId = useSearchParams().get("run");
  const [run, setRun] = useState<QueryResponse | null>(null);
  const [recent, setRecent] = useState<RunSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api.runs(15).then((r) => !cancelled && setRecent(r)).catch(() => !cancelled && setRecent([]));
    if (runId) {
      api.run(runId)
        .then((r) => { if (!cancelled) { setRun(r); setError(null); } })
        .catch((e: Error) => { if (!cancelled) { setRun(null); setError(e.message); } });
    }
    return () => { cancelled = true; };
  }, [runId]);

  const picker = (
    <Card>
      <CardHeader title="Recent runs" description="Select a run to inspect its trace" />
      {recent == null ? <div className="px-5 py-6"><Spinner label="Loading…" /></div> : recent.length === 0 ? (
        <EmptyState title="No runs yet">Ask a question first; its trace will be available here.</EmptyState>
      ) : (
        <ul className="divide-y divide-border">
          {recent.map((r) => (
            <li key={r.run_id}>
              <Link href={`/trace?run=${r.run_id}`} aria-current={r.run_id === runId ? "true" : undefined}
                className="flex flex-wrap items-center gap-x-3 gap-y-1 px-5 py-2.5 text-sm hover:bg-surface-2 aria-[current]:bg-accent-soft">
                <span className="min-w-0 flex-1 truncate text-fg">{r.question}</span>
                <Badge tone={r.method === "adaptive" ? "accent" : "neutral"}>{r.method}</Badge>
                <StatusBadge status={r.status} />
              </Link>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );

  if (!runId) return picker;
  if (error) return <div className="space-y-4"><Notice tone="danger" title="Could not load run">{error}</Notice>{picker}</div>;
  if (!run) return <Spinner label="Loading trace…" />;

  return (
    <div className="space-y-6">
      <Card className="p-5">
        <p className="text-sm text-muted">Question</p>
        <p className="mt-1 font-medium text-fg">{run.question}</p>
        <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-5">
          <Stat label="Method" value={run.method} />
          <Stat label="Outcome" value={<StatusBadge status={run.status} />} />
          <Stat label="Latency" value={fmt.ms(run.stats.latency_ms)} />
          <Stat label="LLM calls" value={run.stats.llm_calls} hint={`${run.stats.nli_calls} NLI call(s)`} />
          <Stat label="Retrieval attempts" value={run.stats.retrieval_attempts} hint={`${run.stats.retries} retr${run.stats.retries === 1 ? "y" : "ies"}`} />
        </div>
        <p className="mt-4 text-xs text-subtle">Run {run.run_id} · {fmt.date(run.created_at)} · mode: {run.mode} · generator: {run.models.generator}</p>
      </Card>
      {run.method === "standard" && (
        <Notice tone="neutral" title="Standard RAG has no agents">This baseline runs a single retrieval and generation pass. The verifier step shown is post-hoc analysis and did not influence the answer.</Notice>
      )}
      <Card className="p-5"><TraceTimeline trace={run.trace} /></Card>
      <Card className="p-5">
        <p className="text-sm font-medium text-fg">Final answer</p>
        <p className="mt-1 text-sm text-muted">{run.answer}</p>
        <Link href="/trace" className="mt-3 inline-block text-sm text-accent hover:underline">← All recent runs</Link>
      </Card>
    </div>
  );
}
