"use client";

import { ArrowRight, BarChart3, FileUp, MessageSquareText } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";
import { useHealth } from "@/components/providers";
import { StatusBadge } from "@/components/result-views";
import { Badge, Card, CardHeader, EmptyState, Notice, PageHeader, Spinner, Stat } from "@/components/ui";
import { api, fmt } from "@/lib/api";
import type { RunSummary } from "@/lib/types";

const AGENTS = [
  ["Planner", "Classifies the question, extracts keywords and focus terms, splits multi-part questions and sets retrieval parameters."],
  ["Evidence", "Runs every planned query with dense (Qdrant) + BM25 search and fuses rankings with reciprocal-rank fusion."],
  ["Verifier", "Splits the draft answer into claims and checks each against the evidence (lexical + local NLI model)."],
  ["Confidence", "Combines relevance, coverage, verification and citation consistency into a heuristic score: accept, retry or abstain."],
];

export default function OverviewPage() {
  const { health, error, loading } = useHealth();
  const [runs, setRuns] = useState<RunSummary[] | null>(null);

  useEffect(() => {
    api.runs(8).then(setRuns).catch(() => setRuns([]));
  }, []);

  return (
    <>
      <PageHeader
        title="RAG-Based Adaptive Framework for Multi-Agent Systems"
        description="A research prototype comparing Standard Retrieval-Augmented Generation with an adaptive pipeline of four agents (Planner, Evidence, Verifier, Confidence) that verifies answers against retrieved documents and retries retrieval or abstains when support is weak."
      />

      {error && !health && (
        <Notice tone="danger" title="Backend not reachable" className="mb-6">
          {error} Start it with <code className="font-mono">uvicorn adaptive_rag.api.app:app --port 8000</code>.
        </Notice>
      )}
      {health?.dev_mode_warning && <Notice tone="warning" title="Development mode" className="mb-6">{health.dev_mode_warning}</Notice>}

      <div className="grid gap-4 md:grid-cols-3">
        <Card className="p-5">
          {loading && !health ? <Spinner label="Loading…" /> : (
            <Stat label="Knowledge base" value={health ? `${health.knowledge_base.documents} documents` : "n/a"}
              hint={health ? `${health.knowledge_base.chunks} indexed chunks` : undefined} />
          )}
        </Card>
        <Card className="p-5">
          <Stat label="Run mode" value={health ? (health.mode === "dev" ? "Development" : "Local models") : "n/a"}
            hint={health ? `Top-k ${health.config.top_k} · max ${health.config.max_retries} retries` : undefined} />
        </Card>
        <Card className="p-5">
          <Stat label="Thresholds" value={health ? `accept ≥ ${health.config.accept_threshold}` : "n/a"}
            hint={health ? `abstain below ${health.config.abstain_threshold} (experimental settings)` : undefined} />
        </Card>
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-[1fr_1.2fr]">
        <Card>
          <CardHeader title="Model readiness" description="Reported live by the backend" />
          <ul className="divide-y divide-border">
            {health ? (Object.entries(health.models) as [string, { name: string; ready: boolean; detail: string }][]).map(([role, m]) => (
              <li key={role} className="flex items-start justify-between gap-3 px-5 py-3">
                <div className="min-w-0">
                  <p className="text-sm font-medium text-fg">{({ embedder: "Embedder", generator: "Generator (LLM)", nli: "NLI verifier", reranker: "Reranker" } as Record<string, string>)[role] ?? role}</p>
                  <p className="truncate text-xs text-muted" title={m.name}>{m.name}</p>
                  <p className="text-xs text-subtle">{m.detail}</p>
                </div>
                <Badge tone={m.ready ? "success" : role === "generator" ? "danger" : "neutral"}>{m.ready ? "ready" : role === "reranker" || role === "nli" ? "off" : "not ready"}</Badge>
              </li>
            )) : <li className="px-5 py-6"><Spinner label="Waiting for backend…" /></li>}
          </ul>
        </Card>

        <Card>
          <CardHeader title="Quick actions" />
          <div className="grid gap-3 p-5 sm:grid-cols-3">
            {[
              { href: "/documents", icon: FileUp, title: "Upload", text: "Add PDF, DOCX, TXT or Markdown files" },
              { href: "/ask", icon: MessageSquareText, title: "Ask", text: "Question your documents" },
              { href: "/evaluation", icon: BarChart3, title: "Evaluate", text: "Run a benchmark" },
            ].map(({ href, icon: Icon, title, text }) => (
              <Link key={href} href={href} className="group rounded-md border border-border p-4 transition-colors hover:border-accent hover:bg-accent-soft">
                <Icon className="size-5 text-accent" aria-hidden />
                <p className="mt-3 flex items-center gap-1 text-sm font-medium text-fg">{title} <ArrowRight className="size-3.5 opacity-0 transition-opacity group-hover:opacity-100" /></p>
                <p className="mt-0.5 text-xs text-muted">{text}</p>
              </Link>
            ))}
          </div>
          <div className="border-t border-border px-5 py-4">
            <p className="mb-3 text-xs font-medium uppercase tracking-wide text-subtle">The four agents</p>
            <dl className="grid gap-3 sm:grid-cols-2">
              {AGENTS.map(([name, text], i) => (
                <div key={name}>
                  <dt className="text-sm font-medium text-fg">{i + 1}. {name} Agent</dt>
                  <dd className="mt-0.5 text-xs leading-5 text-muted">{text}</dd>
                </div>
              ))}
            </dl>
          </div>
        </Card>
      </div>

      <Card className="mt-6">
        <CardHeader title="Recent query activity" description="Saved runs from this backend" />
        {runs == null ? <div className="px-5 py-6"><Spinner label="Loading…" /></div> : runs.length === 0 ? (
          <EmptyState title="No queries yet">Ask a question on the Ask Documents page; runs and their agent traces will appear here.</EmptyState>
        ) : (
          <ul className="divide-y divide-border">
            {runs.map((r) => (
              <li key={r.run_id}>
                <Link href={`/trace?run=${r.run_id}`} className="flex flex-wrap items-center gap-x-3 gap-y-1 px-5 py-3 hover:bg-surface-2">
                  <span className="min-w-0 flex-1 truncate text-sm text-fg">{r.question}</span>
                  <Badge tone={r.method === "adaptive" ? "accent" : "neutral"}>{r.method}</Badge>
                  <StatusBadge status={r.status} />
                  <span className="w-16 text-right text-xs tabular-nums text-muted">{fmt.ms(r.latency_ms)}</span>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </Card>
    </>
  );
}
