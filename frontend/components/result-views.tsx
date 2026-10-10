"use client";

// Views that render a real backend QueryResponse: answer, citations, evidence, verification, confidence, trace.
import { CheckCircle2, ChevronDown, CircleSlash, FileText, MinusCircle, XCircle } from "lucide-react";
import { Fragment, useState } from "react";
import { fmt } from "@/lib/api";
import type { ClaimLabel, ConfidenceAssessment, Evidence, QueryResponse, TraceStep, VerificationResult } from "@/lib/types";
import { Badge, Card, CardHeader, Meter, Notice, cn, type Tone } from "./ui";

export const STATUS_META: Record<string, { label: string; tone: Tone }> = {
  answered: { label: "Answered", tone: "success" },
  abstained: { label: "Abstained", tone: "warning" },
  no_documents: { label: "No documents", tone: "neutral" },
  no_retrieval_needed: { label: "No retrieval needed", tone: "neutral" },
  error: { label: "Error", tone: "danger" },
};

export const VERIFY_META: Record<string, { label: string; tone: Tone; icon: typeof CheckCircle2 }> = {
  supported: { label: "Supported", tone: "success", icon: CheckCircle2 },
  partially_supported: { label: "Partially supported", tone: "warning", icon: MinusCircle },
  unsupported: { label: "Unsupported", tone: "danger", icon: XCircle },
  contradicted: { label: "Contradicted", tone: "danger", icon: XCircle },
  not_applicable: { label: "Not applicable", tone: "neutral", icon: CircleSlash },
};

const DECISION_TONE: Record<string, Tone> = { accept: "success", retry: "accent", abstain: "warning" };

export function StatusBadge({ status }: { status: string }) {
  const m = STATUS_META[status] ?? { label: status, tone: "neutral" as Tone };
  return <Badge tone={m.tone}>{m.label}</Badge>;
}

export function VerificationBadge({ status }: { status: string }) {
  const m = VERIFY_META[status] ?? VERIFY_META.not_applicable;
  const Icon = m.icon;
  return (
    <Badge tone={m.tone}>
      <Icon className="size-3" aria-hidden /> {m.label}
    </Badge>
  );
}

/** Renders answer text, turning [n] markers into links to the evidence passage. */
export function AnswerText({ text, idPrefix }: { text: string; idPrefix: string }) {
  const parts = text.split(/(\[\d+(?:\s*,\s*\d+)*\])/g);
  return (
    <p className="text-[15px] leading-7 text-fg">
      {parts.map((p, i) => {
        const m = p.match(/^\[([\d,\s]+)\]$/);
        if (!m) return <Fragment key={i}>{p}</Fragment>;
        return m[1].split(",").map((n) => (
          <a key={`${i}-${n}`} href={`#${idPrefix}-ev-${n.trim()}`}
            className="mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded bg-accent-soft px-1 align-text-top text-[11px] font-semibold text-accent hover:underline">
            {n.trim()}
          </a>
        ));
      })}
    </p>
  );
}

export function AnswerCard({ r, idPrefix = "r" }: { r: QueryResponse; idPrefix?: string }) {
  return (
    <Card>
      <CardHeader
        title="Answer"
        description={`${r.method === "adaptive" ? "Adaptive multi-agent RAG" : "Standard RAG"} · ${fmt.ms(r.stats.latency_ms)} · ${r.stats.llm_calls} LLM call(s)`}
        action={<StatusBadge status={r.status} />}
      />
      <div className="space-y-4 px-5 py-4">
        {r.status === "no_documents" ? (
          <Notice tone="accent" title="The knowledge base is empty">{r.answer}</Notice>
        ) : r.status === "abstained" ? (
          <Notice tone="warning" title="The system abstained">{r.answer}</Notice>
        ) : (
          <AnswerText text={r.answer} idPrefix={idPrefix} />
        )}
        {r.mode === "dev" && (
          <p className="text-xs text-warning">Development mode: answer produced by the extractive generator, not a language model.</p>
        )}
        {r.citations.length > 0 && (
          <div>
            <p className="mb-2 text-xs font-medium uppercase tracking-wide text-subtle">Citations</p>
            <ul className="flex flex-wrap gap-2">
              {r.citations.map((c) => (
                <li key={c.ref}>
                  <a href={`#${idPrefix}-ev-${c.ref}`} className="inline-flex items-center gap-1.5 rounded-md border border-border bg-surface-2 px-2 py-1 text-xs text-fg hover:border-border-strong">
                    <span className="font-semibold text-accent">[{c.ref}]</span>
                    <FileText className="size-3 text-subtle" aria-hidden />
                    {c.filename}
                    <span className="text-muted">{c.page ? `p. ${c.page} · ` : ""}chunk {c.chunk_index}</span>
                  </a>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </Card>
  );
}

export function EvidenceList({ evidence, cited, idPrefix = "r" }: { evidence: Evidence[]; cited: number[]; idPrefix?: string }) {
  const [open, setOpen] = useState<number | null>(null);
  if (!evidence.length) return null;
  return (
    <Card>
      <CardHeader title="Retrieved evidence" description="Hybrid dense + BM25 retrieval, fused with RRF. Scores measure relevance to the query, not truth." />
      <ul className="divide-y divide-border">
        {evidence.map((e) => {
          const isOpen = open === e.ref;
          return (
            <li key={e.chunk_id} id={`${idPrefix}-ev-${e.ref}`} className="scroll-mt-20 px-5 py-3">
              <button className="flex w-full items-start gap-3 text-left" onClick={() => setOpen(isOpen ? null : e.ref)} aria-expanded={isOpen}>
                <span className={cn("mt-0.5 inline-flex size-6 shrink-0 items-center justify-center rounded text-xs font-semibold",
                  cited.includes(e.ref) ? "bg-accent text-accent-fg" : "bg-surface-2 text-muted")}>{e.ref}</span>
                <span className="min-w-0 flex-1">
                  <span className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm">
                    <span className="font-medium text-fg">{e.filename}</span>
                    <span className="text-xs text-muted">{e.page ? `page ${e.page} · ` : ""}chunk {e.chunk_index}</span>
                    {cited.includes(e.ref) && <Badge tone="accent">cited</Badge>}
                  </span>
                  <span className={cn("mt-1 block text-sm text-muted", !isOpen && "line-clamp-2")}>{e.text}</span>
                </span>
                <ChevronDown className={cn("mt-1 size-4 shrink-0 text-subtle transition-transform", isOpen && "rotate-180")} aria-hidden />
              </button>
              {isOpen && (
                <dl className="ml-9 mt-3 grid grid-cols-2 gap-x-6 gap-y-1 text-xs sm:grid-cols-5">
                  {[["RRF", fmt.num(e.rrf_score, 4)], ["Dense cosine", fmt.num(e.dense_score, 3)], ["BM25", fmt.num(e.bm25_score, 2)],
                    ["Rerank", e.rerank_score == null ? "off" : fmt.num(e.rerank_score, 2)], ["Keyword overlap", fmt.pct(e.keyword_overlap)]].map(([k, v]) => (
                    <div key={k}><dt className="text-subtle">{k}</dt><dd className="font-medium tabular-nums text-fg">{v}</dd></div>
                  ))}
                </dl>
              )}
            </li>
          );
        })}
      </ul>
    </Card>
  );
}

const CLAIM_TONE: Record<ClaimLabel, Tone> = { supported: "success", partially_supported: "warning", unsupported: "danger", contradicted: "danger" };

export function VerificationCard({ v, usedInDecision }: { v: VerificationResult; usedInDecision: boolean }) {
  return (
    <Card>
      <CardHeader
        title="Verifier Agent"
        description={usedInDecision ? v.method : `${v.method} · post-hoc analysis only (not used by Standard RAG)`}
        action={<VerificationBadge status={v.status} />}
      />
      <div className="space-y-3 px-5 py-4">
        <p className="text-sm text-muted">{v.summary}</p>
        {v.claims.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-xs text-subtle">
                <tr><th className="pb-2 pr-3 font-medium">Claim</th><th className="pb-2 pr-3 font-medium">Label</th>
                  <th className="pb-2 pr-3 font-medium">Lexical</th><th className="pb-2 pr-3 font-medium">NLI entail</th><th className="pb-2 font-medium">Best ev.</th></tr>
              </thead>
              <tbody className="divide-y divide-border">
                {v.claims.map((c, i) => (
                  <tr key={i} className="align-top">
                    <td className="py-2 pr-3 text-fg">{c.claim}<p className="mt-0.5 text-xs text-subtle">{c.reason}</p></td>
                    <td className="py-2 pr-3"><Badge tone={CLAIM_TONE[c.label]}>{c.label.replace("_", " ")}</Badge></td>
                    <td className="py-2 pr-3 tabular-nums">{c.lexical_support.toFixed(2)}</td>
                    <td className="py-2 pr-3 tabular-nums">{c.nli_entailment == null ? "n/a" : c.nli_entailment.toFixed(2)}</td>
                    <td className="py-2 tabular-nums">{c.best_evidence_ref ? `[${c.best_evidence_ref}]` : "–"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </Card>
  );
}

const COMPONENT_LABEL: Record<string, string> = { relevance: "Relevance", coverage: "Coverage", verification: "Verification", consistency: "Citation consistency" };

export function ConfidenceCard({ c }: { c: ConfidenceAssessment }) {
  return (
    <Card>
      <CardHeader title="Confidence Agent" description={c.note}
        action={<Badge tone={DECISION_TONE[c.decision]}>Decision: {c.decision}</Badge>} />
      <div className="grid gap-5 px-5 py-4 sm:grid-cols-[160px_1fr]">
        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-subtle">Score</p>
          <p className="mt-1 text-3xl font-semibold tabular-nums text-fg">{c.score == null ? "n/a" : c.score.toFixed(2)}</p>
          <p className="mt-1 text-xs text-muted">weighted mean of available components</p>
        </div>
        <div className="space-y-3">
          {Object.entries(c.components).map(([k, v]) => (
            <Meter key={k} value={v} label={`${COMPONENT_LABEL[k] ?? k} (w=${c.weights[k]})`} />
          ))}
        </div>
        <p className="text-sm text-muted sm:col-span-2"><span className="font-medium text-fg">Reason: </span>{c.reason}
          {c.missing_components.length > 0 && <> Missing components (excluded from the mean): {c.missing_components.join(", ")}.</>}
        </p>
      </div>
    </Card>
  );
}

const STEP_LABEL: Record<string, string> = {
  planner: "Planner Agent", evidence: "Evidence Agent", generator: "Answer generator", verifier: "Verifier Agent",
  confidence: "Confidence Agent", retrieval: "Retrieval", citation_check: "Citation check",
};
const STEP_STATUS_TONE: Record<string, Tone> = { completed: "success", skipped: "neutral", failed: "danger" };

function JsonBlock({ data }: { data: unknown }) {
  return (
    <pre className="max-h-80 overflow-auto rounded-md bg-code p-3 font-mono text-xs leading-5 text-fg">{JSON.stringify(data, null, 2)}</pre>
  );
}

function StepItem({ s }: { s: TraceStep }) {
  const [open, setOpen] = useState(false);
  return (
    <li className="relative pl-8">
      <span className={cn("absolute left-1.5 top-1.5 size-3 rounded-full border-2 border-surface ring-1",
        s.status === "completed" ? "bg-success ring-success" : s.status === "failed" ? "bg-danger ring-danger" : "bg-subtle ring-subtle")} aria-hidden />
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-medium text-fg">{s.title || STEP_LABEL[s.step]}</span>
        <Badge tone={STEP_STATUS_TONE[s.status]}>{s.status}</Badge>
        {s.decision && <Badge tone={DECISION_TONE[s.decision] ?? "neutral"}>{s.decision}</Badge>}
        <span className="text-xs tabular-nums text-subtle">{fmt.ms(s.duration_ms)}</span>
      </div>
      {s.note && <p className="mt-1 text-sm text-muted">{s.note}</p>}
      {(Object.keys(s.input).length > 0 || Object.keys(s.output).length > 0) && (
        <button onClick={() => setOpen(!open)} className="mt-1 text-xs font-medium text-accent hover:underline" aria-expanded={open}>
          {open ? "Hide" : "Show"} inputs and outputs
        </button>
      )}
      {open && (
        <div className="mt-2 grid gap-3 lg:grid-cols-2">
          <div><p className="mb-1 text-xs text-subtle">Input</p><JsonBlock data={s.input} /></div>
          <div><p className="mb-1 text-xs text-subtle">Output</p><JsonBlock data={s.output} /></div>
        </div>
      )}
    </li>
  );
}

export function TraceTimeline({ trace }: { trace: TraceStep[] }) {
  const attempts = [...new Set(trace.map((t) => t.attempt))];
  return (
    <div className="space-y-6">
      {attempts.map((a) => (
        <section key={a}>
          <h3 className="mb-3 text-xs font-semibold uppercase tracking-wide text-subtle">
            {a === 0 ? "Attempt 1 (initial)" : `Attempt ${a + 1} (adaptive retry ${a})`}
          </h3>
          <ol className="relative space-y-4 before:absolute before:bottom-2 before:left-[11px] before:top-2 before:w-px before:bg-border">
            {trace.filter((t) => t.attempt === a).map((s, i) => <StepItem key={i} s={s} />)}
          </ol>
        </section>
      ))}
    </div>
  );
}
