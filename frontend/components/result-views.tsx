"use client";

// Views that render a real backend QueryResponse: answer, citations, evidence, verification, confidence, trace.
import { CheckCircle2, ChevronDown, CircleSlash, FileText, MinusCircle, XCircle } from "lucide-react";
import { Fragment, useState } from "react";
import { fmt } from "@/lib/api";
import type { ClaimLabel, ConfidenceAssessment, Evidence, EvidenceFilterReport, QueryResponse, TraceStep, VerificationResult } from "@/lib/types";
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

const SUPPORT_META: Record<string, { label: string; tone: Tone; hint: string }> = {
  verified: { label: "Verified answer", tone: "success", hint: "Every claim is supported by the retrieved evidence." },
  partial: { label: "Partially supported", tone: "warning", hint: "Some claims could not be confirmed from the evidence." },
  unverified: { label: "Unverified", tone: "danger", hint: "Accepted with weak support." },
};

export function SupportBadge({ r }: { r: QueryResponse }) {
  if (r.method !== "adaptive" || r.status !== "answered") return <StatusBadge status={r.status} />;
  const m = SUPPORT_META[r.support_level] ?? SUPPORT_META.unverified;
  return <Badge tone={m.tone} title={m.hint}>{m.label}</Badge>;
}

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
        description={`${r.method === "adaptive" ? "Multi-Agent RAG" : "Standard RAG (baseline)"} · ${fmt.ms(r.stats.latency_ms)} · ${r.stats.llm_calls} LLM call(s) · ${r.stats.retrieval_attempts} retrieval round(s)${r.route && r.route !== "full" ? ` · route: ${r.route}` : ""}`}
        action={<SupportBadge r={r} />}
      />
      <div className="space-y-4 px-5 py-4">
        {r.status === "no_documents" ? (
          <Notice tone="accent" title="The knowledge base is empty">{r.answer}</Notice>
        ) : r.status === "abstained" ? (
          <Notice tone="warning" title="The system abstained">{r.answer}</Notice>
        ) : (
          <AnswerText text={r.answer} idPrefix={idPrefix} />
        )}
        {r.status === "answered" && r.support_level === "partial" && (r.unresolved_claims?.length ?? 0) > 0 && (
          <Notice tone="warning" title="Unresolved parts (not confirmed by the evidence)">
            <ul className="list-disc space-y-0.5 pl-4">
              {r.unresolved_claims.map((c, i) => <li key={i}>{c}</li>)}
            </ul>
          </Notice>
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

function PassageItem({ e, cited, idPrefix, filtered }: { e: Evidence; cited: boolean; idPrefix: string; filtered?: boolean }) {
  const [open, setOpen] = useState(false);
  return (
    <li id={filtered ? undefined : `${idPrefix}-ev-${e.ref}`} className={cn("scroll-mt-20 px-5 py-3", filtered && "opacity-80")}>
      <button className="flex w-full items-start gap-3 text-left" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className={cn("mt-0.5 inline-flex size-6 shrink-0 items-center justify-center rounded text-xs font-semibold",
          filtered ? "bg-surface-2 text-subtle" : cited ? "bg-accent text-accent-fg" : "bg-surface-2 text-muted")}>
          {filtered ? "–" : e.ref}
        </span>
        <span className="min-w-0 flex-1">
          <span className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm">
            <span className="font-medium text-fg">{e.filename}</span>
            <span className="text-xs text-muted">{e.page ? `page ${e.page} · ` : ""}chunk {e.chunk_index}</span>
            {cited && !filtered && <Badge tone="accent">cited</Badge>}
            {e.judge_score != null && (
              <Badge tone={e.judge_score > 0 ? "success" : "neutral"} title="Evidence-judge log-odds: relevance, not proof of truth">
                judge {e.judge_score > 0 ? "+" : ""}{e.judge_score.toFixed(1)}
              </Badge>
            )}
            {filtered && <Badge tone="warning">filtered out</Badge>}
          </span>
          {filtered && e.filter_reason && <span className="mt-0.5 block text-xs text-warning">{e.filter_reason}</span>}
          <span className={cn("mt-1 block text-sm text-muted", !open && "line-clamp-2")}>{e.text}</span>
        </span>
        <ChevronDown className={cn("mt-1 size-4 shrink-0 text-subtle transition-transform", open && "rotate-180")} aria-hidden />
      </button>
      {open && (
        <dl className="ml-9 mt-3 grid grid-cols-2 gap-x-6 gap-y-1 text-xs sm:grid-cols-6">
          {[["Retrieval rank", e.retrieval_rank ?? "–"], ["RRF", fmt.num(e.rrf_score, 4)], ["Dense cosine", fmt.num(e.dense_score, 3)],
            ["BM25", fmt.num(e.bm25_score, 2)], ["Judge log-odds", e.judge_score == null ? "off" : fmt.num(e.judge_score, 2)],
            ["Keyword overlap", fmt.pct(e.keyword_overlap)]].map(([k, v]) => (
            <div key={String(k)}><dt className="text-subtle">{k}</dt><dd className="font-medium tabular-nums text-fg">{v}</dd></div>
          ))}
          {e.candidate_answer && (
            <div className="col-span-full"><dt className="text-subtle">Per-passage predictor answer</dt><dd className="text-fg">{e.candidate_answer}</dd></div>
          )}
        </dl>
      )}
    </li>
  );
}

export function EvidenceList({ evidence, cited, idPrefix = "r", filtered = [], report = null }: {
  evidence: Evidence[]; cited: number[]; idPrefix?: string; filtered?: Evidence[]; report?: EvidenceFilterReport | null;
}) {
  const [showFiltered, setShowFiltered] = useState(false);
  if (!evidence.length && !filtered.length) return null;
  const description = report
    ? `Evidence Agent judged ${report.pool_size} retrieved passage(s) and kept ${report.retained} at or above the adaptive bar τq${report.threshold != null ? ` = ${report.threshold.toFixed(2)}` : ""} (mean − ${report.n_sigma}·σ), ordered by judge score.`
    : "Hybrid dense + BM25 retrieval fused with RRF (no evidence filtering).";
  return (
    <Card>
      <CardHeader title="Evidence explorer" description={description} />
      {report?.fallback && <div className="px-5 pt-3"><Notice tone="warning">{report.fallback}</Notice></div>}
      <p className="px-5 pt-3 text-xs text-subtle">Retrieval and judge scores measure relevance to the question. Factual support is checked separately by the Verifier Agent.</p>
      <ul className="divide-y divide-border">
        {evidence.map((e) => <PassageItem key={e.chunk_id} e={e} cited={cited.includes(e.ref)} idPrefix={idPrefix} />)}
      </ul>
      {filtered.length > 0 && (
        <div className="border-t border-border">
          <button onClick={() => setShowFiltered(!showFiltered)} className="w-full px-5 py-3 text-left text-sm font-medium text-accent hover:underline" aria-expanded={showFiltered}>
            {showFiltered ? "Hide" : "Show"} {filtered.length} filtered-out passage(s)
          </button>
          {showFiltered && (
            <ul className="divide-y divide-border">
              {filtered.map((e) => <PassageItem key={e.chunk_id} e={e} cited={false} idPrefix={idPrefix} filtered />)}
            </ul>
          )}
        </div>
      )}
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
