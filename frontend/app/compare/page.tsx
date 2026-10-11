"use client";

import { GitCompareArrows, Square } from "lucide-react";
import Link from "next/link";
import { useRef, useState } from "react";
import { AnswerCard, EvidenceList, VerificationBadge } from "@/components/result-views";
import { Badge, Button, Card, CardHeader, Notice, PageHeader, Spinner } from "@/components/ui";
import { api, ApiError, fmt } from "@/lib/api";
import type { CompareResponse, QueryResponse } from "@/lib/types";

function Row({ label, std, ada, hint }: { label: string; std: React.ReactNode; ada: React.ReactNode; hint?: string }) {
  return (
    <tr>
      <th scope="row" className="py-2.5 pr-4 text-left font-normal text-muted">{label}{hint && <span className="block text-xs text-subtle">{hint}</span>}</th>
      <td className="py-2.5 pr-4 tabular-nums text-fg">{std}</td>
      <td className="py-2.5 tabular-nums text-fg">{ada}</td>
    </tr>
  );
}

export default function ComparePage() {
  const [question, setQuestion] = useState("");
  const [res, setRes] = useState<CompareResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const ctrl = useRef<AbortController | null>(null);

  async function run(e: React.FormEvent) {
    e.preventDefault();
    if (question.trim().length < 2) return;
    ctrl.current = new AbortController();
    setLoading(true);
    setError(null);
    try {
      setRes(await api.compare(question.trim(), ctrl.current.signal));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  const v = (r: QueryResponse) => (r.verification ? <VerificationBadge status={r.verification.status} /> : "n/a");

  return (
    <>
      <PageHeader title="Compare RAG" description="Run the proposed Multi-Agent RAG and the Standard RAG baseline on the same question, knowledge base, retriever, generator and evidence budget. All numbers are measured for this single run." />
      <Card className="p-4 sm:p-5">
        <form onSubmit={run} className="flex flex-col gap-3 sm:flex-row">
          <label htmlFor="cq" className="sr-only">Question</label>
          <input id="cq" value={question} onChange={(e) => setQuestion(e.target.value)} maxLength={2000}
            placeholder="Question to run through both pipelines"
            className="h-9 flex-1 rounded-md border border-border bg-bg px-3 text-sm text-fg placeholder:text-subtle focus:border-accent focus:outline-none" />
          <div className="flex gap-2">
            {loading && <Button type="button" variant="ghost" onClick={() => ctrl.current?.abort()}><Square className="size-3.5" /> Cancel</Button>}
            <Button type="submit" variant="primary" loading={loading} disabled={question.trim().length < 2}>
              {!loading && <GitCompareArrows className="size-4" />} Compare
            </Button>
          </div>
        </form>
        {loading && <p className="mt-3"><Spinner label="Running Standard RAG, then Adaptive RAG…" /></p>}
      </Card>

      {error && <Notice tone="danger" title="Comparison failed" className="mt-4">{error}</Notice>}

      {res && !loading && (
        <div className="mt-6 space-y-6">
          <Card>
            <CardHeader title="Side-by-side measurements" description={res.note} />
            <div className="overflow-x-auto px-5 py-2">
              <table className="w-full min-w-[480px] text-sm">
                <thead className="text-xs text-subtle">
                  <tr><th className="py-2 text-left font-medium">Measure</th><th className="py-2 text-left font-medium">Standard RAG (baseline)</th><th className="py-2 text-left font-medium">Multi-Agent RAG</th></tr>
                </thead>
                <tbody className="divide-y divide-border">
                  <Row label="Outcome" std={res.standard.status} ada={res.adaptive.status} />
                  <Row label="Latency" hint="pipeline time, excluding post-hoc analysis" std={fmt.ms(res.standard.stats.latency_ms)} ada={fmt.ms(res.adaptive.stats.latency_ms)} />
                  <Row label="LLM calls" std={res.standard.stats.llm_calls} ada={res.adaptive.stats.llm_calls} />
                  <Row label="NLI verifier calls" std={res.standard.stats.nli_calls} ada={res.adaptive.stats.nli_calls} />
                  <Row label="Retrieval attempts" std={res.standard.stats.retrieval_attempts} ada={res.adaptive.stats.retrieval_attempts} />
                  <Row label="Citations" std={res.standard.citations.length} ada={res.adaptive.citations.length} />
                  <Row label="Verification" hint="Standard: post-hoc only" std={v(res.standard)} ada={v(res.adaptive)} />
                  <Row label="Support score" std={fmt.num(res.standard.verification?.support_score)} ada={fmt.num(res.adaptive.verification?.support_score)} />
                  <Row label="Confidence" hint="Standard RAG has no confidence agent" std="not computed" ada={res.adaptive.confidence ? `${fmt.num(res.adaptive.confidence.score)} → ${res.adaptive.confidence.decision}` : "n/a"} />
                </tbody>
              </table>
            </div>
            <p className="border-t border-border px-5 py-3 text-xs text-subtle">
              A single question is an anecdote, not evidence that one method is better. Use the Evaluation page for measured results over a dataset.
            </p>
          </Card>
          <div className="grid gap-6 lg:grid-cols-2">
            {(["standard", "adaptive"] as const).map((m) => {
              const r = res[m];
              return (
                <div key={m} className="min-w-0 space-y-4">
                  <div className="flex items-center justify-between">
                    <h2 className="font-semibold text-fg">{m === "standard" ? "Standard RAG (baseline)" : "Multi-Agent RAG (proposed)"}</h2>
                    <Link href={`/trace?run=${r.run_id}`} className="text-sm text-accent hover:underline">Trace</Link>
                  </div>
                  <AnswerCard r={r} idPrefix={m} />
                  {m === "adaptive" && r.stats.retries > 0 && (
                    <Notice tone="accent" title={`Adaptive retries: ${r.stats.retries}`}>
                      {r.trace.filter((t) => t.step === "confidence" && t.decision === "retry").map((t, i) => <span key={i} className="block">Attempt {t.attempt + 1}: {t.note}</span>)}
                    </Notice>
                  )}
                  <EvidenceList evidence={r.evidence} cited={r.citations.map((c) => c.ref)} idPrefix={m} filtered={r.filtered_evidence ?? []} report={r.evidence_filter} />
                </div>
              );
            })}
          </div>
          <p className="text-xs text-subtle">Mode: <Badge>{res.adaptive.mode}</Badge> · generator {res.adaptive.models.generator}</p>
        </div>
      )}
    </>
  );
}
