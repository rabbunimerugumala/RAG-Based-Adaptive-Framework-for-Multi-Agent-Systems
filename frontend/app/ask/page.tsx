"use client";

import { Network, Send, Square } from "lucide-react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useRef, useState } from "react";
import { useHealth } from "@/components/providers";
import { AnswerCard, ConfidenceCard, EvidenceList, VerificationCard } from "@/components/result-views";
import { Button, Card, Notice, PageHeader, SegmentedControl, Spinner } from "@/components/ui";
import { api, ApiError } from "@/lib/api";
import type { Method, QueryResponse } from "@/lib/types";

const LAST_KEY = "arag-last-ask";

export default function AskPage() {
  return <Suspense fallback={<Spinner label="Loading…" />}><AskView /></Suspense>;
}

function AskView() {
  const runParam = useSearchParams().get("run");
  const { health } = useHealth();
  const [question, setQuestion] = useState("");
  const [method, setMethod] = useState<Method>("adaptive");
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const ctrl = useRef<AbortController | null>(null);

  // deep link /ask?run=<id> shows a saved run; otherwise restore the last result of this browser session
  useEffect(() => {
    if (runParam) {
      api.run(runParam).then(setResult).catch((e: Error) => setError(e.message));
      return;
    }
    try {
      const saved = sessionStorage.getItem(LAST_KEY);
      // eslint-disable-next-line react-hooks/set-state-in-effect -- one-time restore from browser storage
      if (saved) setResult(JSON.parse(saved));
    } catch { /* storage unavailable */ }
  }, [runParam]);

  async function submit(e?: React.FormEvent) {
    e?.preventDefault();
    if (question.trim().length < 2 || loading) return;
    ctrl.current = new AbortController();
    setLoading(true);
    setError(null);
    try {
      const r = await api.query(question.trim(), method, ctrl.current.signal);
      setResult(r);
      try { sessionStorage.setItem(LAST_KEY, JSON.stringify(r)); } catch { /* ignore */ }
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  const emptyKb = health && health.knowledge_base.documents === 0;

  return (
    <>
      <PageHeader title="Ask Documents" description="Ask a question about your documents. The four-agent Multi-Agent RAG plans the search, filters the evidence, verifies every claim, and then answers, retrieves again or abstains. Standard RAG is available as the comparison baseline." />

      {emptyKb && (
        <Notice tone="accent" title="Your knowledge base is empty" className="mb-4">
          Upload documents in the <Link href="/documents" className="font-medium text-accent underline">Document Library</Link> first. Without documents the system will not generate an answer.
        </Notice>
      )}

      <Card className="p-4 sm:p-5">
        <form onSubmit={submit} className="space-y-3">
          <label htmlFor="q" className="sr-only">Question</label>
          <textarea
            id="q"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) submit(); }}
            rows={3}
            maxLength={2000}
            placeholder="e.g. What is the energy density of the Kestrel-9 battery?"
            className="w-full resize-y rounded-md border border-border bg-bg px-3 py-2 text-sm text-fg placeholder:text-subtle focus:border-accent focus:outline-none"
          />
          <div className="flex flex-wrap items-center justify-between gap-3">
            <SegmentedControl label="RAG method" value={method} onChange={setMethod}
              options={[{ value: "adaptive", label: "Multi-Agent RAG" }, { value: "standard", label: "Standard RAG (baseline)" }]} />
            <div className="flex items-center gap-2">
              <span className="hidden text-xs text-subtle sm:inline">Ctrl + Enter</span>
              {loading && (
                <Button type="button" variant="ghost" onClick={() => ctrl.current?.abort()}>
                  <Square className="size-3.5" /> Cancel
                </Button>
              )}
              <Button type="submit" variant="primary" loading={loading} disabled={question.trim().length < 2}>
                {!loading && <Send className="size-4" />} Ask
              </Button>
            </div>
          </div>
        </form>
        {loading && (
          <p className="mt-3 text-xs text-muted">
            <Spinner label={method === "adaptive" ? "Planner → Evidence (retrieve, judge, filter) → Draft → Verifier → Confidence; may retrieve again up to twice…" : "Retrieving and generating (baseline)…"} />
          </p>
        )}
        {loading && <p className="mt-1 text-xs text-subtle">Cancel stops waiting in the browser; a run already started on the backend still finishes and is saved.</p>}
      </Card>

      {error && <Notice tone="danger" title="Query failed" className="mt-4">{error}</Notice>}

      {result && !loading && (
        <div className="mt-6 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="text-sm text-muted">Question: <span className="text-fg">{result.question}</span></p>
            {result.trace.length > 0 && (
              <Link href={`/trace?run=${result.run_id}`} className="inline-flex items-center gap-1.5 text-sm font-medium text-accent hover:underline">
                <Network className="size-4" /> View agent trace
              </Link>
            )}
          </div>
          <AnswerCard r={result} />
          <div className="grid gap-4 lg:grid-cols-2">
            {result.confidence && <ConfidenceCard c={result.confidence} />}
            {result.verification && <VerificationCard v={result.verification} usedInDecision={result.verification_used_in_decision} />}
          </div>
          <EvidenceList evidence={result.evidence} cited={result.citations.map((c) => c.ref)} filtered={result.filtered_evidence ?? []} report={result.evidence_filter} />
        </div>
      )}
    </>
  );
}
