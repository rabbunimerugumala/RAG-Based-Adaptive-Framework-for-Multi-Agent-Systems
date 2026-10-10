"use client";

import { Download, Play, Square } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { Badge, Button, Card, CardHeader, EmptyState, Notice, PageHeader, Spinner, Stat, cn } from "@/components/ui";
import { api, ApiError, fmt } from "@/lib/api";
import type { DatasetInfo, EvalJob, EvalRow, Method } from "@/lib/types";
import { MetricsChart } from "./metrics-chart";

const TABLE_METRICS: [string, string, "pct" | "num" | "ms"][] = [
  ["accuracy", "Accuracy (alias match, abstain = wrong)", "pct"],
  ["exact_match", "Exact match", "pct"],
  ["f1", "Token F1", "num"],
  ["answer_rate_answerable", "Answer rate on answerable", "pct"],
  ["selective_accuracy", "Selective accuracy (answered only)", "pct"],
  ["correct_abstention_rate", "Correct abstention (unanswerable)", "pct"],
  ["wrong_abstention_rate", "Wrong abstention (answerable)", "pct"],
  ["citation_gold_rate", "Cites a gold document", "pct"],
  ["evidence_recall", "Evidence recall (answer in retrieved text)", "pct"],
  ["mean_support_score", "Mean support score (own verifier)", "num"],
  ["mean_latency_ms", "Mean latency", "ms"],
  ["median_latency_ms", "Median latency", "ms"],
  ["mean_llm_calls", "Mean LLM calls", "num"],
  ["mean_retrieval_attempts", "Mean retrieval attempts", "num"],
  ["n_errors", "Failed cases", "num"],
];

function show(v: number | null | undefined, kind: "pct" | "num" | "ms") {
  if (v == null) return "n/a";
  return kind === "pct" ? fmt.pct(v, 1) : kind === "ms" ? fmt.ms(v) : Number.isInteger(v) ? String(v) : v.toFixed(2);
}

function Outcome({ r }: { r: EvalRow }) {
  if (r.error) return <Badge tone="danger">error</Badge>;
  if (r.abstained) return <Badge tone={r.answerable ? "warning" : "success"}>{r.answerable ? "abstained (wrong)" : "abstained (correct)"}</Badge>;
  if (!r.answerable) return <Badge tone="danger">answered unanswerable</Badge>;
  return <Badge tone={r.correct ? "success" : "danger"}>{r.correct ? "correct" : "incorrect"}</Badge>;
}

export default function EvaluationPage() {
  const [datasets, setDatasets] = useState<DatasetInfo[] | null>(null);
  const [jobs, setJobs] = useState<EvalJob[]>([]);
  const [dataset, setDataset] = useState("");
  const [limit, setLimit] = useState("");
  const [methods, setMethods] = useState<Method[]>(["standard", "adaptive"]);
  const [job, setJob] = useState<EvalJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  const loadJobs = useCallback(() => api.evaluations().then(setJobs).catch(() => {}), []);

  useEffect(() => {
    api.datasets().then((d) => {
      setDatasets(d);
      setDataset((cur) => cur || d.find((x) => x.valid)?.name || "");
    }).catch((e: Error) => { setDatasets([]); setError(e.message); });
    loadJobs();
  }, [loadJobs]);

  // poll the selected job while it is running
  useEffect(() => {
    if (!job || job.status !== "running") return;
    const id = setInterval(async () => {
      try {
        const j = await api.evaluation(job.job_id);
        setJob(j);
        if (j.status !== "running") loadJobs();
      } catch { /* transient */ }
    }, 1500);
    return () => clearInterval(id);
  }, [job, loadJobs]);

  async function start() {
    setStarting(true);
    setError(null);
    try {
      const n = limit.trim() ? Number(limit) : null;
      if (n != null && (!Number.isInteger(n) || n < 1)) throw new ApiError("Sample size must be a positive whole number.");
      setJob(await api.startEvaluation(dataset, methods, n));
      loadJobs();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setStarting(false);
    }
  }

  const ds = datasets?.find((d) => d.name === dataset);
  const res = job?.result;
  const running = job?.status === "running";

  return (
    <>
      <PageHeader title="Evaluation" description="Run both methods on the same benchmark questions against the same benchmark knowledge base (kept separate from your uploads). Every number shown is computed from the actual run." />

      <div className="grid gap-6 lg:grid-cols-[360px_1fr]">
        <Card className="h-fit">
          <CardHeader title="Configure run" />
          <div className="space-y-4 p-5">
            {datasets == null ? <Spinner label="Loading datasets…" /> : datasets.length === 0 ? (
              <p className="text-sm text-muted">No datasets found in <code className="font-mono">data/datasets/</code>.</p>
            ) : (
              <fieldset className="space-y-2">
                <legend className="mb-2 text-sm font-medium text-fg">Dataset</legend>
                {datasets.map((d) => (
                  <label key={d.name} className={cn("block cursor-pointer rounded-md border p-3 text-sm transition-colors",
                    dataset === d.name ? "border-accent bg-accent-soft" : "border-border hover:bg-surface-2", !d.valid && "opacity-60")}>
                    <input type="radio" name="dataset" className="sr-only" value={d.name} checked={dataset === d.name} disabled={!d.valid} onChange={() => setDataset(d.name)} />
                    <span className="flex flex-wrap items-center gap-2 font-medium text-fg">
                      {d.title ?? d.name}
                      {d.synthetic ? <Badge tone="warning">synthetic</Badge> : <Badge tone="accent">research benchmark sample</Badge>}
                    </span>
                    <span className="mt-1 block text-xs text-muted">{d.num_questions} questions ({d.num_answerable} answerable, {d.num_unanswerable} unanswerable) · {d.num_documents} documents</span>
                  </label>
                ))}
              </fieldset>
            )}
            <div>
              <label htmlFor="limit" className="text-sm font-medium text-fg">Sample size</label>
              <input id="limit" inputMode="numeric" value={limit} onChange={(e) => setLimit(e.target.value)} placeholder={ds ? `all ${ds.num_questions}` : "all"}
                className="mt-1 h-9 w-full rounded-md border border-border bg-bg px-3 text-sm text-fg placeholder:text-subtle focus:border-accent focus:outline-none" />
              <p className="mt-1 text-xs text-subtle">Fixed-seed random sample (seed 0) for reproducibility.</p>
            </div>
            <fieldset>
              <legend className="text-sm font-medium text-fg">Methods</legend>
              <div className="mt-2 flex gap-4 text-sm">
                {(["standard", "adaptive"] as Method[]).map((m) => (
                  <label key={m} className="flex items-center gap-2 text-fg">
                    <input type="checkbox" className="accent-[var(--accent)]" checked={methods.includes(m)}
                      onChange={(e) => setMethods((cur) => e.target.checked ? [...cur, m].sort() as Method[] : cur.filter((x) => x !== m))} />
                    {m === "standard" ? "Standard" : "Adaptive"}
                  </label>
                ))}
              </div>
            </fieldset>
            <Button variant="primary" className="w-full" onClick={start} loading={starting} disabled={!dataset || !methods.length || running}>
              {!starting && <Play className="size-4" />} Run evaluation
            </Button>
            <p className="text-xs text-subtle">With local models a run takes roughly 3–12 s per question per method on the target laptop.</p>
          </div>
          <div className="border-t border-border p-5">
            <p className="mb-2 text-sm font-medium text-fg">Previous runs</p>
            {jobs.length === 0 ? <p className="text-xs text-muted">None yet.</p> : (
              <ul className="space-y-1">
                {jobs.slice(0, 8).map((j) => (
                  <li key={j.job_id}>
                    <button onClick={() => api.evaluation(j.job_id).then(setJob).catch((e: Error) => setError(e.message))}
                      className={cn("w-full rounded px-2 py-1.5 text-left text-xs hover:bg-surface-2", job?.job_id === j.job_id && "bg-surface-2")}>
                      <span className="font-medium text-fg">{j.dataset}</span>
                      <span className="ml-2 text-muted">{j.done}/{j.total ?? "?"} · {j.status}{j.mode ? ` · ${j.mode}` : ""}</span>
                      <span className="block text-subtle">{fmt.date(j.started_at)}</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </Card>

        <div className="min-w-0 space-y-6">
          {error && <Notice tone="danger" title="Evaluation error">{error}</Notice>}
          {!job && !error && (
            <Card><EmptyState title="No evaluation selected">Choose a dataset and start a run, or open a previous run.</EmptyState></Card>
          )}

          {job && (
            <Card>
              <CardHeader title={`Run ${job.job_id}`} description={`${job.dataset} · ${job.methods.join(" vs ")}`}
                action={running ? (
                  <Button size="sm" variant="ghost" onClick={() => api.cancelEvaluation(job.job_id)}><Square className="size-3" /> Cancel</Button>
                ) : <Badge tone={job.status === "completed" ? "success" : job.status === "failed" ? "danger" : "warning"}>{job.status}</Badge>} />
              <div className="space-y-4 p-5">
                {running && (
                  <div>
                    <div className="mb-1 flex justify-between text-xs text-muted"><span>Progress</span><span className="tabular-nums">{job.done}/{job.total ?? "…"}</span></div>
                    <div className="h-2 overflow-hidden rounded-full bg-surface-2" role="progressbar" aria-valuenow={job.done} aria-valuemax={job.total ?? undefined} aria-label="Evaluation progress">
                      <div className="h-full bg-accent transition-all" style={{ width: `${job.total ? (job.done / job.total) * 100 : 2}%` }} />
                    </div>
                    <p className="mt-2 text-xs text-subtle">The first question also builds the benchmark index; this can take a minute.</p>
                  </div>
                )}
                {job.error && <Notice tone={job.status === "cancelled" ? "warning" : "danger"}>{job.error}</Notice>}
                {res && (
                  <>
                    {res.dev_mode_warning && <Notice tone="warning" title="Development mode results">{res.dev_mode_warning}</Notice>}
                    {res.dataset.synthetic && <Notice tone="warning" title="Synthetic dataset">This is the project&apos;s smoke-test set. It verifies the pipeline and must not be reported as a research benchmark result.</Notice>}
                    <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                      <Stat label="Questions" value={res.sample.num_questions} hint={`of ${res.dataset.num_questions_available} available`} />
                      <Stat label="Failed cases" value={Object.values(res.summary).reduce((a, s) => a + (s.n_errors ?? 0), 0)} />
                      <Stat label="Runtime" value={fmt.ms(res.runtime_s * 1000)} />
                      <Stat label="Mode" value={res.mode} hint={res.models.generator} />
                    </div>
                    <div className="flex flex-wrap gap-2">
                      <a href={api.exportUrl(job.job_id, "csv")} className="inline-flex h-8 items-center gap-1.5 rounded-md border border-border px-3 text-xs font-medium text-fg hover:bg-surface-2"><Download className="size-3.5" /> CSV</a>
                      <a href={api.exportUrl(job.job_id, "json")} className="inline-flex h-8 items-center gap-1.5 rounded-md border border-border px-3 text-xs font-medium text-fg hover:bg-surface-2"><Download className="size-3.5" /> JSON (with config)</a>
                    </div>
                  </>
                )}
              </div>
            </Card>
          )}

          {res && (
            <Card>
              <CardHeader title="Results by method" description={`n = ${res.sample.num_questions} questions per method; identical questions for both methods`} />
              <div className="p-5"><MetricsChart summary={res.summary} methods={res.methods} /></div>
              <div className="overflow-x-auto border-t border-border px-5 py-3">
                <table className="w-full min-w-[440px] text-sm">
                  <thead className="text-xs text-subtle"><tr><th className="py-2 text-left font-medium">Metric</th>
                    {res.methods.map((m) => <th key={m} className="py-2 text-right font-medium capitalize">{m}</th>)}</tr></thead>
                  <tbody className="divide-y divide-border">
                    {TABLE_METRICS.map(([key, label, kind]) => (
                      <tr key={key}><td className="py-2 text-muted">{label}</td>
                        {res.methods.map((m) => <td key={m} className="py-2 text-right tabular-nums text-fg">{show(res.summary[m]?.[key], kind)}</td>)}</tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}

          {job?.rows && job.rows.length > 0 && (
            <Card>
              <CardHeader title="Per-question outcomes" description={`${job.rows.length} result row(s)`} />
              <div className="max-h-[520px] overflow-auto">
                <table className="w-full min-w-[640px] text-sm">
                  <thead className="sticky top-0 bg-surface text-xs text-subtle">
                    <tr><th className="px-5 py-2 text-left font-medium">Question</th><th className="px-3 py-2 text-left font-medium">Method</th>
                      <th className="px-3 py-2 text-left font-medium">Outcome</th><th className="px-3 py-2 text-right font-medium">Attempts</th><th className="px-5 py-2 text-right font-medium">Latency</th></tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {job.rows.map((r, i) => (
                      <tr key={i} className="align-top">
                        <td className="px-5 py-2"><span className="text-fg">{r.question}</span>
                          <span className="mt-0.5 block text-xs text-subtle">{r.answerable ? `gold: ${r.gold_answers.slice(0, 3).join(" | ")}` : "unanswerable"}</span>
                          {r.error && <span className="block text-xs text-danger">{r.error}</span>}</td>
                        <td className="px-3 py-2 capitalize text-muted">{r.method}</td>
                        <td className="px-3 py-2"><Outcome r={r} /></td>
                        <td className="px-3 py-2 text-right tabular-nums text-muted">{r.retrieval_attempts}</td>
                        <td className="px-5 py-2 text-right tabular-nums text-muted">{fmt.ms(r.latency_ms)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}

          <Card>
            <CardHeader title="Limitations of these measurements" />
            <ul className="list-disc space-y-1 px-9 py-4 text-sm text-muted">
              <li>Accuracy uses normalised alias matching; it can miss correct paraphrases and accept answers that mention the alias in the wrong context.</li>
              <li>The support score comes from this project&apos;s own verifier, so it is not an independent faithfulness measure.</li>
              <li>Confidence scores are heuristics and are not calibrated probabilities.</li>
              <li>Small samples give wide uncertainty; differences of a few points between methods may not be meaningful.</li>
              <li>The TriviaQA sample uses truncated evidence; &quot;unanswerable&quot; means the answer is absent from the retained text.</li>
            </ul>
          </Card>
        </div>
      </div>
    </>
  );
}
