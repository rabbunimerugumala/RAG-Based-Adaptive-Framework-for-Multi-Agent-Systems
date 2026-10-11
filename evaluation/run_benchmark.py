"""Run Standard RAG and Adaptive RAG on a benchmark dataset and write results to evaluation/results/<run_id>/.

Usage (PowerShell, project root, venv active):
    python evaluation/run_benchmark.py --dataset smoke_test                 # mode from .env / ARAG_MODE
    python evaluation/run_benchmark.py --dataset smoke_test --mode real
    python evaluation/run_benchmark.py --dataset triviaqa_unfiltered_sample --limit 50 --mode real

Stop the API server first if it is running a benchmark on the same dataset: Qdrant local mode allows
only one process to open a knowledge-base folder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_rag.config import Settings  # noqa: E402
from adaptive_rag.evaluation.runner import run_evaluation, save_results  # noqa: E402
from adaptive_rag.models import build_models  # noqa: E402

COLUMNS = [("accuracy", "Accuracy (alias match)"), ("exact_match", "Exact match"), ("f1", "Token F1"),
           ("answer_rate_answerable", "Answer rate (answerable)"), ("selective_accuracy", "Selective accuracy"),
           ("correct_abstention_rate", "Correct abstention (unanswerable)"),
           ("wrong_abstention_rate", "Wrong abstention (answerable)"), ("citation_gold_rate", "Cites gold doc"),
           ("evidence_recall", "Evidence recall"), ("mean_support_score", "Mean support score*"),
           ("mean_latency_ms", "Mean latency (ms)"), ("median_latency_ms", "Median latency (ms)"),
           ("mean_llm_calls", "Mean LLM calls"), ("mean_retrieval_attempts", "Mean retrieval attempts"),
           ("abstention_rate", "Abstention rate (all)"), ("unsupported_answer_rate", "Unsupported-answer rate"),
           ("evidence_precision", "Evidence precision"), ("grounded_rate", "Grounded answers (lexical)"),
           ("verified_rate", "Verified answers (own verifier)"), ("cold_latency_ms", "Cold-start latency (ms)"),
           ("warm_mean_latency_ms", "Warm mean latency (ms)"), ("n_errors", "Failed cases")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--methods", default="standard,adaptive",
                    help="comma list of: standard, adaptive, adaptive_no_retry, adaptive_no_filter, adaptive_v1")
    ap.add_argument("--repeat", type=int, default=0, help="re-run the first N questions to measure consistency")
    ap.add_argument("--limit", type=int, default=None, help="fixed-seed random sample of N questions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["dev", "real"], default=None, help="overrides ARAG_MODE")
    args = ap.parse_args()

    settings = Settings(**({"mode": args.mode} if args.mode else {}))
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    models = build_models(settings)
    print("Models:", {k: (v["name"], "ready" if v["ready"] else "NOT READY") for k, v in models.status.items()})
    if not models.status["generator"]["ready"]:
        sys.exit(f"Generator not ready: {models.status['generator']['detail']}")

    def progress(done, total, row):
        flag = "ERR" if row["error"] else ("abst" if row["abstained"] else ("ok " if row["correct"] else "x  "))
        print(f"[{done:>4}/{total}] {row['method']:<8} {flag} {row['question_id']}  {row['latency_ms']:.0f} ms")

    results = run_evaluation(settings, models, args.dataset, methods, args.limit, args.seed, progress, repeat=args.repeat)
    out = save_results(results, settings.results_dir)
    if results["dev_mode_warning"]:
        print("\n!!", results["dev_mode_warning"])
    print(f"\nDataset: {results['dataset']['title']}  |  questions: {results['sample']['num_questions']}  |  "
          f"runtime {results['runtime_s']} s")
    print(f"{'Metric':<36}" + "".join(f"{m[:18]:>20}" for m in methods))
    for key, label in COLUMNS:
        vals = [results["summary"][m][key] for m in methods]
        print(f"{label:<36}" + "".join(f"{'n/a' if v is None else v:>20}" for v in vals))
    if results.get("consistency"):
        print("Response consistency (repeat runs):", results["consistency"])
    print("* support score comes from this project's own verifier; see docs/evaluation_protocol.md")
    print(f"\nSaved: {out / 'results.json'} and results.csv")


if __name__ == "__main__":
    main()
