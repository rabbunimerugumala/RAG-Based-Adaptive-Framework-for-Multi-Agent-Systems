"""Benchmark runner shared by the CLI (evaluation/run_benchmark.py) and the API.

Both methods answer the SAME questions, in the same order, against the SAME benchmark knowledge base,
with the same retriever, generator and default top_k. Results are written only after a run finishes.
"""

from __future__ import annotations

import csv
import json
import random
import threading
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from adaptive_rag import __version__
from adaptive_rag.config import LEGACY_V1, Settings
from adaptive_rag.evaluation.datasets import load_dataset
from adaptive_rag.evaluation.metrics import aggregate, contains_answer, exact_match, token_f1
from adaptive_rag.models import ModelBundle
from adaptive_rag.pipeline import RAGSystem
from adaptive_rag.retrieval.store import KnowledgeBase
from adaptive_rag.text import content_terms, normalize_answer, strip_citations

CSV_FIELDS = ["question_id", "method", "question", "answerable", "type", "status", "abstained", "correct", "em",
              "f1", "cited_gold", "evidence_has_answer", "evidence_precision", "grounded", "support_level",
              "verification", "support_score", "confidence", "latency_ms", "llm_calls", "nli_calls",
              "retrieval_attempts", "passages_used", "passages_filtered", "answer", "gold_answers", "error"]

# Evaluated systems. Every variant shares the same knowledge base, retriever, embedder, generator, NLI model and
# questions; only the listed settings differ. "adaptive" is the proposed v2 multi-agent system.
VARIANTS: dict[str, tuple[str, dict]] = {
    "standard": ("standard", {}),
    "adaptive": ("adaptive", {}),
    "adaptive_no_retry": ("adaptive", {"max_retries": 0}),
    "adaptive_no_filter": ("adaptive", {"doc_filter": "off"}),
    "adaptive_v1": ("adaptive", dict(LEGACY_V1)),
    "adaptive_routed": ("adaptive", {"question_routing": True}),
}
VARIANT_LABELS = {
    "standard": "Standard RAG (baseline)",
    "adaptive": "Multi-Agent RAG v2 (proposed, full)",
    "adaptive_no_retry": "Multi-Agent v2 without adaptive retries",
    "adaptive_no_filter": "Multi-Agent v2 without evidence filtering",
    "adaptive_v1": "Multi-Agent v1 (original pipeline)",
    "adaptive_routed": "Multi-Agent v2 + question routing",
}


def grounded_answer(answer: str, passages: list[str], min_share: float = 0.8) -> bool:
    """Verifier-independent lexical groundedness: >= 80% of the answer's content terms occur in the passages."""
    terms = set(content_terms(strip_citations(answer)))
    if not terms:
        return False
    text_terms: set[str] = set()
    for p in passages:
        text_terms.update(content_terms(p))
    return len(terms & text_terms) / len(terms) >= min_share


def attempt_log(resp) -> list[dict]:
    """Per-attempt drafts and decisions, taken from the real execution trace (for later error analysis)."""
    log: dict[int, dict] = {}
    for t in resp.trace:
        a = log.setdefault(t.attempt, {"attempt": t.attempt})
        if t.step == "generator":
            a["draft"] = t.output.get("answer")
            a["insufficient"] = t.output.get("insufficient")
        elif t.step == "verifier":
            a["verification"] = t.decision
        elif t.step == "confidence" and "final" not in t.title:
            a["decision"], a["score"] = t.decision, t.output.get("score")
        elif t.step == "evidence":
            a["evidence_decision"] = t.decision
    return list(log.values())


class EvaluationCancelled(Exception):
    pass


def select_questions(questions: list[dict], limit: int | None, seed: int) -> list[dict]:
    """All questions, or a fixed-seed random sample of ``limit`` questions kept in file order."""
    if not limit or limit >= len(questions):
        return list(questions)
    idx = sorted(random.Random(seed).sample(range(len(questions)), limit))
    return [questions[i] for i in idx]


def open_benchmark_kb(settings: Settings, models: ModelBundle, dataset) -> KnowledgeBase:
    kb = KnowledgeBase(settings.runtime_dir / "benchmarks" / dataset.name, models.embedder,
                       settings.chunk_size, settings.chunk_overlap)
    kb.add_texts(dataset.corpus)
    return kb


def score_row(q: dict, method: str, resp) -> dict:
    golds = q.get("answers", [])
    abstained = resp.status != "answered"
    gold_docs = set(q.get("gold_doc_ids", []))
    cited_docs = {c.doc_id for c in resp.citations if c.valid}
    v = resp.verification
    cited_refs = {c.ref for c in resp.citations if c.valid}
    cited_text = [e.text for e in resp.evidence if e.ref in cited_refs] or [e.text for e in resp.evidence]
    return {
        "evidence_precision": (round(sum(contains_answer(e.text, golds) for e in resp.evidence) / len(resp.evidence), 4)
                               if q["answerable"] and golds and resp.evidence else None),
        "grounded": grounded_answer(resp.answer, cited_text) if not abstained else None,
        "support_level": resp.support_level,
        "route": resp.route,
        "passages_used": len(resp.evidence),
        "passages_filtered": len(resp.filtered_evidence),
        "attempts": attempt_log(resp),
        "question_id": q["id"], "method": method, "question": q["question"], "answerable": q["answerable"],
        "type": q.get("type", ""), "status": resp.status, "abstained": abstained,
        "correct": bool(q["answerable"] and not abstained and contains_answer(resp.answer, golds)),
        "em": exact_match(resp.answer, golds) if q["answerable"] and not abstained else 0.0,
        "f1": round(token_f1(resp.answer, golds), 4) if q["answerable"] and not abstained else 0.0,
        "cited_gold": (bool(cited_docs & gold_docs) if gold_docs and q["answerable"] and not abstained else None),
        "evidence_has_answer": (any(contains_answer(e.text, golds) for e in resp.evidence)
                                if q["answerable"] and golds else None),
        "verification": v.status if v else None,
        "support_score": v.support_score if v else None,
        "confidence": resp.confidence.score if resp.confidence else None,
        "latency_ms": resp.stats.latency_ms, "llm_calls": resp.stats.llm_calls, "nli_calls": resp.stats.nli_calls,
        "retrieval_attempts": resp.stats.retrieval_attempts, "answer": resp.answer, "gold_answers": golds,
        "run_id": resp.run_id, "error": None,
    }


def run_evaluation(
    settings: Settings, models: ModelBundle, dataset_name: str, methods: list[str], limit: int | None = None,
    seed: int = 0, progress: Callable[[int, int, dict], None] | None = None,
    cancel: threading.Event | None = None, run_id: str | None = None, repeat: int = 0,
) -> dict:
    unknown = [m for m in methods if m not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown method(s) {unknown}; choose from {list(VARIANTS)}")
    dataset = load_dataset(settings.data_dir / "datasets", dataset_name)
    questions = select_questions(dataset.questions, limit, seed)
    run_id = run_id or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    started = datetime.now(timezone.utc)
    commit_at_start = _git_commit()  # the code that actually runs (later commits do not affect a running process)
    t0 = time.perf_counter()
    kb = open_benchmark_kb(settings, models, dataset)
    try:
        systems = {m: (VARIANTS[m][0], RAGSystem(settings.model_copy(update=VARIANTS[m][1]), kb, models.generator,
                                                 models.nli, models.reranker)) for m in methods}
        rows: list[dict] = []
        total = len(questions) * len(methods)
        for q in questions:
            for method in methods:
                if cancel is not None and cancel.is_set():
                    raise EvaluationCancelled()
                try:
                    pipeline, system = systems[method]
                    resp = system.run(q["question"], pipeline)
                    row = score_row(q, method, resp)
                except Exception as exc:  # recorded as a failed case, never silently dropped
                    row = {"question_id": q["id"], "method": method, "question": q["question"],
                           "answerable": q["answerable"], "type": q.get("type", ""), "status": "error",
                           "abstained": True, "correct": False, "em": 0.0, "f1": 0.0, "cited_gold": None,
                           "evidence_has_answer": None, "verification": None, "support_score": None,
                           "confidence": None, "latency_ms": 0.0, "llm_calls": 0, "nli_calls": 0,
                           "retrieval_attempts": 0, "answer": "", "gold_answers": q.get("answers", []),
                           "evidence_precision": None, "grounded": None, "support_level": "none",
                           "passages_used": 0, "passages_filtered": 0, "attempts": [],
                           "run_id": None, "error": f"{exc.__class__.__name__}: {exc}"}
                rows.append(row)
                if progress:
                    progress(len(rows), total, row)
        # response consistency: re-run the first `repeat` questions and compare with the first run
        consistency: dict[str, dict] = {}
        for method in methods if repeat else []:
            pipeline, system = systems[method]
            first = {r["question_id"]: r for r in rows if r["method"] == method}
            same_answer = same_status = n = 0
            for q in questions[:repeat]:
                if cancel is not None and cancel.is_set():
                    raise EvaluationCancelled()
                try:
                    again = system.run(q["question"], pipeline)
                except Exception:
                    continue
                n += 1
                same_status += again.status == first[q["id"]]["status"]
                same_answer += normalize_answer(again.answer) == normalize_answer(first[q["id"]]["answer"])
            consistency[method] = {"n": n, "same_status": round(same_status / n, 4) if n else None,
                                   "same_answer": round(same_answer / n, 4) if n else None}
    finally:
        kb.close()
    return {
        "run_id": run_id,
        "dataset": {**dataset.manifest, "name": dataset.name, "num_questions_available": len(dataset.questions),
                    "num_documents": len(dataset.corpus)},
        "methods": methods,
        "sample": {"limit": limit, "seed": seed, "num_questions": len(questions),
                   "question_ids": [q["id"] for q in questions]},
        "mode": settings.mode,
        "dev_mode_warning": (
            "DEV MODE: hashing embedder + extractive generator. These numbers test the software, they are "
            "NOT real-model experimental results." if settings.mode == "dev" else None),
        "models": {k: v.get("name") for k, v in models.status.items()},
        "config": settings.public_dict(),
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_s": round(time.perf_counter() - t0, 2),
        "summary": {m: aggregate([r for r in rows if r["method"] == m]) for m in methods},
        "method_labels": {m: VARIANT_LABELS[m] for m in methods},
        "variant_settings": {m: VARIANTS[m][1] for m in methods},
        "consistency": consistency,
        "software": {"adaptive_rag": __version__, "git_commit": commit_at_start},
        "rows": rows,
    }


def _git_commit() -> str | None:
    try:
        import subprocess

        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5,
                             cwd=Path(__file__).resolve().parents[3])
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5,
                               cwd=Path(__file__).resolve().parents[3]).stdout.strip()
        return (out.stdout.strip() + ("-dirty" if dirty else "")) or None
    except Exception:
        return None


def save_results(results: dict, results_dir: Path) -> Path:
    out = results_dir / results["run_id"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in results["rows"]:
            w.writerow({**r, "gold_answers": " | ".join(r["gold_answers"])})
    return out
