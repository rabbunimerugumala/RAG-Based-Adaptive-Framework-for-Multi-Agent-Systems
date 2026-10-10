"""Evaluation metrics. Definitions are documented in docs/evaluation_protocol.md.

Answer matching uses SQuAD/TriviaQA normalisation (lower-case, strip punctuation and articles).
"""

from __future__ import annotations

import re
import statistics
from collections import Counter

from adaptive_rag.text import normalize_answer


def exact_match(prediction: str, golds: list[str]) -> float:
    p = normalize_answer(prediction)
    return float(any(p == normalize_answer(g) for g in golds))


def token_f1(prediction: str, golds: list[str]) -> float:
    p_toks = normalize_answer(prediction).split()
    best = 0.0
    for g in golds:
        g_toks = normalize_answer(g).split()
        common = Counter(p_toks) & Counter(g_toks)
        same = sum(common.values())
        if not p_toks or not g_toks or same == 0:
            continue
        prec, rec = same / len(p_toks), same / len(g_toks)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def contains_answer(text: str, golds: list[str]) -> bool:
    """True if any normalised gold alias occurs in the normalised text as a whole-word sequence."""
    t = f" {normalize_answer(text)} "
    for g in golds:
        g_n = normalize_answer(g)
        if g_n and re.search(rf"(?<!\w){re.escape(g_n)}(?!\w)", t):
            return True
    return False


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def _rate(num: int, den: int):
    return round(num / den, 4) if den else None


def aggregate(rows: list[dict]) -> dict:
    """Aggregate per-question rows of ONE method into summary metrics."""
    ok = [r for r in rows if not r.get("error")]
    ans = [r for r in ok if r["answerable"]]
    unans = [r for r in ok if not r["answerable"]]
    answered_ans = [r for r in ans if not r["abstained"]]
    cited = [r for r in answered_ans if r.get("cited_gold") is not None]
    with_ev = [r for r in ans if r.get("evidence_has_answer") is not None]
    lat = [r["latency_ms"] for r in ok]
    return {
        "n_total": len(rows),
        "n_errors": len(rows) - len(ok),
        "n_answerable": len(ans),
        "n_unanswerable": len(unans),
        "accuracy": _rate(sum(r["correct"] for r in ans), len(ans)),
        "exact_match": _mean([r["em"] for r in ans]),
        "f1": _mean([r["f1"] for r in ans]),
        "answer_rate_answerable": _rate(len(answered_ans), len(ans)),
        "selective_accuracy": _rate(sum(r["correct"] for r in answered_ans), len(answered_ans)),
        "correct_abstention_rate": _rate(sum(r["abstained"] for r in unans), len(unans)),
        "wrong_abstention_rate": _rate(sum(r["abstained"] for r in ans), len(ans)),
        "citation_gold_rate": _rate(sum(bool(r["cited_gold"]) for r in cited), len(cited)),
        "evidence_recall": _rate(sum(bool(r["evidence_has_answer"]) for r in with_ev), len(with_ev)),
        "mean_support_score": _mean([r.get("support_score") for r in ok if not r["abstained"]]),
        "mean_latency_ms": _mean(lat),
        "median_latency_ms": round(statistics.median(lat), 2) if lat else None,
        "mean_llm_calls": _mean([r["llm_calls"] for r in ok]),
        "mean_retrieval_attempts": _mean([r["retrieval_attempts"] for r in ok]),
    }
