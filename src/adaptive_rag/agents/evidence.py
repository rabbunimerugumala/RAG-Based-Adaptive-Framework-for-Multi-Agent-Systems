"""Agent 2 - Evidence.

Stage 1 - retrieval: runs the planner's search queries with hybrid retrieval (dense + BM25 + RRF, optional
          cross-encoder rerank).
Stage 2 - (doc_filter="judge") MAIN-RAG-inspired evidence filtering (Chang et al., ACL 2025, Sec. 3.2-3.3):
   a. near-duplicate passages are removed (word-set Jaccard >= dedup_jaccard);
   b. every remaining candidate is judged by the shared LLM. The score is the paper's
      s_i = log P("Yes") - log P("No") of the first generated token. In "triplet" mode the paper's
      Predictor first answers from each passage and the Judge sees (Document, Question, LLM Answer) with the
      paper's Appendix C instruction; in "pair" mode (this project's cheaper adaptation) the Judge sees
      (Document, Question) only;
   c. adaptive judge bar tau_q = mean(s) - n * std(s) (paper default n = 0); passages with s_i >= tau_q are kept;
   d. kept passages are ordered by descending score (ties keep the retrieval order) and cut to the evidence
      budget (plan.top_k); at least doc_min_keep passages are always kept.

Judge scores measure whether a passage is relevant/supportive for the question. They are not proof that a
claim is true - that is the Verifier Agent's job.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from adaptive_rag.config import Settings
from adaptive_rag.generation.llm import Generator, LLMError
from adaptive_rag.retrieval.hybrid import CrossEncoderReranker, hybrid_search
from adaptive_rag.retrieval.store import KnowledgeBase
from adaptive_rag.schemas import Evidence, EvidenceFilterReport, RetrievalPlan
from adaptive_rag.text import content_terms


@dataclass
class EvidencePackage:
    evidence: list[Evidence]                      # passages given to the generator, refs renumbered 1..k
    filtered: list[Evidence] = field(default_factory=list)  # removed passages with filter_reason
    report: EvidenceFilterReport | None = None
    llm_calls: int = 0
    summary: dict = field(default_factory=dict)


def jaccard(a: str, b: str) -> float:
    sa, sb = set(content_terms(a)), set(content_terms(b))
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


def adaptive_threshold(scores: list[float], n_sigma: float) -> tuple[float, float, float]:
    """MAIN-RAG Sec. 3.3: tau_q = mean - n * std (population std). Returns (tau, mean, std)."""
    mean = statistics.fmean(scores)
    std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    return mean - n_sigma * std, mean, std


def select_by_threshold(scored: list[Evidence], n_sigma: float, budget: int, min_keep: int
                        ) -> tuple[list[Evidence], list[Evidence], float, float, float, str | None]:
    """Apply the adaptive judge bar. `scored` must carry judge_score and retrieval_rank."""
    scores = [e.judge_score for e in scored]
    tau, mean, std = adaptive_threshold(scores, n_sigma)
    ordered = sorted(scored, key=lambda e: (-e.judge_score, e.retrieval_rank))
    kept = [e for e in ordered if e.judge_score >= tau - 1e-9]
    fallback = None
    if len(kept) < min_keep:
        kept = ordered[:min_keep]
        fallback = f"fewer than {min_keep} passage(s) passed tau_q; kept the top {min_keep} by score"
    over_budget = kept[budget:]
    kept = kept[:budget]
    kept_ids = {e.chunk_id for e in kept}
    removed = []
    for e in ordered:
        if e.chunk_id in kept_ids:
            continue
        reason = (f"over evidence budget (top {budget} kept)" if e in over_budget
                  else f"judge score {e.judge_score:.2f} < tau_q {tau:.2f}")
        removed.append(e.model_copy(update={"filter_reason": reason, "ref": 0}))
    return kept, removed, tau, mean, std, fallback


class EvidenceAgent:
    def __init__(self, settings: Settings, kb: KnowledgeBase, reranker: CrossEncoderReranker | None,
                 generator: Generator | None = None):
        self.settings, self.kb, self.reranker, self.generator = settings, kb, reranker, generator

    def retrieve(self, plan: RetrievalPlan, judge_cache: dict | None = None) -> EvidencePackage:
        s = self.settings
        use_judge = s.doc_filter == "judge" and plan.pool_k > 0 and self.generator is not None
        k = plan.pool_k if use_judge else plan.top_k
        pool = hybrid_search(self.kb, plan.search_queries, plan.keywords, k, plan.candidate_k, s.rrf_k, self.reranker)
        pool = [e.model_copy(update={"retrieval_rank": i}) for i, e in enumerate(pool, start=1)]
        if not use_judge:
            return EvidencePackage(evidence=pool, summary=self._summary(plan, pool, []))

        # (a) near-duplicate removal (the higher-ranked copy is kept)
        unique: list[Evidence] = []
        dups: list[Evidence] = []
        for e in pool:
            twin = next((u for u in unique if jaccard(u.text, e.text) >= s.dedup_jaccard), None)
            if twin:
                dups.append(e.model_copy(update={"ref": 0, "filter_reason": f"near-duplicate of {twin.chunk_id}"}))
            else:
                unique.append(e)

        # (b) judge every candidate (cached per run so retries do not re-judge the same passage)
        cache = judge_cache if judge_cache is not None else {}
        calls = 0
        scored: list[Evidence] = []
        try:
            for e in unique:
                key = (e.chunk_id, s.doc_judge_mode)
                if key not in cache:
                    answer = None
                    if s.doc_judge_mode == "triplet":
                        answer = self.generator.predict(plan.query, e.text) or "No answer"
                        calls += int(self.generator.is_llm)
                    score, _ = self.generator.judge(plan.query, e.text, answer)
                    calls += int(self.generator.is_llm)
                    cache[key] = (score, answer)
                score, answer = cache[key]
                scored.append(e.model_copy(update={"judge_score": score, "candidate_answer": answer}))
        except LLMError as exc:
            # graceful fallback: keep plain retrieval order, record why
            evidence = [e.model_copy(update={"ref": i}) for i, e in enumerate(unique[: plan.top_k], start=1)]
            report = EvidenceFilterReport(mode=s.doc_judge_mode, pool_size=len(pool), duplicates_removed=len(dups),
                                          threshold=None, mean=None, std=None, n_sigma=s.doc_filter_n_sigma,
                                          retained=len(evidence), filtered=len(pool) - len(evidence),
                                          fallback=f"judge unavailable ({exc}); used retrieval order", judge_calls=calls)
            return EvidencePackage(evidence, dups, report, calls, self._summary(plan, evidence, dups, report))

        if not scored:  # nothing retrieved
            report = EvidenceFilterReport(mode=s.doc_judge_mode, pool_size=0, duplicates_removed=0, threshold=None,
                                          mean=None, std=None, n_sigma=s.doc_filter_n_sigma, retained=0, filtered=0,
                                          judge_calls=0)
            return EvidencePackage([], [], report, 0, self._summary(plan, [], [], report))

        # (c)+(d) adaptive judge bar, ordering, budget
        kept, removed, tau, mean, std, fallback = select_by_threshold(
            scored, s.doc_filter_n_sigma, plan.top_k, s.doc_min_keep)
        evidence = [e.model_copy(update={"ref": i}) for i, e in enumerate(kept, start=1)]
        filtered = dups + removed
        report = EvidenceFilterReport(mode=s.doc_judge_mode, pool_size=len(pool), duplicates_removed=len(dups),
                                      threshold=round(tau, 4), mean=round(mean, 4), std=round(std, 4),
                                      n_sigma=s.doc_filter_n_sigma, retained=len(evidence), filtered=len(filtered),
                                      fallback=fallback, judge_calls=calls)
        return EvidencePackage(evidence, filtered, report, calls, self._summary(plan, evidence, filtered, report))

    def _summary(self, plan: RetrievalPlan, evidence: list[Evidence], filtered: list[Evidence],
                 report: EvidenceFilterReport | None = None) -> dict:
        return {
            "num_queries": len(plan.search_queries),
            "ranked_lists_fused": 2 * len(plan.search_queries),
            "reranker": self.reranker.name if self.reranker else None,
            "returned": len(evidence),
            "documents": sorted({e.filename for e in evidence}),
            "filter": report.model_dump() if report else {"mode": "off"},
            "passages": [
                {"ref": e.ref, "chunk_id": e.chunk_id, "filename": e.filename, "page": e.page,
                 "retrieval_rank": e.retrieval_rank, "rrf": e.rrf_score, "dense": e.dense_score,
                 "bm25": e.bm25_score, "rerank": e.rerank_score, "judge_score": e.judge_score,
                 "keyword_overlap": e.keyword_overlap}
                for e in evidence
            ],
            "filtered_out": [
                {"chunk_id": e.chunk_id, "filename": e.filename, "retrieval_rank": e.retrieval_rank,
                 "judge_score": e.judge_score, "reason": e.filter_reason}
                for e in filtered
            ],
        }
