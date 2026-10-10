"""Agent 2 - Evidence. Executes the planner's search queries with hybrid retrieval (dense + BM25 + RRF,
optional cross-encoder rerank) and returns passages with full source metadata.

Retrieval scores say how *related* a passage is to the query; they are never treated as proof that a
claim is true - that is the Verifier Agent's job.
"""

from __future__ import annotations

from adaptive_rag.config import Settings
from adaptive_rag.retrieval.hybrid import CrossEncoderReranker, hybrid_search
from adaptive_rag.retrieval.store import KnowledgeBase
from adaptive_rag.schemas import Evidence, RetrievalPlan


class EvidenceAgent:
    def __init__(self, settings: Settings, kb: KnowledgeBase, reranker: CrossEncoderReranker | None):
        self.settings, self.kb, self.reranker = settings, kb, reranker

    def retrieve(self, plan: RetrievalPlan) -> tuple[list[Evidence], dict]:
        evidence = hybrid_search(
            self.kb, plan.search_queries, plan.keywords, plan.top_k, plan.candidate_k,
            self.settings.rrf_k, self.reranker,
        )
        summary = {
            "num_queries": len(plan.search_queries),
            "ranked_lists_fused": 2 * len(plan.search_queries),
            "reranker": self.reranker.name if self.reranker else None,
            "returned": len(evidence),
            "documents": sorted({e.filename for e in evidence}),
            "passages": [
                {"ref": e.ref, "chunk_id": e.chunk_id, "filename": e.filename, "page": e.page,
                 "rrf": e.rrf_score, "dense": e.dense_score, "bm25": e.bm25_score,
                 "rerank": e.rerank_score, "keyword_overlap": e.keyword_overlap}
                for e in evidence
            ],
        }
        return evidence, summary
