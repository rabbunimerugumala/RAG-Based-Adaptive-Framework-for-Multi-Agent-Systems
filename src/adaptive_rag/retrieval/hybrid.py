"""Hybrid retrieval shared by Standard RAG and the Evidence Agent.

For every search query: dense (Qdrant) ranking + BM25 ranking -> reciprocal-rank fusion over all lists
-> optional cross-encoder rerank -> top-k Evidence objects. Scores are *relevance* signals only.
"""

from __future__ import annotations

import math
import threading

from adaptive_rag.retrieval.fusion import reciprocal_rank_fusion
from adaptive_rag.retrieval.store import KnowledgeBase
from adaptive_rag.schemas import Evidence
from adaptive_rag.text import content_terms


class CrossEncoderReranker:
    """Optional reranker (fastembed ONNX cross-encoder, default Xenova/ms-marco-MiniLM-L-6-v2)."""

    def __init__(self, model_name: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model_name)
        self._lock = threading.Lock()
        self.name = model_name

    def score(self, query: str, passages: list[str]) -> list[float]:
        with self._lock:
            return [float(s) for s in self._model.rerank(query, passages)]


def keyword_overlap(keywords: list[str], text: str) -> float:
    if not keywords:
        return 0.0
    terms = set(content_terms(text))
    return sum(1 for k in keywords if k in terms) / len(keywords)


def hybrid_search(
    kb: KnowledgeBase,
    queries: list[str],
    keywords: list[str],
    top_k: int,
    candidate_k: int,
    rrf_k: int,
    reranker: CrossEncoderReranker | None = None,
    exclude_chunk_ids: set[str] | None = None,
) -> list[Evidence]:
    ranked_lists: list[list[str]] = []
    dense_scores: dict[str, float] = {}
    bm25_scores: dict[str, float] = {}
    matched: dict[str, list[str]] = {}
    for q in queries:
        dense = kb.dense_search(q, candidate_k)
        sparse = kb.bm25_search(q, candidate_k)
        for lst, store in ((dense, dense_scores), (sparse, bm25_scores)):
            ranked_lists.append([cid for cid, _ in lst])
            for cid, s in lst:
                store[cid] = max(store.get(cid, float("-inf")), s)
                matched.setdefault(cid, [])
                if q not in matched[cid]:
                    matched[cid].append(q)
    fused = reciprocal_rank_fusion(ranked_lists, k=rrf_k)
    if exclude_chunk_ids:
        fused = [(cid, s) for cid, s in fused if cid not in exclude_chunk_ids]
    candidates = fused[: max(top_k * 3, top_k)] if reranker else fused[:top_k]

    rerank_scores: dict[str, float] = {}
    if reranker and candidates:
        texts = [kb.get_chunk(cid).text for cid, _ in candidates]
        for (cid, _), s in zip(candidates, reranker.score(queries[0], texts)):
            rerank_scores[cid] = s
        candidates = sorted(candidates, key=lambda x: rerank_scores[x[0]], reverse=True)[:top_k]

    evidence: list[Evidence] = []
    for i, (cid, rrf) in enumerate(candidates, start=1):
        c = kb.get_chunk(cid)
        evidence.append(
            Evidence(
                ref=i, chunk_id=cid, doc_id=c.doc_id, filename=c.filename, page=c.page,
                chunk_index=c.chunk_index, text=c.text, rrf_score=round(rrf, 6),
                dense_score=round(dense_scores[cid], 4) if cid in dense_scores else None,
                bm25_score=round(bm25_scores[cid], 4) if cid in bm25_scores else None,
                rerank_score=round(rerank_scores[cid], 4) if cid in rerank_scores else None,
                keyword_overlap=round(keyword_overlap(keywords, c.text), 4),
                matched_queries=matched.get(cid, []),
            )
        )
    return evidence


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))
