"""Reciprocal-rank fusion (Cormack et al., 2009): RRF(d) = sum over ranked lists of 1 / (k + rank(d))."""

from __future__ import annotations


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in ranked_lists:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))
