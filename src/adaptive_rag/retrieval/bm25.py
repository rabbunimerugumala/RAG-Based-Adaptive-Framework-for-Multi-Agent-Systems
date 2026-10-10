"""Okapi BM25 keyword index (k1=1.5, b=0.75) with the always-positive Lucene IDF.

idf(t)      = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))
score(q, d) = sum_t idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
"""

from __future__ import annotations

import math
from collections import Counter

from adaptive_rag.text import content_terms


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids: list[str] = []
        self._tfs: list[Counter] = []
        self._lens: list[int] = []
        self._idf: dict[str, float] = {}
        self._avgdl = 0.0

    def build(self, docs: dict[str, str]) -> None:
        self.ids = list(docs)
        self._tfs = [Counter(content_terms(docs[i])) for i in self.ids]
        self._lens = [sum(tf.values()) for tf in self._tfs]
        n = len(self.ids)
        self._avgdl = (sum(self._lens) / n) if n else 0.0
        df: Counter = Counter()
        for tf in self._tfs:
            df.update(tf.keys())
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        terms = set(content_terms(query))
        if not terms or not self.ids:
            return []
        scores: list[tuple[str, float]] = []
        for doc_id, tf, dl in zip(self.ids, self._tfs, self._lens):
            s = 0.0
            for t in terms:
                f = tf.get(t)
                if f:
                    s += self._idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self._avgdl))
            if s > 0:
                scores.append((doc_id, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
