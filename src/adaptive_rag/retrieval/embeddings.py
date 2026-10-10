"""Embedding backends.

* HashEmbedder      - DEV MODE ONLY. Signed feature hashing of stemmed terms and term bigrams. It is
                      lexical, not semantic; it exists so the full pipeline runs without downloads.
* FastEmbedEmbedder - Real mode. ONNX Runtime on CPU (no GPU memory used), default BAAI/bge-small-en-v1.5
                      (MIT licence, 384-d, ~67 MB download).
"""

from __future__ import annotations

import threading
import zlib
from typing import Protocol

import numpy as np

from adaptive_rag.text import content_terms


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class HashEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hash-dev-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        terms = content_terms(text)
        feats = terms + [f"{a}_{b}" for a, b in zip(terms, terms[1:])]
        for f in feats:
            h = zlib.crc32(f.encode("utf-8"))
            v[h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class FastEmbedEmbedder:
    def __init__(self, model_name: str, cache_dir: str | None = None):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self._lock = threading.Lock()
        self.name = model_name
        self.dim = int(self.embed_query("dimension probe").shape[0])

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        with self._lock:
            return np.asarray(list(self._model.embed(texts, batch_size=32)), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        with self._lock:
            return np.asarray(next(iter(self._model.query_embed(text))), dtype=np.float32)
