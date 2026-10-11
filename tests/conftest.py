"""Shared fixtures. All tests in this folder use MOCKED / dev-mode models unless marked ``real_model``.

* HashEmbedder        - deterministic dev-mode embedder (no download)
* ScriptedLLM         - mock of the shared LLM: returns pre-programmed answers, counts calls
* FakeNLI             - mock NLI: 'entails' when every hypothesis term occurs in the premise
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from adaptive_rag.config import LEGACY_V1, Settings
from adaptive_rag.generation.llm import INSUFFICIENT_ANSWER
from adaptive_rag.models import ModelBundle
from adaptive_rag.pipeline import RAGSystem
from adaptive_rag.retrieval.embeddings import HashEmbedder
from adaptive_rag.retrieval.store import KnowledgeBase
from adaptive_rag.schemas import Evidence, GeneratedAnswer
from adaptive_rag.text import content_terms, keywords

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "data" / "datasets" / "smoke_test"


class ScriptedLLM:
    """Mock LLM. ``script`` is a list of answers (consumed in order) or a callable(question, evidence)."""

    is_llm = True
    name = "mock-llm"

    def __init__(self, script: list | Callable[[str, list[Evidence]], GeneratedAnswer], judge_fn=None):
        self.script = script
        self.judge_fn = judge_fn
        self.answer_calls = 0
        self.rewrite_calls = 0
        self.judge_calls = 0
        self.predict_calls = 0

    def answer(self, question: str, evidence: list[Evidence]) -> GeneratedAnswer:
        self.answer_calls += 1
        if callable(self.script):
            return self.script(question, evidence)
        item = self.script[min(self.answer_calls - 1, len(self.script) - 1)]
        if item is None:
            return GeneratedAnswer(answer=INSUFFICIENT_ANSWER, cited_refs=[], insufficient=True)
        text, refs = item
        return GeneratedAnswer(answer=text, cited_refs=refs, insufficient=False)

    def rewrite_queries(self, question: str, missing: str) -> list[str]:
        self.rewrite_calls += 1
        return ["kestrel energy density"]

    def judge(self, question: str, passage: str, answer: str | None = None) -> tuple[float, dict]:
        """Mock evidence judge: +10 if most question keywords (and the answer, if given) are in the passage."""
        self.judge_calls += 1
        if self.judge_fn is not None:
            return self.judge_fn(question, passage, answer), {"mock": True}
        terms = set(content_terms(passage))
        q = keywords(question)
        ok = sum(t in terms for t in q) / max(len(q), 1) >= 0.5
        if answer is not None:
            ok = ok and set(content_terms(answer)) <= terms
        return (10.0 if ok else -10.0), {"mock": True}

    def predict(self, question: str, passage: str) -> str:
        self.predict_calls += 1
        return passage.split(".")[0]


class FakeNLI:
    name = "fake-nli"

    def predict(self, pairs):
        out = []
        for premise, hyp in pairs:
            h, p = set(content_terms(hyp)), set(content_terms(premise))
            full = bool(h) and h <= p
            negated = " never " in f" {hyp.lower()} " or " not " in f" {hyp.lower()} "
            out.append({"entailment": 0.95 if full and not negated else 0.05,
                        "contradiction": 0.9 if negated and len(h & p) >= len(h) / 2 else 0.02})
        return out


def load_smoke_corpus() -> list[dict]:
    return [json.loads(line) for line in (SMOKE / "corpus.jsonl").read_text(encoding="utf-8").splitlines() if line]


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(mode="dev", runtime_dir=tmp_path / "runtime", results_dir=tmp_path / "results",
                    data_dir=ROOT / "data", _env_file=None)


@pytest.fixture
def v1_settings(settings) -> Settings:
    """The original (v1) Adaptive RAG configuration, kept reproducible for baseline comparisons."""
    return settings.model_copy(update=LEGACY_V1)


@pytest.fixture
def kb(settings):
    k = KnowledgeBase(settings.runtime_dir / "kb", HashEmbedder(), settings.chunk_size, settings.chunk_overlap)
    yield k
    k.close()


@pytest.fixture
def smoke_kb(kb):
    for d in load_smoke_corpus():
        kb.add_text(d["doc_id"], d["title"], d["text"])
    return kb


def make_system(settings, kb, generator, nli=None) -> RAGSystem:
    return RAGSystem(settings, kb, generator, nli, None)


def dev_bundle(generator=None, nli=None) -> ModelBundle:
    from adaptive_rag.generation.llm import ExtractiveGenerator

    gen = generator or ExtractiveGenerator()
    return ModelBundle(HashEmbedder(), gen, nli, None, status={
        "embedder": {"name": "hash-dev-512", "ready": True, "detail": "test"},
        "generator": {"name": gen.name, "ready": True, "detail": "test"},
        "nli": {"name": "none", "ready": False, "detail": "test"},
        "reranker": {"name": "none", "ready": False, "detail": "test"},
    })


def make_pdf(pages: list[str]) -> bytes:
    """Build a minimal valid PDF with one text line per page (Helvetica), for extraction tests."""
    objs: list[bytes] = []
    n = len(pages)
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    font_id = 3 + 2 * n
    for i, text in enumerate(pages):
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET".encode()
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {4 + 2 * i} 0 R "
                    f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode())
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)
