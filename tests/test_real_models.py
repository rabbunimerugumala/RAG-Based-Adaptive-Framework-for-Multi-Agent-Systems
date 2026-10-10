"""REAL-MODEL integration tests (opt-in). They need downloaded models and a running Ollama server:

    $env:ARAG_RUN_REAL_MODEL_TESTS = "1"; pytest -m real_model

Without the environment variable they are reported as skipped (never silently passed).
"""

import os

import pytest

from adaptive_rag.config import Settings

pytestmark = [
    pytest.mark.real_model,
    pytest.mark.skipif(os.getenv("ARAG_RUN_REAL_MODEL_TESTS") != "1",
                       reason="real-model tests are opt-in: set ARAG_RUN_REAL_MODEL_TESTS=1"),
]


@pytest.fixture(scope="module")
def real_settings():
    return Settings(mode="real")


def test_fastembed_embeddings_are_semantic(real_settings):
    from adaptive_rag.retrieval.embeddings import FastEmbedEmbedder

    e = FastEmbedEmbedder(real_settings.embedding_model, cache_dir=str(real_settings.runtime_dir / "model_cache"))
    q = e.embed_query("How tall is the dam?")
    docs = e.embed_documents(["The dam is 61 metres high.", "Bakers sell pear bread at the fair."])
    assert e.dim == 384 and float(docs[0] @ q) > float(docs[1] @ q)


def test_nli_model_entails_and_contradicts(real_settings):
    from adaptive_rag.generation.nli import NLIModel

    nli = NLIModel(real_settings.nli_model)
    p = "Swimming in the reservoir is prohibited, but fishing is allowed with a permit."
    ent, con = nli.predict([(p, "Fishing is allowed with a permit."), (p, "Swimming in the reservoir is allowed.")])
    assert ent["entailment"] > 0.5 and con["contradiction"] > 0.5


def test_ollama_generates_grounded_json_answer(real_settings):
    from adaptive_rag.generation.llm import OllamaLLM
    from adaptive_rag.schemas import Evidence

    s = real_settings
    llm = OllamaLLM(s.ollama_url, s.ollama_model, s.ollama_num_ctx, s.llm_max_tokens, s.llm_timeout_s)
    if not llm.status()["ready"]:
        pytest.fail(llm.status()["detail"])
    ev = [Evidence(ref=1, chunk_id="d:0", doc_id="d", filename="d.txt", chunk_index=0, rrf_score=0.1,
                   text="Quillon servers listen on TCP port 7425 by default.")]
    a = llm.answer("Which port do Quillon servers use?", ev)
    assert not a.insufficient and "7425" in a.answer and 1 in a.cited_refs
    none = llm.answer("What is the capital of the Moon colony Selene?", ev)
    assert none.insufficient


def test_full_real_pipeline_smoke(real_settings, tmp_path):
    import json

    from adaptive_rag.models import build_models
    from adaptive_rag.pipeline import RAGSystem
    from adaptive_rag.retrieval.store import KnowledgeBase
    from tests.conftest import SMOKE

    models = build_models(real_settings)
    kb = KnowledgeBase(tmp_path / "kb", models.embedder, real_settings.chunk_size, real_settings.chunk_overlap)
    try:
        for line in (SMOKE / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            kb.add_text(d["doc_id"], d["title"], d["text"])
        r = RAGSystem(real_settings, kb, models.generator, models.nli, models.reranker).run_adaptive(
            "Who created the Quillon messaging protocol?")
        assert r.status == "answered" and "Ostberg" in r.answer and r.mode == "real"
    finally:
        kb.close()
