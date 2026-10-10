"""BM25, reciprocal-rank fusion, the knowledge base and hybrid retrieval (dev-mode hash embedder)."""

from adaptive_rag.retrieval.bm25 import BM25Index
from adaptive_rag.retrieval.embeddings import HashEmbedder
from adaptive_rag.retrieval.fusion import reciprocal_rank_fusion
from adaptive_rag.retrieval.hybrid import hybrid_search, keyword_overlap
from adaptive_rag.retrieval.store import KnowledgeBase
from tests.conftest import make_pdf


def test_bm25_ranks_matching_document_first_and_ignores_non_matches():
    idx = BM25Index()
    idx.build({"a": "cats purr and sleep", "b": "dogs bark loudly", "c": "cats and dogs together"})
    res = idx.search("cats purr", 10)
    assert res[0][0] == "a"
    assert "b" not in {d for d, _ in res}


def test_bm25_idf_positive_on_tiny_corpus():
    idx = BM25Index()
    idx.build({"only": "single glacier corpus"})
    assert idx.search("glacier", 5)[0][1] > 0


def test_rrf_combines_rankings():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert {fused[0][0], fused[1][0]} == {"a", "b"}
    assert fused[0][1] == fused[1][1] == 1 / 61 + 1 / 62
    assert fused[-1][0] in {"c", "d"}


def test_rrf_rewards_agreement():
    fused = dict(reciprocal_rank_fusion([["x", "y"], ["y", "z"]], k=1))
    assert fused["y"] > fused["x"] and fused["y"] > fused["z"]


def test_hash_embedder_is_deterministic_and_normalised():
    e = HashEmbedder()
    v1, v2 = e.embed_query("energy density"), e.embed_query("energy density")
    assert (v1 == v2).all() and abs(float((v1 ** 2).sum()) - 1) < 1e-5


def test_upload_index_and_retrieve_with_citation_metadata(kb):
    res = kb.add_bytes("notes.txt", b"The Kestrel-9 battery has an energy density of 165 Wh/kg.")
    assert res.status == "indexed" and res.document.num_chunks == 1
    ev = hybrid_search(kb, ["Kestrel-9 energy density"], ["kestrel", "energy", "density"], 3, 10, 60)
    top = ev[0]
    assert top.ref == 1 and top.filename == "notes.txt" and top.doc_id == res.document.doc_id
    assert top.chunk_id == f"{top.doc_id}:0" and top.page is None
    assert top.dense_score is not None and top.bm25_score is not None and top.keyword_overlap == 1.0


def test_pdf_pages_are_kept_in_evidence(kb):
    kb.add_bytes("report.pdf", make_pdf(["Intro text about rivers.", "The dam is 61 metres high."]))
    ev = hybrid_search(kb, ["dam height metres"], ["dam", "metr"], 2, 10, 60)
    assert ev[0].page == 2 and ev[0].filename == "report.pdf"


def test_duplicate_upload_detected(kb):
    kb.add_bytes("a.txt", b"identical content here")
    dup = kb.add_bytes("b.txt", b"identical content here")
    assert dup.status == "duplicate" and kb.num_documents == 1


def test_failed_uploads_return_reasons(kb):
    assert kb.add_bytes("x.exe", b"MZ").status == "failed"
    bad = kb.add_bytes("empty.txt", b"")
    assert bad.status == "failed" and "empty" in bad.message.lower()
    assert kb.num_documents == 0


def test_delete_removes_chunks_and_file(kb):
    doc = kb.add_bytes("a.txt", b"Otters hold hands while sleeping.").document
    kb.add_bytes("b.txt", b"Beavers build dams.")
    assert kb.delete_document(doc.doc_id) is True
    assert kb.delete_document(doc.doc_id) is False
    ev = hybrid_search(kb, ["otters sleeping"], ["otter"], 5, 10, 60)
    assert all(e.doc_id != doc.doc_id for e in ev)
    assert not list((kb.root / "files").glob(f"{doc.doc_id}_*"))


def test_knowledge_base_persists_across_reopen(settings):
    root = settings.runtime_dir / "kb"
    kb1 = KnowledgeBase(root, HashEmbedder(), 900, 150)
    kb1.add_bytes("a.txt", b"Persistent fact about glaciers.")
    kb1.close()
    kb2 = KnowledgeBase(root, HashEmbedder(), 900, 150)
    try:
        assert kb2.num_documents == 1 and kb2.num_chunks == 1
        assert kb2.bm25_search("glaciers", 3)
    finally:
        kb2.close()


def test_changing_embedder_reindexes_from_stored_files(settings):
    root = settings.runtime_dir / "kb"
    kb1 = KnowledgeBase(root, HashEmbedder(512), 900, 150)
    kb1.add_bytes("a.txt", b"Glaciers move slowly downhill.")
    kb1.close()
    kb2 = KnowledgeBase(root, HashEmbedder(256), 900, 150)
    try:
        assert kb2.reindexed_on_startup and kb2.num_documents == 1
        assert kb2.dense_search("glaciers", 1)
    finally:
        kb2.close()


def test_exclusion_and_keyword_overlap(smoke_kb):
    first = hybrid_search(smoke_kb, ["Quillon port"], ["quillon", "port"], 2, 10, 60)
    again = hybrid_search(smoke_kb, ["Quillon port"], ["quillon", "port"], 2, 10, 60,
                          exclude_chunk_ids={first[0].chunk_id})
    assert first[0].chunk_id not in {e.chunk_id for e in again}
    assert keyword_overlap(["quillon", "port"], "Quillon listens on port 7425") == 1.0
    assert keyword_overlap([], "anything") == 0.0
