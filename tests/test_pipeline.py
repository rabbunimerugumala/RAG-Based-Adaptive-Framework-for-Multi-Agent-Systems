"""Standard RAG and Adaptive RAG end-to-end with a mocked LLM (and dev extractive generator)."""

from adaptive_rag.generation.llm import ExtractiveGenerator
from adaptive_rag.pipeline import EMPTY_KB_MESSAGE
from tests.conftest import FakeNLI, ScriptedLLM, make_system


def steps(resp):
    return [(t.step, t.attempt) for t in resp.trace]


def test_empty_knowledge_base_gives_instructions_not_an_answer(settings, kb):
    llm = ScriptedLLM([("anything", [1])])
    for method in ("standard", "adaptive"):
        r = make_system(settings, kb, llm).run("Who founded Varnholm?", method)
        assert r.status == "no_documents" and r.answer == EMPTY_KB_MESSAGE and r.citations == []
    assert llm.answer_calls == 0


def test_standard_rag_answers_with_validated_citations(settings, smoke_kb):
    llm = ScriptedLLM([("Varnholm was founded in 1847 by the miller Edda Larsk [1][9].", [1, 9])])
    r = make_system(settings, smoke_kb, llm).run_standard("Who founded Varnholm?")
    assert r.status == "answered" and r.stats.llm_calls == 1 and r.stats.retrieval_attempts == 1
    assert [c.ref for c in r.citations] == [1] and r.citations[0].filename == "Varnholm Town History"
    assert r.evidence[0].doc_id == "varnholm_history"
    check = next(t for t in r.trace if t.step == "citation_check")
    assert check.output["removed_invalid"] == [9]
    assert r.verification is not None and r.verification_used_in_decision is False
    assert [t.step for t in r.trace] == ["retrieval", "generator", "citation_check", "verifier"]


def test_adaptive_accepts_supported_answer_in_one_attempt(settings, smoke_kb):
    llm = ScriptedLLM([("Varnholm was founded in 1847 by the miller Edda Larsk [1].", [1])])
    r = make_system(settings, smoke_kb, llm, FakeNLI()).run_adaptive("Who founded Varnholm?")
    assert r.status == "answered" and r.confidence.decision == "accept"
    assert steps(r) == [("planner", 0), ("evidence", 0), ("generator", 0), ("verifier", 0), ("confidence", 0)]
    assert r.verification.status == "supported" and r.verification_used_in_decision
    assert r.stats.llm_calls == 1 and r.stats.retries == 0 and r.stats.nli_calls >= 1


def test_adaptive_retries_after_unsupported_draft_then_accepts(settings, smoke_kb):
    llm = ScriptedLLM([
        ("The Kestrel-9 weighs 900 kilograms and glows purple [1].", [1]),
        ("The Kestrel-9 has an energy density of 165 Wh/kg [1].", [1]),
    ])
    r = make_system(settings, smoke_kb, llm, FakeNLI()).run_adaptive("What is the energy density of the Kestrel-9?")
    assert r.status == "answered" and "165" in r.answer
    assert r.stats.retrieval_attempts == 2 and r.stats.retries == 1
    assert ("planner", 1) in steps(r) and ("confidence", 1) in steps(r)
    first_conf = next(t for t in r.trace if t.step == "confidence" and t.attempt == 0)
    assert first_conf.decision == "retry"
    # 2 answer calls + 1 query-rewrite call by the planner on retry
    assert r.stats.llm_calls == 3 and llm.rewrite_calls == 1


def test_adaptive_abstains_after_max_retries(settings, smoke_kb):
    llm = ScriptedLLM([("The Kestrel-9 costs 4,999 dollars at retail [1].", [1])])
    r = make_system(settings, smoke_kb, llm, FakeNLI()).run_adaptive(
        "What is the retail price of a Kestrel-9 battery module?")
    assert r.status == "abstained" and r.citations == []
    assert r.stats.retrieval_attempts == settings.max_retries + 1 == 3
    assert r.confidence.decision == "abstain"
    assert sum(1 for t in r.trace if t.step == "evidence") == 3


def test_retry_budget_is_configurable(settings, smoke_kb):
    s = settings.model_copy(update={"max_retries": 0})
    llm = ScriptedLLM([("The Kestrel-9 costs 4,999 dollars [1].", [1])])
    r = make_system(s, smoke_kb, llm, FakeNLI()).run_adaptive("What is the retail price of a Kestrel-9?")
    assert r.stats.retrieval_attempts == 1 and r.status == "abstained"


def test_adaptive_abstains_when_generator_declines_every_time(settings, smoke_kb):
    llm = ScriptedLLM([None])
    r = make_system(settings, smoke_kb, llm).run_adaptive("Which encryption algorithm does Quillon use?")
    assert r.status == "abstained" and llm.answer_calls == 3
    assert r.verification.status == "not_applicable"


def test_small_talk_skips_retrieval(settings, smoke_kb):
    llm = ScriptedLLM([("x", [1])])
    r = make_system(settings, smoke_kb, llm).run_adaptive("hello")
    assert r.status == "no_retrieval_needed" and llm.answer_calls == 0 and steps(r) == [("planner", 0)]


def test_generator_failure_is_recorded_as_failed_step(settings, smoke_kb):
    from adaptive_rag.generation.llm import LLMError

    def boom(q, e):
        raise LLMError("connection refused")

    r = make_system(settings, smoke_kb, ScriptedLLM(boom)).run_standard("Who founded Varnholm?")
    failed = [t for t in r.trace if t.status == "failed"]
    assert failed and failed[0].step == "generator" and "connection refused" in failed[0].note
    assert r.status == "abstained"


def test_dev_mode_extractive_pipeline(settings, smoke_kb):
    sys_ = make_system(settings, smoke_kb, ExtractiveGenerator())
    r = sys_.run_adaptive("Which TCP port do Quillon servers listen on by default?")
    assert r.status == "answered" and "7425" in r.answer and r.stats.llm_calls == 0
    assert r.mode == "dev" and r.models["generator"] == "extractive-dev"


def test_both_methods_use_same_evidence_on_first_attempt(settings, smoke_kb):
    sys_ = make_system(settings, smoke_kb, ExtractiveGenerator())
    q = "Who is the chief executive officer of Orvane Labs?"
    std, ada = sys_.run_standard(q), sys_.run_adaptive(q)
    assert std.evidence[0].chunk_id == ada.evidence[0].chunk_id
