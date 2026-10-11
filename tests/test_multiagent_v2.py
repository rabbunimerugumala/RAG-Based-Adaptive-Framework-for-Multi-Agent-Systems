"""Tests for the v2 multi-agent pipeline (mocked models): MAIN-RAG-inspired evidence filtering, grounded
verification of short answers, Confidence v2 attempt selection, failure handling and LLM-interface robustness."""

import httpx
import pytest

from adaptive_rag.agents.confidence import AttemptRecord, ConfidenceAgent
from adaptive_rag.agents.evidence import EvidenceAgent, adaptive_threshold, jaccard, select_by_threshold
from adaptive_rag.agents.planner import PlannerAgent
from adaptive_rag.agents.verifier import VerifierAgent, is_fallback_hypothesis
from adaptive_rag.generation.llm import ExtractiveGenerator, LLMError, OllamaLLM, yes_no_score
from adaptive_rag.schemas import (
    ConfidenceAssessment,
    Evidence,
    GeneratedAnswer,
    RetrievalPlan,
    VerificationResult,
)
from tests.conftest import FakeNLI, ScriptedLLM, make_system


def ev(ref, text, score=None, rank=None, doc="d", page=None):
    return Evidence(ref=ref, chunk_id=f"{doc}:{ref}", doc_id=doc, filename=f"{doc}.pdf", page=page, chunk_index=ref,
                    text=text, rrf_score=0.01, judge_score=score, retrieval_rank=rank or ref, keyword_overlap=1.0)


# ------------------------------------------------------------------ judge score (MAIN-RAG Sec. 3.2)
def test_yes_no_score_is_log_odds():
    score, d = yes_no_score([{"token": "Yes", "logprob": -0.1}, {"token": " no", "logprob": -2.5}])
    assert score == pytest.approx(2.4) and d["floor_used"] is False


def test_yes_no_score_floor_when_token_missing():
    score, d = yes_no_score([{"token": "Yes", "logprob": 0.0}, {"token": "The", "logprob": -9.0}])
    assert d["floor_used"] is True and score == pytest.approx(10.0)  # no = floor = -9 - 1


# ------------------------------------------------------------------ adaptive judge bar (Sec. 3.3)
def test_adaptive_threshold_mean_and_sigma():
    tau, mean, std = adaptive_threshold([1.0, 3.0, 5.0], 0.0)
    assert tau == mean == 3.0
    tau2, _, std = adaptive_threshold([1.0, 3.0, 5.0], 1.0)
    assert tau2 == pytest.approx(3.0 - std)


def test_filter_keeps_above_bar_orders_descending_and_preserves_metadata():
    scored = [ev(1, "a", -5.0, 1, page=3), ev(2, "b", 9.0, 2, page=7), ev(3, "c", 4.0, 3), ev(4, "d", -12.0, 4)]
    kept, removed, tau, *_ = select_by_threshold(scored, 0.0, budget=5, min_keep=1)
    assert tau == pytest.approx(-1.0)
    assert [e.chunk_id for e in kept] == ["d:2", "d:3"]
    assert kept[0].page == 7 and kept[0].filename == "d.pdf"
    assert {e.chunk_id for e in removed} == {"d:1", "d:4"}
    assert all("tau_q" in e.filter_reason for e in removed)


def test_filter_ties_keep_retrieval_order_and_equal_scores_keep_all():
    scored = [ev(1, "a", 2.0, 1), ev(2, "b", 2.0, 2), ev(3, "c", 2.0, 3)]
    kept, removed, *_ = select_by_threshold(scored, 0.0, budget=5, min_keep=1)
    assert [e.chunk_id for e in kept] == ["d:1", "d:2", "d:3"] and removed == []


def test_filter_never_removes_all_evidence():
    scored = [ev(1, "a", -3.0, 1), ev(2, "b", -1.0, 2)]
    kept, _, _, _, _, fallback = select_by_threshold(scored, 3.0, budget=5, min_keep=1)
    assert len(kept) >= 1
    kept, _, _, _, _, fallback = select_by_threshold(scored, 0.0, budget=5, min_keep=2)
    assert len(kept) == 2 and fallback


def test_filter_respects_evidence_budget():
    scored = [ev(i, str(i), 10.0 - i, i) for i in range(1, 7)]
    kept, removed, *_ = select_by_threshold(scored, 3.0, budget=2, min_keep=1)
    assert len(kept) == 2 and any("budget" in e.filter_reason for e in removed)


def test_jaccard_dedup_measure():
    assert jaccard("The dam is 61 metres high", "the dam is 61 metres high.") == 1.0
    assert jaccard("cats purr", "dogs bark") == 0.0


# ------------------------------------------------------------------ Evidence Agent end to end
def test_evidence_agent_filters_with_judge_and_renumbers(settings, smoke_kb):
    settings = settings.model_copy(update={"doc_filter_stage": "always"})
    llm = ScriptedLLM([])
    plan = PlannerAgent(settings, llm).plan("Who created the Quillon messaging protocol?")
    assert plan.pool_k == settings.doc_pool_k
    pkg = EvidenceAgent(settings, smoke_kb, None, llm).retrieve(plan)
    assert pkg.report.pool_size == 8 and pkg.report.judge_calls == llm.judge_calls == pkg.llm_calls
    assert pkg.evidence[0].doc_id == "quillon_protocol" and pkg.evidence[0].judge_score > 0
    assert [e.ref for e in pkg.evidence] == list(range(1, len(pkg.evidence) + 1))
    assert all(e.ref == 0 and e.filter_reason for e in pkg.filtered)
    assert len(pkg.evidence) + len(pkg.filtered) == pkg.report.pool_size


def test_evidence_agent_judge_cache_avoids_rejudging(settings, smoke_kb):
    settings = settings.model_copy(update={"doc_filter_stage": "always"})
    llm = ScriptedLLM([])
    agent = EvidenceAgent(settings, smoke_kb, None, llm)
    plan = PlannerAgent(settings, llm).plan("Who created the Quillon messaging protocol?")
    cache: dict = {}
    agent.retrieve(plan, cache)
    first = llm.judge_calls
    pkg = agent.retrieve(plan, cache)
    assert llm.judge_calls == first and pkg.llm_calls == 0


def test_evidence_agent_triplet_mode_uses_predictor(settings, smoke_kb):
    s = settings.model_copy(update={"doc_judge_mode": "triplet", "doc_pool_k": 3, "doc_filter_stage": "always"})
    llm = ScriptedLLM([])
    plan = PlannerAgent(s, llm).plan("Who created the Quillon messaging protocol?")
    pkg = EvidenceAgent(s, smoke_kb, None, llm).retrieve(plan)
    assert llm.predict_calls == llm.judge_calls > 0 and pkg.llm_calls == 2 * llm.judge_calls
    assert pkg.evidence[0].candidate_answer


def test_evidence_agent_falls_back_when_judge_fails(settings, smoke_kb):
    settings = settings.model_copy(update={"doc_filter_stage": "always"})
    def broken(q, p, a):
        raise LLMError("timeout")
    llm = ScriptedLLM([], judge_fn=broken)
    plan = PlannerAgent(settings, llm).plan("Who created the Quillon messaging protocol?")
    pkg = EvidenceAgent(settings, smoke_kb, None, llm).retrieve(plan)
    assert pkg.evidence and "judge unavailable" in pkg.report.fallback


def test_filter_off_matches_plain_retrieval(settings, smoke_kb):
    s = settings.model_copy(update={"doc_filter": "off"})
    plan = PlannerAgent(s, ExtractiveGenerator()).plan("Who created the Quillon messaging protocol?")
    assert plan.pool_k == 0
    pkg = EvidenceAgent(s, smoke_kb, None, ScriptedLLM([])).retrieve(plan)
    assert pkg.report is None and len(pkg.evidence) == plan.top_k and pkg.llm_calls == 0


# ------------------------------------------------------------------ grounded verifier
class QuestionStyleNLI:
    """Mimics the measured failure: an ill-formed 'question? answer.' hypothesis gets a high contradiction."""
    name = "question-style-nli"

    def predict(self, pairs):
        return [{"entailment": 0.1, "contradiction": 0.95 if "?" in h else 0.02} for _, h in pairs]


def test_fallback_hypothesis_detection():
    assert is_fallback_hypothesis("Of which country is Niamey the capital?", "Of which country is Niamey the capital? Niger.")
    assert not is_fallback_hypothesis("Who founded X?", "Edda founded X.")


def test_v1_verifier_rejects_correct_short_answer_reproducing_the_baseline_failure(settings):
    s = settings.model_copy(update={"verifier_mode": "v1"})
    evidence = [ev(1, "Niamey is the capital and largest city of Niger.", 10.0)]
    res, _ = VerifierAgent(s, QuestionStyleNLI()).verify(
        "Of which African country is Niamey the capital?", GeneratedAnswer(answer="Niger", cited_refs=[1], insufficient=False), evidence)
    assert res.status == "contradicted"


def test_grounded_verifier_accepts_correct_short_answer_with_judge(settings):
    llm = ScriptedLLM([])
    evidence = [ev(1, "Niamey is the capital and largest city of Niger.", 10.0)]
    res, _ = VerifierAgent(settings, QuestionStyleNLI(), llm).verify(
        "Of which African country is Niamey the capital?", GeneratedAnswer(answer="Niger", cited_refs=[1], insufficient=False), evidence)
    assert res.status == "supported" and res.llm_calls == 1 and "judge confirms" in res.claims[0].reason


def test_grounded_verifier_rejects_short_answer_not_in_evidence(settings):
    llm = ScriptedLLM([])
    evidence = [ev(1, "Niamey is the capital and largest city of Niger.", 10.0)]
    res, _ = VerifierAgent(settings, QuestionStyleNLI(), llm).verify(
        "Of which African country is Niamey the capital?", GeneratedAnswer(answer="Mali", cited_refs=[1], insufficient=False), evidence)
    assert res.status == "unsupported" and llm.judge_calls == 0  # no span -> judge is not even asked


def test_grounded_verifier_requires_judge_confirmation(settings):
    llm = ScriptedLLM([], judge_fn=lambda q, p, a: -8.0)
    evidence = [ev(1, "Niamey is the capital and largest city of Niger.", 10.0)]
    res, _ = VerifierAgent(settings, QuestionStyleNLI(), llm).verify(
        "Of which African country is Niamey the capital?", GeneratedAnswer(answer="Niger", cited_refs=[1], insufficient=False), evidence)
    assert res.status != "supported"


def test_grounded_verifier_still_detects_real_contradictions(settings):
    evidence = [ev(1, "Swimming in the reservoir is prohibited.", 5.0)]
    res, _ = VerifierAgent(settings, FakeNLI(), ScriptedLLM([])).verify(
        "q", GeneratedAnswer(answer="Swimming in the reservoir is not prohibited [1].", cited_refs=[1], insufficient=False), evidence)
    assert res.status == "contradicted"


def test_grounded_contradiction_ignores_uncited_irrelevant_passage(settings):
    class OffTopicContradicts:
        name = "off-topic"

        def predict(self, pairs):
            return [{"entailment": 0.9 if "Kestrel-9 stores energy" in p else 0.0,
                     "contradiction": 0.95 if "Kestrel-9 was cancelled" in p else 0.0} for p, _ in pairs]

    evidence = [ev(1, "The Kestrel-9 stores energy for the grid.", 9.0),
                ev(2, "Rumours said the Kestrel-9 was cancelled, which stores energy for the grid.", -9.0)]
    ans = GeneratedAnswer(answer="The Kestrel-9 stores energy for the grid [1].", cited_refs=[1], insufficient=False)
    assert VerifierAgent(settings, OffTopicContradicts(), ScriptedLLM([])).verify("q", ans, evidence)[0].status == "supported"
    v1 = settings.model_copy(update={"verifier_mode": "v1"})
    assert VerifierAgent(v1, OffTopicContradicts()).verify("q", ans, evidence)[0].status == "supported"


# ------------------------------------------------------------------ confidence v2
def _rec(attempt, insufficient, status, support, score):
    plan = RetrievalPlan(query="q", needs_retrieval=True, query_type="factoid", keywords=["k"], subquestions=[],
                         search_queries=["q"], top_k=5, candidate_k=20, strategy="initial", explanation="")
    conf = ConfidenceAssessment(score=score, components={}, weights={}, missing_components=[], decision="retry", reason="")
    ver = VerificationResult(status=status, support_score=support, method="t", summary="")
    return AttemptRecord(attempt, plan, [ev(1, "k")], GeneratedAnswer(answer="x", cited_refs=[], insufficient=insufficient), ver, conf)


def test_v2_selection_prefers_verified_answer_over_high_scoring_insufficient_attempt(settings):
    attempts = [_rec(0, True, "not_applicable", None, 0.71), _rec(1, False, "partially_supported", 0.5, 0.55)]
    best, final = ConfidenceAgent(settings).select_final(attempts)
    assert best.attempt == 1 and final.decision == "accept"
    v1 = settings.model_copy(update={"confidence_mode": "v1"})
    best1, final1 = ConfidenceAgent(v1).select_final(attempts)
    assert best1.attempt == 0 and final1.decision == "abstain"  # the original flaw, kept for reproducibility


def test_v2_relevance_uses_judge_scores(settings):
    plan = PlannerAgent(settings, ExtractiveGenerator()).plan("Who created Quillon?")
    comps = ConfidenceAgent(settings).components(plan, [ev(1, "Quillon x", 5.0), ev(2, "other", -3.0)],
                                                 VerificationResult(status="supported", support_score=1.0,
                                                                    citation_consistency=1.0, method="t", summary=""))
    assert comps["relevance"] == 0.5


# ------------------------------------------------------------------ v2 pipeline end to end
def test_v2_pipeline_accepts_correct_short_answer_without_escalation(settings, smoke_kb):
    llm = ScriptedLLM([("Mara Ostberg [1]", [1])])
    r = make_system(settings, smoke_kb, llm, QuestionStyleNLI()).run_adaptive("Who created the Quillon messaging protocol?")
    assert r.status == "answered" and r.support_level == "verified" and r.unresolved_claims == []
    assert r.stats.retrieval_attempts == 1 and r.evidence_filter is None  # round 1: plain evidence, no judging
    assert r.citations[0].doc_id == "quillon_protocol"
    assert r.stats.llm_calls == 1 + r.verification.llm_calls


def test_v2_pipeline_escalates_to_judge_filtering_on_retry(settings, smoke_kb):
    llm = ScriptedLLM([None, ("Quillon servers listen on TCP port 7425 by default [1].", [1])])
    r = make_system(settings, smoke_kb, llm, FakeNLI()).run_adaptive("Which TCP port do Quillon servers listen on by default?")
    assert r.status == "answered" and r.stats.retrieval_attempts == 2
    assert r.evidence_filter is not None and r.evidence_filter.judge_calls > 0
    assert len(r.evidence) + len(r.filtered_evidence) == r.evidence_filter.pool_size
    ev_steps = [t for t in r.trace if t.step == "evidence"]
    assert ev_steps[0].output["filter"] == {"mode": "off"} and "tau_q" in ev_steps[1].decision


def test_v2_always_mode_judges_first_round(settings, smoke_kb):
    s = settings.model_copy(update={"doc_filter_stage": "always"})
    llm = ScriptedLLM([("Mara Ostberg [1]", [1])])
    r = make_system(s, smoke_kb, llm, QuestionStyleNLI()).run_adaptive("Who created the Quillon messaging protocol?")
    assert r.evidence_filter is not None and r.stats.llm_calls == 1 + r.evidence_filter.judge_calls + r.verification.llm_calls


def test_nli_contradiction_needs_corroboration(settings):
    class ContradictsEverything:
        name = "contradicts"

        def predict(self, pairs):
            return [{"entailment": 0.01, "contradiction": 0.97} for _ in pairs]

    evidence = [ev(1, "MC Hammer launched the Too Legit to Quit Tour in 1992.", 5.0)]
    ans = GeneratedAnswer(answer="MC Hammer had a Too Legit to Quit Tour [1].", cited_refs=[1], insufficient=False)
    res, _ = VerifierAgent(settings, ContradictsEverything(), ScriptedLLM([])).verify("Who had a Too Legit To Quit Tour?", ans, evidence)
    assert res.status == "partially_supported" and "conflicting signals" in res.claims[0].reason
    no_x = settings.model_copy(update={"contradiction_cross_check": False})
    res2, _ = VerifierAgent(no_x, ContradictsEverything(), ScriptedLLM([])).verify("Who had a Too Legit To Quit Tour?", ans, evidence)
    assert res2.status == "contradicted"
    judge_no = ScriptedLLM([], judge_fn=lambda q, p, a: -5.0)
    res3, _ = VerifierAgent(settings, ContradictsEverything(), judge_no).verify("Who had a Too Legit To Quit Tour?", ans, evidence)
    assert res3.status == "contradicted"  # both verifiers disagree with the claim


def test_v2_pipeline_partial_answer_lists_unresolved_claims(settings, smoke_kb):
    llm = ScriptedLLM([("Quillon servers listen on TCP port 7425 by default [1]. The protocol is used by banks [1].", [1])])
    s = settings.model_copy(update={"max_retries": 0})
    r = make_system(s, smoke_kb, llm, FakeNLI()).run_adaptive("Which TCP port do Quillon servers listen on by default?")
    assert r.status == "answered" and r.support_level == "partial"
    assert r.unresolved_claims == ["The protocol is used by banks."]


def test_v2_pipeline_abstains_on_unanswerable(settings, smoke_kb):
    llm = ScriptedLLM([None])
    v2 = settings.model_copy(update={"question_routing": False})  # evaluated v2: two escalation rounds
    r = make_system(v2, smoke_kb, llm, FakeNLI()).run_adaptive("Which encryption algorithm does Quillon use?")
    assert r.status == "abstained" and r.support_level == "none" and r.stats.retrieval_attempts == 3


def test_retrieval_failure_is_recorded_and_handled(settings, smoke_kb, monkeypatch):
    sys_ = make_system(settings, smoke_kb, ScriptedLLM([("x", [1])]))
    monkeypatch.setattr(sys_.evidence_agent, "retrieve", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("index offline")))
    r = sys_.run_adaptive("Who created the Quillon messaging protocol?")
    assert r.status == "abstained"
    assert any(t.step == "evidence" and t.status == "failed" and "index offline" in t.note for t in r.trace)


# ------------------------------------------------------------------ LLM interface robustness (mocked HTTP)
def _llm():
    return OllamaLLM("http://ollama.test", "m", 2048, 64, 1.0)


def test_ollama_timeout_raises_llm_error(monkeypatch):
    def boom(*a, **k):
        raise httpx.ReadTimeout("timed out")
    monkeypatch.setattr(httpx, "post", boom)
    with pytest.raises(LLMError):
        _llm().judge("q", "p")


def test_ollama_malformed_answer_json(monkeypatch):
    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "this is not json"}}
    monkeypatch.setattr(httpx, "post", lambda *a, **k: R())
    with pytest.raises(LLMError, match="valid JSON"):
        _llm().answer("q", [ev(1, "t")])


def test_ollama_missing_logprobs_raises(monkeypatch):
    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "Yes"}}
    monkeypatch.setattr(httpx, "post", lambda *a, **k: R())
    with pytest.raises(LLMError, match="log-probabilities"):
        _llm().judge("q", "p")
