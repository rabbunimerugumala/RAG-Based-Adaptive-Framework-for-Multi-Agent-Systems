"""Unit tests for each of the four agents (mocked models)."""

import pytest

from adaptive_rag.agents.confidence import ConfidenceAgent
from adaptive_rag.agents.evidence import EvidenceAgent
from adaptive_rag.agents.planner import PlannerAgent, classify, subquestions
from adaptive_rag.agents.verifier import VerifierAgent, atomic_claims, declarative, lexical_support
from adaptive_rag.generation.llm import ExtractiveGenerator
from adaptive_rag.schemas import Evidence, GeneratedAnswer, RetrievalPlan, VerificationResult
from tests.conftest import FakeNLI, ScriptedLLM


def ev(ref, text, overlap=1.0, doc="d"):
    return Evidence(ref=ref, chunk_id=f"{doc}:{ref}", doc_id=doc, filename=f"{doc}.txt", chunk_index=ref,
                    text=text, rrf_score=0.01, keyword_overlap=overlap)


def ans(text, refs=(1,)):
    return GeneratedAnswer(answer=text, cited_refs=list(refs), insufficient=False)


# ================================================================ Planner
@pytest.mark.parametrize("q,expected", [
    ("What is the difference between TCP and UDP?", "comparison"),
    ("Who founded the town and when was it founded?", "multi_part"),
    ("How many employees does Orvane Labs have?", "numeric"),
    ("What is photosynthesis?", "definition"),
    ("How do I reset the device?", "procedural"),
    ("Who founded Varnholm?", "factoid"),
])
def test_planner_classifies_queries(q, expected):
    assert classify(q) == expected


def test_planner_decomposes_comparison_and_multipart():
    assert subquestions("What is the difference between the Kestrel-9 and the Kestrel-12?", "comparison") == [
        "What is the Kestrel-9?", "What is the Kestrel-12?"]
    subs = subquestions("Who founded Varnholm and when was the town hall built?", "multi_part")
    assert len(subs) == 2 and subs[1].startswith("when")


def test_planner_skips_retrieval_for_small_talk(settings):
    plan = PlannerAgent(settings, ExtractiveGenerator()).plan("Hello!")
    assert plan.needs_retrieval is False and plan.search_queries == []


def test_planner_builds_structured_plan(settings):
    plan = PlannerAgent(settings, ExtractiveGenerator()).plan(
        "What is the difference between the Kestrel-9 and the Kestrel-12?")
    assert plan.needs_retrieval and plan.query_type == "comparison"
    assert plan.search_queries[0] == plan.query and len(plan.search_queries) >= 3
    assert plan.top_k == settings.top_k + 2 and plan.keywords and plan.explanation


def test_planner_extracts_focus_terms(settings):
    plan = PlannerAgent(settings, ExtractiveGenerator()).plan("What is the retail price of the Kestrel-9?")
    assert plan.focus_terms == ["retail", "price"]


def test_replan_targets_unsupported_claims_and_uses_llm_once(settings):
    llm = ScriptedLLM([])
    planner = PlannerAgent(settings, llm)
    first = planner.plan("What is the energy density of the Kestrel-9?")
    ver = VerificationResult(status="partially_supported", support_score=0.5, citation_consistency=0.5,
                             unsupported_claims=["It weighs 12 kilograms"], method="test", summary="")
    plan, calls = planner.replan(first, "targeted", [ev(1, "unrelated text")], ver, attempt=1)
    assert calls == 1 and llm.rewrite_calls == 1
    assert "weigh 12 kilogram" in " ".join(plan.search_queries)
    assert "kestrel energy density" in plan.search_queries
    assert plan.top_k == first.top_k + 2 and plan.strategy == "retry-1:targeted"


def test_replan_broaden_increases_candidates_without_llm(settings):
    planner = PlannerAgent(settings, ExtractiveGenerator())
    first = planner.plan("What is the storage capacity of Brightwater Reservoir?")
    plan, calls = planner.replan(first, "broaden", [], None, attempt=1)
    assert calls == 0 and plan.candidate_k == first.candidate_k * 2 and plan.top_k == first.top_k + 3
    assert len(plan.search_queries) > len(first.search_queries)


# ================================================================ Evidence
def test_evidence_agent_runs_all_planned_queries(settings, smoke_kb):
    plan = PlannerAgent(settings, ExtractiveGenerator()).plan("Who created the Quillon messaging protocol?")
    pkg = EvidenceAgent(settings.model_copy(update={"doc_filter": "off"}), smoke_kb, None).retrieve(plan)
    evidence, summary = pkg.evidence, pkg.summary
    assert evidence[0].doc_id == "quillon_protocol"
    assert summary["ranked_lists_fused"] == 2 * len(plan.search_queries)
    assert len(evidence) == plan.top_k and [e.ref for e in evidence] == list(range(1, plan.top_k + 1))


# ================================================================ Verifier
def test_lexical_support_requires_numbers_to_match():
    p = "The dam is 61 metres high."
    assert lexical_support("The dam is 61 metres high", p) == 1.0
    assert lexical_support("The dam is 75 metres high", p) <= 0.4


def test_atomic_claims_and_declarative_rewrites():
    assert len(atomic_claims("The Kestrel-9 is sodium-ion based, while the Kestrel-12 uses iron phosphate")) == 2
    assert atomic_claims("A short claim, while b") == ["A short claim, while b"]
    assert declarative("Who founded Varnholm?", "Edda Larsk") == "Edda Larsk founded Varnholm."
    assert declarative("In what year was Varnholm founded?", "1847") == "Varnholm was founded in 1847."


def test_verifier_supported_partial_unsupported_lexical_only(settings):
    v = VerifierAgent(settings, None)
    evidence = [ev(1, "Varnholm was founded in 1847 by the miller Edda Larsk.")]
    res, calls = v.verify("q", ans("Varnholm was founded in 1847 by Edda Larsk [1]."), evidence)
    assert res.status == "supported" and res.support_score == 1.0 and calls == 0
    assert "lexical-only" in res.method
    res, _ = v.verify("q", ans("Varnholm was founded in 1847 [1]. It has a famous zoo with penguins [1]."), evidence)
    assert res.status == "partially_supported" and res.unsupported_claims == ["It has a famous zoo with penguins."]
    res, _ = v.verify("q", ans("Penguins live in Antarctica [1]."), evidence)
    assert res.status == "unsupported" and res.support_score == 0.0


@pytest.mark.parametrize("mode", ["v1", "grounded"])
def test_verifier_with_nli_detects_contradiction(settings, mode):
    settings = settings.model_copy(update={"verifier_mode": mode})
    v = VerifierAgent(settings, FakeNLI())
    evidence = [ev(1, "Swimming in the reservoir is prohibited.")]
    res, calls = v.verify("q", ans("Swimming in the reservoir is not prohibited [1]."), evidence)
    assert res.status == "contradicted" and res.contradicted_claims and calls == 1


def test_verifier_does_not_accept_lexical_overlap_without_entailment(settings):
    """With NLI available, word overlap alone must not produce 'supported'."""

    class NeutralNLI:
        name = "neutral"

        def predict(self, pairs):
            return [{"entailment": 0.1, "contradiction": 0.1} for _ in pairs]

    res, _ = VerifierAgent(settings, NeutralNLI()).verify(
        "q", ans("Varnholm was founded in 1847 [1]."), [ev(1, "Varnholm was founded in 1847.")])
    assert res.status == "partially_supported"


def test_verifier_checks_citation_consistency(settings):
    evidence = [ev(1, "Quillon uses port 7425."), ev(2, "The Lantern Fair is in October.")]
    res, _ = VerifierAgent(settings, None).verify("q", ans("Quillon uses port 7425 [2].", refs=[2]), evidence)
    assert res.status == "supported" and res.citation_consistency == 0.0


def test_verifier_not_applicable_when_generator_declines(settings):
    res, _ = VerifierAgent(settings, None).verify(
        "q", GeneratedAnswer(answer="x", cited_refs=[], insufficient=True), [ev(1, "t")])
    assert res.status == "not_applicable" and res.support_score is None


# ================================================================ Confidence
def plan_for(keywords, focus=(), subs=()):
    return RetrievalPlan(query="q", needs_retrieval=True, query_type="factoid", keywords=list(keywords),
                         focus_terms=list(focus), subquestions=list(subs), search_queries=["q"], top_k=5,
                         candidate_k=20, strategy="initial", explanation="")


def ver(status="supported", support=1.0, consistency=1.0, unsupported=()):
    return VerificationResult(status=status, support_score=support, citation_consistency=consistency,
                              unsupported_claims=list(unsupported), method="t", summary="")


def test_confidence_formula_matches_documented_weights(settings):
    agent = ConfidenceAgent(settings)
    comps = {"relevance": 0.4, "coverage": 0.6, "verification": 1.0, "consistency": 0.5}
    score, missing = agent.score(comps)
    assert missing == [] and score == pytest.approx(0.25 * 0.4 + 0.25 * 0.6 + 0.35 * 1.0 + 0.15 * 0.5, abs=1e-4)


def test_confidence_missing_components_are_reported_and_renormalised(settings):
    score, missing = ConfidenceAgent(settings).score(
        {"relevance": 1.0, "coverage": 0.5, "verification": None, "consistency": None})
    assert missing == ["verification", "consistency"]
    assert score == pytest.approx((0.25 * 1.0 + 0.25 * 0.5) / 0.5, abs=1e-4)
    assert ConfidenceAgent(settings).score(dict.fromkeys(["relevance", "coverage", "verification", "consistency"])) == (
        None, ["relevance", "coverage", "verification", "consistency"])


def test_confidence_components(settings):
    agent = ConfidenceAgent(settings)
    comps = agent.components(plan_for(["kestrel", "price"], focus=["price"]),
                             [ev(1, "Kestrel battery", overlap=0.5)], ver())
    assert comps["relevance"] == 0.5
    assert comps["coverage"] == pytest.approx((0.5 + 0.0) / 2)


def test_confidence_accepts_strong_supported_answer(settings):
    c = ConfidenceAgent(settings).assess(plan_for(["kestrel"]), [ev(1, "kestrel")], ans("x"), ver(), retries_left=2)
    assert c.decision == "accept" and c.score >= settings.accept_threshold and "calibrated" in c.note


def test_confidence_retries_weak_answer_then_abstains(settings):
    agent = ConfidenceAgent(settings)
    weak = ver("unsupported", 0.0, 0.0, unsupported=["claim"])
    c = agent.assess(plan_for(["kestrel"]), [ev(1, "kestrel")], ans("x"), weak, retries_left=1)
    assert c.decision == "retry" and c.retry_strategy == "targeted"
    c = agent.assess(plan_for(["kestrel"]), [ev(1, "kestrel")], ans("x"), weak, retries_left=0)
    assert c.decision == "abstain"


def test_confidence_broadens_when_retrieval_is_weak(settings):
    c = ConfidenceAgent(settings).assess(plan_for(["kestrel", "price"]), [ev(1, "other", overlap=0.0)],
                                         ans("x"), ver("partially_supported", 0.5, 0.5), retries_left=2)
    assert c.decision == "retry" and c.retry_strategy == "broaden"


def test_confidence_contradiction_and_insufficient_paths(settings):
    agent = ConfidenceAgent(settings)
    c = agent.assess(plan_for(["k"]), [ev(1, "k")], ans("x"), ver("contradicted", 0.0, 0.0), retries_left=0)
    assert c.decision == "abstain" and "contradicted" in c.reason
    no = GeneratedAnswer(answer="", cited_refs=[], insufficient=True)
    c = agent.assess(plan_for(["k"]), [ev(1, "k")], no, ver("not_applicable", None, None), retries_left=1)
    assert c.decision == "retry" and "verification" in c.missing_components
    c = agent.assess(plan_for(["k"]), [], no, ver("not_applicable", None, None), retries_left=0)
    assert c.decision == "abstain"


def test_confidence_fallback_accepts_moderate_score_after_retries(settings):
    c = ConfidenceAgent(settings).assess(plan_for(["kestrel", "price"]), [ev(1, "kestrel", overlap=0.5)],
                                         ans("x"), ver("partially_supported", 0.5, 1.0), retries_left=0)
    assert settings.abstain_threshold <= c.score < settings.accept_threshold
    assert c.decision == "accept" and "moderate" in c.reason
