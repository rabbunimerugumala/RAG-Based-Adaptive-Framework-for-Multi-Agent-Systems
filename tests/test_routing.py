"""Question routing (Planner) and hardware profiles - mocked models."""

import pytest

from adaptive_rag.agents.planner import PlannerAgent, classify, route_question, subquestions
from adaptive_rag.config import Settings
from adaptive_rag.evaluation.runner import VARIANTS
from adaptive_rag.generation.llm import ExtractiveGenerator
from adaptive_rag.text import keywords
from tests.conftest import FakeNLI, ScriptedLLM, make_system


def route(q):
    qt = classify(q)
    return route_question(q, qt, subquestions(q, qt), keywords(q))[0]


@pytest.mark.parametrize("q,expected", [
    ("Who founded Varnholm?", "simple"),
    ("What is the energy density of the Kestrel-9 battery?", "simple"),
    ("What is the difference between the Kestrel-9 and the Kestrel-12?", "complex"),
    ("Who founded Varnholm and when was the town hall built?", "complex"),
    ("What mythological beast has the head of a man, the body of a lion, and the tail and feet of a dragon in old tales?", "complex"),
    ("Why?", "ambiguous"),
    ("What does it cost?", "ambiguous"),
    ("They changed the opening hours when?", "ambiguous"),
    ("It was built by which Varnholm miller?", "simple"),
])
def test_route_question(q, expected):
    assert route(q) == expected


def routed(settings):
    return settings.model_copy(update={"question_routing": True})


def test_planner_sets_budget_by_route(settings):
    p = PlannerAgent(routed(settings), ExtractiveGenerator())
    simple = p.plan("Who founded Varnholm?")
    complex_ = p.plan("What is the difference between the Kestrel-9 and the Kestrel-12?")
    assert (simple.route, simple.retry_budget) == ("simple", 1)
    assert (complex_.route, complex_.retry_budget) == ("complex", settings.max_retries)
    assert "Route: simple" in simple.explanation


def test_routing_off_keeps_original_behaviour(settings):
    off = settings.model_copy(update={"question_routing": False})
    plan = PlannerAgent(off, ExtractiveGenerator()).plan("Who founded Varnholm?")
    assert plan.route == "full" and plan.retry_budget == settings.max_retries


def test_simple_question_answered_in_lightweight_round(settings, smoke_kb):
    llm = ScriptedLLM([("Mara Ostberg created Quillon [1].", [1])])
    r = make_system(routed(settings), smoke_kb, llm, FakeNLI()).run_adaptive("Who created the Quillon messaging protocol?")
    assert r.status == "answered" and r.route == "simple" and r.stats.retrieval_attempts == 1
    assert r.evidence_filter is None and r.stats.llm_calls == 1 + r.verification.llm_calls
    assert next(t for t in r.trace if t.step == "planner").decision == "route: simple"


def test_simple_question_escalates_once_to_full_workflow(settings, smoke_kb):
    llm = ScriptedLLM([None])  # generator never finds the answer
    r = make_system(routed(settings), smoke_kb, llm, FakeNLI()).run_adaptive("Which encryption algorithm does Quillon use?")
    assert r.route == "simple" and r.status == "abstained"
    assert r.stats.retrieval_attempts == 2  # light round + one full escalation round
    ev = [t for t in r.trace if t.step == "evidence"]
    assert ev[0].output["filter"] == {"mode": "off"} and "tau_q" in ev[1].decision  # escalation used judged evidence


def test_simple_question_escalation_can_succeed(settings, smoke_kb):
    llm = ScriptedLLM([None, ("Quillon servers listen on TCP port 7425 by default [1].", [1])])
    r = make_system(routed(settings), smoke_kb, llm, FakeNLI()).run_adaptive(
        "Which TCP port do Quillon servers listen on by default?")
    assert r.status == "answered" and r.stats.retrieval_attempts == 2


def test_contradicted_simple_answer_escalates(settings, smoke_kb):
    llm = ScriptedLLM([("Swimming in Brightwater Reservoir is not prohibited [1].", [1]), None])
    s = routed(settings).model_copy(update={"contradiction_cross_check": False})
    r = make_system(s, smoke_kb, llm, FakeNLI()).run_adaptive("Is swimming in Brightwater Reservoir prohibited?")
    first_conf = next(t for t in r.trace if t.step == "confidence")
    assert first_conf.decision == "retry" and r.stats.retrieval_attempts == 2


def test_complex_question_keeps_full_budget(settings, smoke_kb):
    llm = ScriptedLLM([None])
    r = make_system(routed(settings), smoke_kb, llm, FakeNLI()).run_adaptive(
        "What is the difference between the Kestrel-9 and the Kestrel-12?")
    assert r.route == "complex" and r.stats.retrieval_attempts == settings.max_retries + 1


def test_routed_variant_registered():
    assert VARIANTS["adaptive_routed"] == ("adaptive", {"question_routing": True})


# ------------------------------------------------------------------ hardware profiles
def test_8gb_profile_defaults():
    s = Settings(hardware_profile="8gb", _env_file=None)
    assert s.ollama_model == "qwen2.5:1.5b" and s.nli_backend == "none" and s.question_routing is True
    assert s.ollama_num_ctx == 2048 and s.doc_pool_k == 6


def test_profile_does_not_override_explicit_settings(monkeypatch):
    monkeypatch.setenv("ARAG_OLLAMA_MODEL", "my-model")
    s = Settings(hardware_profile="8gb", _env_file=None)
    assert s.ollama_model == "my-model" and s.nli_backend == "none"


def test_defaults_enable_routing_on_the_evaluated_32gb_configuration():
    s = Settings(_env_file=None)
    assert s.hardware_profile == "32gb" and s.question_routing is True and s.simple_route_retries == 1
    assert s.ollama_model == "qwen3:4b-instruct-2507-q4_K_M" and s.doc_pool_k == 10 and s.max_retries == 2


def test_routing_can_be_disabled_by_environment(monkeypatch):
    monkeypatch.setenv("ARAG_QUESTION_ROUTING", "false")
    assert Settings(_env_file=None).question_routing is False
    monkeypatch.setenv("ARAG_HARDWARE_PROFILE", "8gb")  # an explicit setting also beats the profile default
    assert Settings(_env_file=None).question_routing is False


def test_documented_variants_are_pinned():
    from adaptive_rag.config import LEGACY_V1
    assert VARIANTS["adaptive_v2"][1] == {"question_routing": False}
    assert VARIANTS["adaptive_no_retry"][1]["question_routing"] is False
    assert VARIANTS["adaptive_no_filter"][1]["question_routing"] is False
    assert LEGACY_V1["question_routing"] is False
