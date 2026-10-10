"""Dataset loading/validation, metric definitions and the benchmark runner (dev-mode models)."""

import json

import pytest

from adaptive_rag.evaluation.datasets import list_datasets, load_dataset, validate_dataset
from adaptive_rag.evaluation.metrics import aggregate, contains_answer, exact_match, token_f1
from adaptive_rag.evaluation.runner import run_evaluation, save_results, select_questions
from tests.conftest import ROOT, dev_bundle


# ---------------------------------------------------------------- metrics
def test_exact_match_uses_squad_normalisation():
    assert exact_match("The Eiffel Tower.", ["eiffel tower"]) == 1.0
    assert exact_match("Eiffel", ["Eiffel Tower"]) == 0.0


def test_token_f1():
    assert token_f1("Edda Larsk", ["Edda Larsk"]) == 1.0
    assert token_f1("founded by Edda", ["Edda Larsk"]) == pytest.approx(2 * (1 / 3) * (1 / 2) / (1 / 3 + 1 / 2))
    assert token_f1("nothing", ["Edda Larsk"]) == 0.0


def test_contains_answer_matches_whole_words_only():
    assert contains_answer("It was founded in 1847 by Edda Larsk.", ["edda larsk"])
    assert contains_answer("It has 4,000 cycles", ["4,000"])
    assert not contains_answer("port 74250", ["7425"])
    assert not contains_answer("anything", [])


def row(answerable, abstained, correct, cited_gold=None, ev=None, latency=10.0):
    return {"answerable": answerable, "abstained": abstained, "correct": correct, "em": float(correct),
            "f1": float(correct), "cited_gold": cited_gold, "evidence_has_answer": ev, "support_score": 1.0,
            "latency_ms": latency, "llm_calls": 1, "retrieval_attempts": 1, "error": None}


def test_aggregate_metric_definitions():
    rows = [row(True, False, True, True, True), row(True, False, False, False, True), row(True, True, False, None, False),
            row(False, True, False), row(False, False, False), {**row(True, True, False), "error": "boom"}]
    m = aggregate(rows)
    assert m["n_total"] == 6 and m["n_errors"] == 1 and m["n_answerable"] == 3 and m["n_unanswerable"] == 2
    assert m["accuracy"] == pytest.approx(1 / 3, abs=1e-3)
    assert m["answer_rate_answerable"] == pytest.approx(2 / 3, abs=1e-3)
    assert m["selective_accuracy"] == 0.5
    assert m["correct_abstention_rate"] == 0.5 and m["wrong_abstention_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert m["citation_gold_rate"] == 0.5 and m["evidence_recall"] == pytest.approx(2 / 3, abs=1e-3)


def test_aggregate_handles_empty_denominators():
    m = aggregate([row(True, False, True)])
    assert m["correct_abstention_rate"] is None and m["citation_gold_rate"] is None


# ---------------------------------------------------------------- datasets
def test_smoke_dataset_is_valid_and_labelled_synthetic():
    rep = validate_dataset(ROOT / "data" / "datasets" / "smoke_test")
    assert rep.ok and rep.num_usable == 24 and rep.num_answerable == 18 and rep.num_unanswerable == 6
    ds = load_dataset(ROOT / "data" / "datasets", "smoke_test")
    assert ds.manifest["synthetic"] is True and len(ds.corpus) == 8


def test_invalid_dataset_rows_are_reported(tmp_path):
    d = tmp_path / "bad"
    d.mkdir()
    (d / "dataset.json").write_text(json.dumps({"name": "bad"}))
    (d / "corpus.jsonl").write_text(json.dumps({"doc_id": "a", "title": "A", "text": "text"}) + "\n")
    (d / "questions.jsonl").write_text("\n".join(json.dumps(q) for q in [
        {"id": "1", "question": "Good question?", "answers": ["x"], "answerable": True, "gold_doc_ids": ["a"]},
        {"id": "1", "question": "Duplicate id?", "answers": ["x"], "answerable": True},
        {"id": "3", "question": "Missing gold?", "answers": ["x"], "answerable": True, "gold_doc_ids": ["zzz"]},
        {"id": "4", "question": "No answers?", "answers": [], "answerable": True},
    ]))
    rep = validate_dataset(d)
    assert rep.ok and rep.num_usable == 1 and len(rep.issues) == 3
    assert len(load_dataset(tmp_path, "bad").questions) == 1
    assert {x["name"] for x in list_datasets(tmp_path)} == {"bad"}


def test_missing_and_path_traversal_datasets_rejected(tmp_path):
    assert not validate_dataset(tmp_path / "nope").ok
    with pytest.raises(FileNotFoundError):
        load_dataset(ROOT / "data" / "datasets", "../datasets/smoke_test/..")


def test_question_sampling_is_deterministic():
    qs = [{"id": str(i)} for i in range(50)]
    a, b = select_questions(qs, 10, seed=0), select_questions(qs, 10, seed=0)
    assert a == b and len(a) == 10 and [int(q["id"]) for q in a] == sorted(int(q["id"]) for q in a)
    assert select_questions(qs, None, 0) == qs


# ---------------------------------------------------------------- runner
def test_runner_evaluates_both_methods_on_same_questions_and_saves(settings):
    res = run_evaluation(settings, dev_bundle(), "smoke_test", ["standard", "adaptive"], limit=8, seed=1)
    ids = {m: [r["question_id"] for r in res["rows"] if r["method"] == m] for m in res["methods"]}
    assert ids["standard"] == ids["adaptive"] and len(ids["standard"]) == 8
    assert res["mode"] == "dev" and "NOT real-model" in res["dev_mode_warning"]
    assert res["summary"]["standard"]["n_total"] == 8 and res["config"]["top_k"] == settings.top_k
    out = save_results(res, settings.results_dir)
    assert (out / "results.json").exists()
    csv_lines = (out / "results.csv").read_text(encoding="utf-8").splitlines()
    assert len(csv_lines) == 1 + 16 and csv_lines[0].startswith("question_id,method")


def test_runner_keeps_benchmark_corpus_separate_from_user_kb(settings):
    run_evaluation(settings, dev_bundle(), "smoke_test", ["standard"], limit=2)
    assert (settings.runtime_dir / "benchmarks" / "smoke_test" / "registry.json").exists()
    assert not (settings.runtime_dir / "kb").exists()
