# Testing

## Summary

| Check | Command | Latest result |
|---|---|---|
| Unit + API tests (mocked / dev models) | `pytest` | **156 passed**, 4 skipped (the opt-in real-model tests) — re-run 2026-10-11 on `main` |
| Real-model integration tests | `$env:ARAG_RUN_REAL_MODEL_TESTS="1"; pytest -m real_model` | **4 passed** (80 s) on `main` (2026-10-11) |
| Python lint | `ruff check src tests evaluation` | no issues |
| Frontend type check / lint / build | `cd frontend; npx tsc --noEmit; npm run lint; npm run build` | no issues; 7 routes prerendered |
| UI check | headless Microsoft Edge screenshots of Ask, Agent Trace, Evaluation, Overview (light and dark) against a running backend in dev mode | pages render real backend data; see notes below |

Not automated: interactive clicking in a real browser (the browser-automation extension was not available). The
pages were checked with HTTP calls plus rendered screenshots; please also try the UI manually.

## Mocked vs real-model tests

| File | Covers | Models |
|---|---|---|
| `test_config_ingestion.py` | settings validation, env vars, filenames, PDF/DOCX/TXT/MD extraction, scanned/corrupt files, chunking | none |
| `test_retrieval.py` | BM25, RRF, embedder, indexing, citation metadata, PDF pages, duplicates, deletion, persistence, re-index | dev hash embedder |
| `test_agents.py` | planner routing/decomposition/focus/replan; evidence; verifier labels, numbers, NLI contradiction (v1 and grounded), citation consistency; confidence formula, missing components, every decision branch | mock LLM, fake NLI |
| `test_pipeline.py` | the **original v1** pipeline pinned by `LEGACY_V1`: empty KB, citations, accept, retry→accept, abstain after max retries, retry budget, generator failure | mock LLM, fake NLI |
| `test_multiagent_v2.py` | judge log-odds and floor; adaptive bar (mean, σ); filtering, ties, all-filtered fallback, evidence budget, metadata preservation; dedup; judge cache; triplet mode; judge failure fallback; escalation on retry; v1 short-answer failure reproduced; grounded verifier accepts correct short answers, rejects absent ones, needs judge confirmation; real contradictions still detected; uncited off-topic contradictions ignored; contradiction corroboration; v2 attempt selection; judge-based relevance; partial answers with unresolved claims; abstention; retrieval failure; Ollama timeout, malformed JSON, missing log-probabilities | mock LLM, fake NLI, mocked HTTP |
| `test_evaluation.py` | metric definitions, dataset validation, path traversal, deterministic sampling, runner, CSV/JSON, benchmark/user KB separation | dev models |
| `test_routing.py` | question routing (simple / complex / ambiguous), per-question escalation budget, lightweight round, escalation on insufficient or contradicted answers, hardware profiles, explicit settings beating profiles, routing on by default and disable-able, pinned benchmark variants | mock LLM, fake NLI |
| `test_api.py` | all endpoints incl. `/api/ready`, `/api/variants`, multi-agent response fields, upload validation, evaluation lifecycle/export, unknown variants, error codes | dev models |
| `test_real_models.py` | fastembed similarity, NLI entail/contradict, Ollama grounded JSON answer and "insufficient", full real pipeline | **real models** (opt-in) |

Mocks (`tests/conftest.py`): `ScriptedLLM` (scripted drafts, a keyword-based judge, call counters), `FakeNLI`,
`make_pdf()`. Real-model tests are skipped unless `ARAG_RUN_REAL_MODEL_TESTS=1`.

## Defects found and fixed

| Defect | Fix |
|---|---|
| Original v1: verifier rejected correct short answers ("Niger") via ill-formed NLI hypotheses (measured on the original benchmark) | grounded verifier: no contradiction signal from fallback hypotheses; answer-text + judge check for short answers |
| Original v1: any off-topic passage could veto an answer by NLI contradiction | contradiction only from cited/judge-relevant related windows where it beats entailment, and must be corroborated by the judge |
| Original v1: final selection could prefer an "insufficient" attempt over an answered one | v2 selection ranks answered, verified attempts first |
| v2 iteration 1: first-round filtering removed answer passages and cost ~14 LLM calls/question (development set) | escalation: filtering only when more evidence is requested |
| v2 development: "short answer" rule counted content words, treating a full sentence as short | short answer = at most 4 words |
| Qdrant local mode reloaded old points after collection re-create | delete and reopen the storage folder |
| Corrupt DOCX raised `BadZipFile` | all parser errors become `ExtractionError` |
| Windows briefly locked `registry.json` during bulk indexing | retry on replace + one registry write per bulk index |
| TriviaQA prep crashed with a relative `--out` path; held-out set shared 1 question with the development set | path resolved; `--exclude-dataset` removes shared questions (recorded in the manifest) |
| Evaluation chart rendered no bars headlessly; value labels collided with 5 systems; legend sorted alphabetically | animation off; direct labels only for ≤ 2 systems; legend keeps system order |
| Frontend request timer was never cleared | `clearTimeout` in `finally` |
