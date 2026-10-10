# Testing

## Summary of the latest run (2026-10-10)

| Check | Command | Result |
|---|---|---|
| Unit + API tests (mocked / dev models) | `pytest` | **98 passed**, 4 skipped (the opt-in real-model tests) |
| Real-model integration tests | `$env:ARAG_RUN_REAL_MODEL_TESTS="1"; pytest -m real_model` | **4 passed** (27 s) |
| Python lint | `ruff check src tests evaluation` | no issues |
| Frontend lint | `cd frontend; npm run lint` | no issues |
| Frontend production build | `cd frontend; npm run build` | success, 7 routes prerendered |
| Live end-to-end (real models) | upload → query → compare → evaluation via HTTP against running servers | passed (see below) |
| Visual check | headless Microsoft Edge screenshots of the pages, light and dark, desktop and 375 px width | layout and both themes render correctly |

The Claude-in-Chrome browser extension was not available, so interactive clicking through the UI was not automated;
the pages were checked by HTTP end-to-end calls plus rendered screenshots. Please also try the UI manually.

## Mocked vs real-model tests

| File | What it covers | Models |
|---|---|---|
| `tests/test_config_ingestion.py` | settings validation, env vars, safe filenames, PDF/DOCX/TXT/MD extraction, scanned-PDF rejection, corrupt files, chunking | none |
| `tests/test_retrieval.py` | BM25, RRF, hash embedder, indexing, citation metadata, PDF page numbers, duplicates, deletion, persistence, automatic re-index | dev hash embedder |
| `tests/test_agents.py` | each agent: planner classification/decomposition/focus/replan; evidence; verifier labels, numbers, NLI contradiction, citation consistency; confidence formula, missing components, every decision branch | **mock LLM** (`ScriptedLLM`) and **fake NLI** |
| `tests/test_pipeline.py` | empty KB, Standard RAG citations, adaptive accept / retry→accept / abstain after max retries / generator declines / small talk / LLM failure, retry budget | mock LLM, fake NLI, dev generator |
| `tests/test_evaluation.py` | metric definitions, dataset validation, path traversal, deterministic sampling, runner, CSV/JSON output, benchmark/user KB separation | dev models |
| `tests/test_api.py` | all endpoints, upload validation, query/compare/runs, evaluation lifecycle and export, error codes | dev models |
| `tests/test_real_models.py` | fastembed semantic similarity, NLI entail/contradict, Ollama grounded JSON answer + "insufficient" behaviour, full real pipeline | **real models** (opt-in, marked `real_model`) |

Mocks live in `tests/conftest.py`:

- `ScriptedLLM` returns pre-programmed answers and counts calls, so retry logic is tested deterministically.
- `FakeNLI` "entails" when every hypothesis term is in the premise and "contradicts" negated claims.
- `make_pdf()` builds a tiny valid PDF in memory for extraction tests.

Real-model tests are **skipped** (and reported as skipped) unless `ARAG_RUN_REAL_MODEL_TESTS=1`, so the basic suite
never needs a model download.

## Live end-to-end checks performed (real mode)

- Uploaded the two sample documents → both indexed (3 and 2 chunks).
- CORS pre-flight from `http://localhost:3000` → allowed.
- "How many books can an undergraduate student borrow and for how long?" → answered, cited the library file,
  verifier *supported*, confidence 0.83 → accept, 1 attempt.
- "What is the WiFi password of the library?" → **abstained** after 3 attempts (information not in documents).
- `/api/compare` on "What is the fine for an overdue book?" → both answered "2 rupees per day", both supported.
- Benchmarks on the smoke set and the TriviaQA sample (see [results.md](results.md)).

## Defects found and fixed during validation

| Defect | Fix |
|---|---|
| Qdrant local mode re-loaded old points after delete + re-create of a collection (dimension mismatch on re-index) | Close the client and delete the storage folder before re-creating |
| A corrupt DOCX raised `BadZipFile` instead of a clean error | All parser exceptions are converted to `ExtractionError` |
| Windows briefly locked `registry.json` while indexing 475 documents (`PermissionError`) | Short retry on replace + bulk indexing writes the registry once |
| Verifier marked short correct answers ("Edda Larsk") as *contradicted* | Short answers are turned into statements before NLI |
| Compound comparison sentences marked *unsupported* | Claims are split into clauses; cited passages are also checked together |
| Unanswerable "price" question accepted in the first prototype | Planner focus terms added to the coverage component |
| Frontend request timer never cleared | `clearTimeout` in `finally` |
