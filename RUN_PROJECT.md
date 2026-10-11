# How to Install, Run, Test and Evaluate the Project (Windows guide)

This guide takes you from an empty Windows 10/11 machine to a running **Adaptive Multi-Agent RAG** system, then shows
how to use it, test it and reproduce the documented evaluation. All commands are for **Windows PowerShell**.

Legend used below:

* ✅ **verified** — the command was executed on the reference laptop (Windows 11, 32 GB RAM, RTX 3050 Laptop 4 GB,
  Python 3.12.10, Node 22.12.0, npm 10.9.0, Ollama 0.31.2) on 2026-10-11 at commit `ef88805` (+ documentation changes).
* ⚠️ **not re-run for this guide** — documented from the code and earlier project runs, but not executed again while
  writing this guide (e.g. multi-hour benchmarks, a completely fresh machine).

---

## A. Project overview

**Title:** RAG-Based Adaptive Framework for Multi-Agent Systems.

**Purpose:** answer questions from your own documents with evidence-grounded answers. Four cooperating agents plan the
search, gather and filter evidence, verify every claim against it, and then **answer, retrieve again, or abstain**.
A **Standard RAG** pipeline is kept as the comparison baseline.

| Agent | Responsibility (file in `src/adaptive_rag/agents/`) |
|---|---|
| **Planner** (`planner.py`) | analyses the question (type, keywords, focus terms, sub-questions), decides whether retrieval is needed, **routes** it (simple / complex / ambiguous), sets the evidence budget, re-plans targeted searches on a retry |
| **Evidence** (`evidence.py`) | hybrid retrieval (dense vectors in Qdrant + BM25 keywords, merged with reciprocal-rank fusion); when more evidence is requested, a MAIN-RAG-inspired filter: the shared LLM scores each candidate passage with `log P(Yes) − log P(No)`, keeps those above the per-question bar `τ = mean − n·σ`, ranks them |
| **Verifier** (`verifier.py`) | splits the draft answer into claims and checks each against the evidence: word/number support, a local NLI model, and for short answers the answer text plus the LLM judge |
| **Confidence** (`confidence.py`) | combines evidence relevance, coverage, verification and citation consistency into a heuristic score and decides: answer, retrieve again, or abstain |

**Adaptive routing (on by default):** simple factual questions get one lightweight round and, if the answer is
missing, contradicted or weakly supported, **one** escalation into the full four-agent workflow. Complex or ambiguous
questions get the full workflow with up to **two** escalation rounds.

**Technology:** Python 3.12, FastAPI, Pydantic, Qdrant (local mode, no server), own BM25, fastembed (ONNX) embeddings,
PyTorch + transformers (NLI verifier), Ollama (local LLM), Next.js 16 + React 19 + Tailwind CSS 4 frontend.

**Request workflow:**

```
Browser (http://localhost:3000) ──HTTP/JSON──► FastAPI (http://127.0.0.1:8000)
   Planner → Evidence (retrieve) → Draft answer (LLM) → Verifier → Confidence
        ├─ supported           → answer + citations
        ├─ weak, budget left   → Planner re-plans → Evidence (judge + filter) → Draft → Verifier → Confidence
        └─ still unsupported   → abstain ("evidence insufficient")
```

More detail: [docs/how_it_works.md](docs/how_it_works.md), [docs/architecture.md](docs/architecture.md).

---

## B. Prerequisites

| Requirement | Version / size | Source of the requirement |
|---|---|---|
| **Python** | 3.11, 3.12 or 3.13 (`requires-python = ">=3.11,<3.14"`); **3.12 is the tested version** | `pyproject.toml` |
| **Node.js** | **20.9.0 or newer** (Next.js 16 requirement); tested with 22.12.0 | `frontend/node_modules/next/package.json` |
| **npm** | comes with Node.js; tested 10.9.0 | – |
| **Git** | any recent version | – |
| **Ollama** | needed for real mode (local LLM); must return token log-probabilities — tested 0.31.2 | https://ollama.com/download |
| **Disk** | ~1.9 GB for the Python environment (1.3 GB incl. CPU PyTorch) and `frontend/node_modules` (0.56 GB); ~2.9 GB of models (LLM 2.5 GB, NLI 0.28 GB, embeddings 0.07 GB); +1 GB for the optional small model | measured |
| **RAM** | 32 GB used for all benchmarks; 16 GB and 8 GB profiles exist (section G) | measured / see G |
| **GPU** | **optional.** An NVIDIA GPU speeds up the LLM; with 4 GB VRAM the 4B model runs ~67 % on GPU, 33 % on CPU. Embeddings and NLI always run on CPU | measured |
| **Network** | needed once: pip/npm packages, `ollama pull`, first start (embedding + NLI model download), TriviaQA preparation | – |
| **Docker / paid APIs** | **not needed.** Everything runs locally; no API keys | – |

Check what you have:

```powershell
py -3.12 --version
node --version
npm --version
git --version
ollama --version
nvidia-smi          # optional - only if you have an NVIDIA GPU
```

---

## C. Repository structure

```
RAG-Based-Adaptive-Framework-for-Multi-Agent-Systems/
├── src/adaptive_rag/            Python backend package
│   ├── agents/                  planner.py · evidence.py · verifier.py · confidence.py   (the four agents)
│   ├── api/app.py               FastAPI application and all /api/* endpoints
│   ├── retrieval/               embeddings.py · bm25.py · fusion.py · hybrid.py · store.py (Qdrant local index)
│   ├── ingestion/               extract.py (PDF/DOCX/TXT/MD) · chunking.py
│   ├── generation/              llm.py (Ollama: answer, judge, predict) · nli.py (NLI verifier)
│   ├── evaluation/              datasets.py · metrics.py · runner.py (benchmark variants)
│   ├── config.py                ALL settings (ARAG_* environment variables), hardware profiles
│   ├── pipeline.py              Standard RAG and Multi-Agent RAG orchestration
│   └── schemas.py · models.py · text.py
├── frontend/                    Next.js dashboard (app/ pages, components/, lib/api.ts)
├── tests/                       pytest suite (mocked models) + test_real_models.py (opt-in)
├── evaluation/                  run_benchmark.py · analyze_results.py · prepare_triviaqa.py · validate_dataset.py
│   └── results/                 benchmark outputs (git-ignored) + MANIFEST.md (checksums, committed)
├── data/
│   ├── datasets/smoke_test/     24-question synthetic test set (committed)
│   ├── datasets/triviaqa_*/     TriviaQA samples (git-ignored, created by a script)
│   ├── sample_documents/        demo files to upload
│   └── runtime/                 generated at run time, git-ignored: kb/ (your document index), benchmarks/
│                                (benchmark indexes), runs/ (saved answers + traces), model_cache/, logs/
├── docs/                        documentation
├── .env.example                 configuration template (copy to .env)
├── pyproject.toml               Python dependencies
├── README.md · RUN_PROJECT.md · LICENSE
```

---

## D. First-time installation

Run every command **from the repository root** unless the step says `cd frontend`.

### D1. Clone the repository ⚠️ (URL verified; clone not re-run)

```powershell
git clone https://github.com/rabbunimerugumala/RAG-Based-Adaptive-Framework-for-Multi-Agent-Systems.git
cd RAG-Based-Adaptive-Framework-for-Multi-Agent-Systems
```

### D2. Python virtual environment and dependencies ⚠️ (these exact commands built the reference environment)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# full install: real mode with the NLI verifier (CPU build of PyTorch, ~200 MB download)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[nli,dev]"

# optional: needed only to prepare the TriviaQA benchmark samples
pip install -e ".[datasets]"
```

Without the `nli` extra the app still runs in real mode; the NLI model is reported as unavailable in `/api/health`
and the verifier uses lexical checks plus the LLM judge.

If `Activate.ps1` is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again.

### D3. Frontend dependencies ⚠️ (re-run not needed for this guide; `npm run build` ✅)

```powershell
cd frontend
npm install
Copy-Item .env.example .env.local      # contains NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
cd ..
```

### D4. Backend configuration (`.env`)

```powershell
Copy-Item .env.example .env
notepad .env          # change ARAG_MODE=dev to ARAG_MODE=real to use the local models
```

* `ARAG_MODE=dev` (the default) — **no models, no LLM**: a lexical embedder and an extractive "answer picker". Good for
  trying the UI and running unit tests; its answers are not real-model results.
* `ARAG_MODE=real` — local models (Ollama LLM, bge-small embeddings, NLI verifier).
* A variable set in the terminal (`$env:ARAG_MODE = "real"`) overrides `.env`. Every variable is listed in
  `.env.example` and defined in `src/adaptive_rag/config.py`.

### D5. Models (real mode) ✅

```powershell
ollama pull qwen3:4b-instruct-2507-q4_K_M      # ~2.5 GB, default model
ollama list                                    # must list the model
# optional, for the 8 GB profile (section G):
ollama pull qwen2.5:1.5b                       # ~1 GB
```

The embedding model (~65 MB → `data/runtime/model_cache/`) and the NLI model (~280 MB → Hugging Face cache in your
user folder) download automatically on the first real-mode start.

### D6. Datasets (only for benchmarks)

* `data/datasets/smoke_test/` is in the repository — nothing to do.
* The TriviaQA samples must be prepared once (internet; streams from Hugging Face, ~10–20 min each) ⚠️:

```powershell
python evaluation/prepare_triviaqa.py --n 100                                         # development set (100 q)
python evaluation/prepare_triviaqa.py --offset 100 --n 200 --out data/datasets/triviaqa_heldout_101_300 `
    --exclude-dataset data/datasets/triviaqa_unfiltered_sample                        # held-out test set
python evaluation/validate_dataset.py                                                 # ✅ check
```

Expected `validate_dataset.py` output on the reference machine ✅:

```
[OK ] smoke_test: 8 documents, 24/24 usable questions (18 answerable, 6 unanswerable)
[OK ] triviaqa_heldout_101_300: 978 documents, 199/199 usable questions (165 answerable, 34 unanswerable)
[OK ] triviaqa_unfiltered_sample: 475 documents, 100/100 usable questions (76 answerable, 24 unanswerable)
```

Hugging Face could change the mirror over time; if your counts differ, your results are not directly comparable.

### D7. Index initialisation — nothing to run

* Your **document index** (`data/runtime/kb/`) is created automatically when the backend starts and filled when you
  upload files.
* Each **benchmark index** (`data/runtime/benchmarks/<dataset>/`) is built automatically by the benchmark from the
  dataset's evidence documents (`corpus.jsonl`). The expected answers are **never** indexed or shown to the system;
  they are only used afterwards to score the answers.

---

## E. Starting the application

Use **two terminals**, both opened in the repository root.

**Terminal 1 — backend** ✅

```powershell
.\.venv\Scripts\Activate.ps1
$env:ARAG_MODE = "real"            # skip this line if .env already says ARAG_MODE=real; use "dev" for no models
uvicorn adaptive_rag.api.app:app --port 8000
```

The first real-mode start downloads models (a few minutes); later starts take ~10–30 s.

**Terminal 2 — frontend** ✅

```powershell
cd frontend
npm run dev                        # development server
# or a production build:
# npm run build
# npm run start
```

| What | URL |
|---|---|
| Web application | http://localhost:3000 |
| Interactive API documentation (Swagger) | http://127.0.0.1:8000/docs |
| Health (mode, model readiness, configuration) | http://127.0.0.1:8000/api/health |
| Readiness (200 when the LLM can answer, 503 otherwise) | http://127.0.0.1:8000/api/ready |

Check from PowerShell ✅:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/ready
(Invoke-RestMethod http://127.0.0.1:8000/api/health).config.question_routing     # True = routing on
```

**Stopping the servers:** press **Ctrl + C** in each terminal. If a server keeps a port busy (for example after
closing a terminal window) ✅:

```powershell
Get-NetTCPConnection -LocalPort 8000,3000 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

---

## F. Using the application

1. **Upload documents** — *Document Library* → **Choose files** (PDF with a text layer, DOCX, TXT, Markdown; max 20 MB
   each). Try the files in `data/sample_documents/`. Each file reports *indexed*, *duplicate* or *failed* with a reason.
   Scanned PDFs are rejected (no OCR). Your index is `data/runtime/kb/`; only uploaded files are searched.
2. **Ask a simple question** — *Ask Documents* (Multi-Agent RAG is selected by default), e.g.
   *"What is the fine for an overdue book?"* The answer header shows LLM calls, retrieval rounds and
   **route: simple**.
3. **Ask a complex question** — e.g. *"What is the difference between monocrystalline panels and LFP batteries?"*
   → **route: complex** (full workflow, up to 2 escalations). A question with fewer than 2 content words is routed
   **ambiguous**.
4. **See the route and every step** — **View agent trace**: the Planner shows `route: simple|complex|ambiguous`; each
   round shows the Evidence decision (e.g. `kept 2 of 5 (tau_q=…)` when filtering ran), the Verifier label and the
   Confidence decision (`accept`, `retry`, `abstain`) with its reason.
5. **Evidence and citations** — citation chips `[1]` link to passages in the *Evidence explorer* (file, page, chunk,
   retrieval scores and, after escalation, the judge score; **Show filtered-out passages** lists removed passages and
   why). Scores mean *relevant*, not *true*.
6. **Confidence and verification** — the badge says *Verified answer*, *Partially supported* (unresolved claims are
   listed) or *Abstained*. The Confidence card shows the heuristic score (0–1) and its four parts; it is **not** a
   calibrated probability. The Verifier card lists each claim with its label and evidence.
7. **Abstention** — if the documents do not contain the answer (try *"What is the WiFi password of the library?"*),
   the system says the evidence is insufficient instead of guessing.
8. **Compare vs Baseline** — runs the default Multi-Agent RAG and Standard RAG on the same question side by side.
9. **Evaluation page** — run benchmarks from the browser, follow progress, open earlier runs (also CLI runs), download
   **CSV** or **JSON**. Link a run directly with `http://localhost:3000/evaluation?run=<run_id>`.

---

## G. Hardware profiles

One variable: **`ARAG_HARDWARE_PROFILE`** = `8gb` | `16gb` | `32gb` (default `32gb`). A profile only fills settings
you did not set yourself.

| Profile | LLM (`ARAG_OLLAMA_MODEL`) | Context | NLI | Judge pool | Routing | Evidence |
|---|---|---|---|---|---|---|
| `32gb` (default) | `qwen3:4b-instruct-2507-q4_K_M` | 4096 | on | 10 | on | **all published benchmarks** |
| `16gb` | `qwen3:4b-instruct-2507-q4_K_M` | 3072 | on | 8 | on | **not run separately** (estimated: Ollama ~3.5 GB + backend 0.6–1.0 GB) |
| `8gb` | `qwen2.5:1.5b` (pull it first) | 2048 | off | 6 | on | **functional check only** (8 smoke questions, 0 failures; backend ~0.3 GB, model 1.1 GB); answer quality **not benchmarked** |

Switch profile (Terminal 1, then restart the backend):

```powershell
# Ctrl + C to stop the backend, then:
$env:ARAG_HARDWARE_PROFILE = "8gb"
uvicorn adaptive_rag.api.app:app --port 8000
```

To make it permanent, set `ARAG_HARDWARE_PROFILE=8gb` in `.env`. **Keep `ARAG_OLLAMA_MODEL` and `ARAG_OLLAMA_NUM_CTX`
commented out in `.env`** (as in `.env.example`); an uncommented value counts as your own setting and overrides the
profile.

Restore the default: `Remove-Item Env:ARAG_HARDWARE_PROFILE` (or set it to `32gb` in `.env`) and restart the backend.
CPU-only generation: `ARAG_OLLAMA_NUM_GPU=0` ⚠️ (supported by the code, not benchmarked; slower).

---

## H. Routing configuration

| Variable | Default | Meaning |
|---|---|---|
| `ARAG_QUESTION_ROUTING` | `true` | Planner routes questions; `false` = the evaluated v2 behaviour (every question may use 2 escalation rounds) |
| `ARAG_SIMPLE_ROUTE_RETRIES` | `1` | escalation rounds for simple questions (0–2) |
| `ARAG_MAX_RETRIES` | `2` | escalation rounds for complex/ambiguous questions (hard cap 2) |

```powershell
$env:ARAG_QUESTION_ROUTING = "false"     # disable routing for this terminal, then restart the backend
Remove-Item Env:ARAG_QUESTION_ROUTING    # back to the default (on)
```

* **Routes (rules, no LLM call):** *ambiguous* if fewer than 2 content words or the question starts with a pronoun and
  names no entity; *complex* if comparison, multi-part, sub-questions, or more than 20 words; otherwise *simple*.
* **Lightweight round (simple):** Planner → plain hybrid retrieval (top 5) → one draft → Verifier (lexical + NLI; a
  short answer also needs the LLM judge's confirmation) → Confidence.
* **Escalation:** when the draft says the evidence is insufficient, a claim is contradicted, or the confidence score is
  below 0.65 (`ARAG_ACCEPT_THRESHOLD`) — the Planner re-plans, the Evidence Agent judges and filters a larger pool,
  and a new draft is verified (the full four-agent workflow).
* **After the budget:** answer if the best attempt scores ≥ 0.40 (`ARAG_ABSTAIN_THRESHOLD`) and is at least partially
  supported, otherwise abstain.
* **Inspect:** the Agent Trace page, or `Invoke-RestMethod http://127.0.0.1:8000/api/runs/<run_id>` (field `route`,
  and the `trace` list).

---

## I. Running tests

From the repository root with the virtual environment active:

| Task | Command | Result on the reference machine |
|---|---|---|
| All fast tests (mocked models) | `pytest` | ✅ 156 passed, 4 skipped |
| One file | `pytest tests/test_routing.py` | ✅ 22 passed |
| Tests matching a name | `pytest -k short_answer` | ✅ 4 passed |
| Real-model integration tests | `$env:ARAG_RUN_REAL_MODEL_TESTS = "1"; pytest -m real_model` | ✅ 4 passed (needs Ollama + models) |
| Turn real tests off again | `Remove-Item Env:ARAG_RUN_REAL_MODEL_TESTS` | – |
| Python lint | `ruff check src tests evaluation` | ✅ no issues |
| Frontend lint | `cd frontend; npm run lint` | ✅ no issues |
| Frontend type check | `cd frontend; npx tsc --noEmit` | ✅ no errors |
| Frontend production build | `cd frontend; npm run build` | ✅ success |

**Mocked vs real:** all tests in `tests/` except `test_real_models.py` use a scripted mock LLM, a fake NLI model and
the dev-mode embedder, so they need no downloads. `test_real_models.py` uses fastembed, the NLI model and Ollama and is
skipped unless `ARAG_RUN_REAL_MODEL_TESTS=1`. Details: [docs/testing.md](docs/testing.md).

---

## J. Running benchmarks

| Dataset | Size | Role | Available |
|---|---|---|---|
| `smoke_test` | 24 questions (18 answerable, 6 not), 8 documents, synthetic | checks the pipeline — **not** a research benchmark | committed |
| `triviaqa_unfiltered_sample` | 100 questions (76 / 24), 475 documents | development set (all design decisions) | `prepare_triviaqa.py --n 100` |
| `triviaqa_heldout_101_300` | 199 questions (165 / 34), 978 documents | held-out **test** set | `prepare_triviaqa.py --offset 100 …` |

**System names** (`--methods`, comma-separated): `standard` (baseline), `adaptive` (current defaults = v2 + routing),
`adaptive_v2` (v2, routing off), `adaptive_routed` (v2 + routing, pinned), and the ablations `adaptive_no_retry`,
`adaptive_no_filter`, `adaptive_v1` (original pipeline).

```powershell
# smoke test, real models (~4-5 min)                                             ✅ (run 20261011-193700-e448b5)
python evaluation/run_benchmark.py --dataset smoke_test --mode real --methods standard,adaptive

# Standard RAG vs v2 vs v2 + routing on the held-out set, with consistency (~2 h)  ⚠️ (= Part D, run earlier)
python evaluation/run_benchmark.py --dataset triviaqa_heldout_101_300 --mode real `
    --methods standard,adaptive_v2,adaptive_routed --repeat 20

# baseline, v2 and ablations (~3 h)                                                 ⚠️ (= Part C, run earlier)
python evaluation/run_benchmark.py --dataset triviaqa_heldout_101_300 --mode real `
    --methods standard,adaptive_v2,adaptive_no_retry,adaptive_no_filter,adaptive_v1 --repeat 20

# quick sample of 50 questions (fixed seed)                                        ⚠️
python evaluation/run_benchmark.py --dataset triviaqa_heldout_101_300 --mode real --methods standard,adaptive --limit 50

# report: recompute metrics + exact McNemar tests from saved rows                  ✅
python evaluation/analyze_results.py <run_id>
python evaluation/analyze_results.py <run_id> --a standard --b adaptive_routed
```

* **Fair comparison:** put all systems in **one** run (`--methods a,b,c`) — they then answer the same questions in the
  same interleaved order on the same index and model. `--repeat N` re-runs the first N questions for consistency.
* **Where results go:** `evaluation/results/<run_id>/results.json` (configuration, models, git commit, per-question
  rows with per-attempt drafts, summaries, consistency) and `results.csv`. A new run always gets a new folder.
* **Keep earlier results safe:** never edit or delete folders in `evaluation/results/`; their checksums are listed in
  [evaluation/results/MANIFEST.md](evaluation/results/MANIFEST.md). For experiments you do not want there, redirect
  the output ✅:

```powershell
$env:ARAG_RESULTS_DIR = "$PWD\data\runtime\scratch_results"; $env:ARAG_RUNTIME_DIR = "$PWD\data\runtime\scratch"
python evaluation/run_benchmark.py --dataset smoke_test --mode dev --methods standard,adaptive --limit 6
Remove-Item Env:ARAG_RESULTS_DIR, Env:ARAG_RUNTIME_DIR
```

* **Do not tune on the held-out set.** Change settings using `triviaqa_unfiltered_sample` only; run the held-out set
  once per frozen version, otherwise its results stop being an independent test.
* Stop the backend before a CLI benchmark of the same dataset (one process per Qdrant index folder) and avoid using the
  app during a benchmark (it shares the GPU and changes latency).

---

## K. Understanding the results

| Metric | Meaning |
|---|---|
| **Accuracy** | share of answerable questions answered **and** containing a correct answer alias (abstaining counts as wrong) |
| **Evidence relevance** | *precision*: share of passages given to the generator that contain the answer; *recall*: at least one does |
| **Faithfulness** | *grounded answers*: ≥ 80 % of the answer's words appear in its cited passages (independent of the verifier) |
| **Citation correctness** | an answered question cites at least one document that contains the answer |
| **Unsupported-answer rate** | answered but not correct (wrong answers + answers to unanswerable questions), over all answered |
| **Wrong refusals** | answerable questions on which the system abstained |
| **Correct abstention** | unanswerable questions on which the system abstained |
| **Latency** | wall-clock time per question (mean, median, first-question "cold" and later "warm") |
| **LLM calls** | drafts + judge calls + query rewrites + verifier judge calls per question |
| **Statistical significance** | exact two-sided **McNemar** test on the questions where only one system was correct; p < 0.05 = unlikely to be chance |

Documented held-out results (199 questions, 165 answerable; full tables in [docs/results.md](docs/results.md)):

| Run | Standard RAG | Multi-Agent v2 | v2 + routing | Paired test vs Standard |
|---|---|---|---|---|
| `20261011-122005-0a2d3c` (Part C) | 67.9 % | 73.9 % | – | v2: 13 vs 3, **p = 0.021** |
| `20261011-171730-9902e7` (Part D) | 69.1 % | 72.7 % | 73.3 % | v2: 9 vs 3, p = 0.146 · routing: 9 vs 2, p = 0.065 |

In Part D, routing used **7.5 instead of 9.5 LLM calls** and **12.1 s instead of 16.4 s** per question with the same
accuracy as v2. Standard RAG needs 1 call and ~3.6 s. The Multi-Agent systems were ahead in both runs, but the
difference was significant in only one, so it is **promising, not conclusive**. Results differ between runs because
Ollama generation is not fully deterministic (Standard RAG gave the same answer in 17 of 20 repeats). One dataset,
one 4B model — do not generalise.

---

## L. Troubleshooting

| Problem | What you see | Fix |
|---|---|---|
| Venv will not activate | "running scripts is disabled on this system" | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then `.\.venv\Scripts\Activate.ps1` |
| Missing dependency | `ModuleNotFoundError` | activate the venv, then `pip install -e ".[nli,dev]"` (and `".[datasets]"` for TriviaQA preparation) |
| LLM not ready | `/api/health` → generator "Ollama not reachable at http://127.0.0.1:11434" | start Ollama (`ollama serve` in a new terminal) |
| Model missing | "Ollama is running but '…' is not pulled (run: ollama pull …)" | run the shown `ollama pull` command; the name must equal `ARAG_OLLAMA_MODEL` |
| Judge not working | trace note "judge unavailable (Ollama did not return token log-probabilities …)" | update Ollama (log-probabilities are required for the evidence judge); the system falls back to plain retrieval order |
| Model download fails | errors on first real-mode start | check the internet connection and retry; models are cached after the first success |
| Not enough RAM/VRAM, or very slow | slow answers, high memory | close other apps; use `ARAG_HARDWARE_PROFILE=16gb` or `8gb`; `ARAG_OLLAMA_NUM_CTX=2048`; escalated questions normally take 20–50 s on a 4 GB GPU |
| Port in use | uvicorn: address already in use | stop the old process (section E) or use `--port 8001` and set `NEXT_PUBLIC_API_URL=http://127.0.0.1:8001` in `frontend/.env.local` |
| Frontend cannot reach backend | "Cannot reach the backend at http://127.0.0.1:8000. Is it running?" | start Terminal 1; check `frontend/.env.local`; a frontend on another port must be added to `ARAG_CORS_ORIGINS` |
| Dataset missing | `validate_dataset.py` shows `[BAD]` or the dataset is absent | run the `prepare_triviaqa.py` commands (section D6) |
| Invalid setting | backend refuses to start with a Pydantic `ValidationError` naming the field (e.g. `ARAG_MAX_RETRIES=3`) | fix or remove the value in `.env` / the terminal |
| Index locked | "Storage folder … is already accessed by another instance of Qdrant client." | another backend or benchmark uses the same index folder: stop it, or point one of them to another `ARAG_RUNTIME_DIR` |
| Start over with your documents | wrong or old documents | stop the backend and delete the folder `data\runtime\kb` (it is recreated empty) |
untime\kb` (it is recreated empty) |
| A benchmark row failed | `Failed cases` > 0, `error` column in `results.csv` | read the error text (often Ollama stopped or a timeout, `ARAG_LLM_TIMEOUT_S`, default 180 s); failed rows are counted, never hidden |

---

## M. Reproducibility and safety

```powershell
git rev-parse --short HEAD            # current commit
git status                            # must be clean before a benchmark you want to report
git tag -n1                           # preserved versions (v1-baseline, v2-heldout-eval, v2-reviewed,
                                      #   v3-routing-heldout-eval, results-2026-10-11)
git check-ignore -v .env              # .env, data/runtime/, datasets and results are ignored
Get-FileHash -Algorithm SHA256 evaluation\results\<run_id>\results.json   # compare with MANIFEST.md
```

* Every `results.json` stores the full configuration, variant settings, model names and the git commit at start.
* `.env`, `data/runtime/` (your documents, indexes, saved runs), the TriviaQA folders, `evaluation/results/<run>`
  folders and PDFs in the root are git-ignored — do not force-add them (`git add -f`). Never commit API keys or
  private documents.
* Do not delete or move tags; do not edit result folders. To compare a new version, make a new run.
* Held-out results must not guide changes: every change you test against the held-out set and keep makes that set a
  development set, and the next result would overstate the improvement.

---

## N. Command reference

| Task | Command (PowerShell, repository root unless noted) |
|---|---|
| Initial setup | `py -3.12 -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install torch --index-url https://download.pytorch.org/whl/cpu; pip install -e ".[nli,dev]"; Copy-Item .env.example .env; ollama pull qwen3:4b-instruct-2507-q4_K_M` then `cd frontend; npm install; Copy-Item .env.example .env.local` |
| Backend startup | `.\.venv\Scripts\Activate.ps1; $env:ARAG_MODE="real"; uvicorn adaptive_rag.api.app:app --port 8000` |
| Frontend startup | `cd frontend; npm run dev` (open http://localhost:3000) |
| Health check | `Invoke-RestMethod http://127.0.0.1:8000/api/ready` |
| Tests | `pytest` |
| Real-model tests | `$env:ARAG_RUN_REAL_MODEL_TESTS="1"; pytest -m real_model` |
| Lint | `ruff check src tests evaluation` and `cd frontend; npm run lint` |
| Type checking | `cd frontend; npx tsc --noEmit` |
| Production build | `cd frontend; npm run build; npm run start` |
| Smoke test | `python evaluation/run_benchmark.py --dataset smoke_test --mode real --methods standard,adaptive` |
| Full benchmark | `python evaluation/run_benchmark.py --dataset triviaqa_heldout_101_300 --mode real --methods standard,adaptive_v2,adaptive_routed --repeat 20` |
| Results inspection | `python evaluation/analyze_results.py <run_id>` |
| Disable routing | `$env:ARAG_QUESTION_ROUTING="false"` and restart the backend |
| Server shutdown | **Ctrl + C** in each terminal, or `Get-NetTCPConnection -LocalPort 8000,3000 -State Listen -ErrorAction SilentlyContinue \| ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }` |
