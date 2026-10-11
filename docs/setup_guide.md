# Setup Guide (Windows 11, PowerShell)

Run every command from the project folder unless stated otherwise.

## 1. Prerequisites

| Software | Version | Check | Get it |
|---|---|---|---|
| Python | 3.12 (3.11 / 3.13 also work) | `py -3.12 --version` | https://www.python.org/downloads/ |
| Node.js | 20 or newer (tested 22) | `node --version` | https://nodejs.org/ |
| Git | any recent | `git --version` | https://git-scm.com/ |
| Ollama | any recent (tested 0.31) | `ollama --version` | https://ollama.com/download (only for real mode) |
| NVIDIA driver | optional | `nvidia-smi` | GPU speeds up the LLM; not required |

If PowerShell blocks `Activate.ps1`, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## 2. Get the code

```powershell
git clone https://github.com/rabbunimerugumala/RAG-Based-Adaptive-Framework-for-Multi-Agent-Systems.git
cd RAG-Based-Adaptive-Framework-for-Multi-Agent-Systems
```

## 3. Python backend

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# Option A - full install (real mode with the NLI verifier). CPU build of PyTorch first (~200 MB download):
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[nli,dev]"

# Option B - minimal install (dev mode, or real mode with lexical-only verification):
pip install -e ".[dev]"

# Optional - only to prepare the TriviaQA sample:
pip install -e ".[datasets]"
```

## 4. Configuration

```powershell
Copy-Item .env.example .env
notepad .env
```

The most important setting is `ARAG_MODE`:

- `ARAG_MODE=dev` — no downloads, no LLM. For trying the UI and running unit tests. Results are **not** experimental results.
- `ARAG_MODE=real` — local models.

Any `ARAG_*` variable can also be set for one terminal session, e.g. `$env:ARAG_MODE = "real"`.
All variables are documented in `.env.example` and `src/adaptive_rag/config.py`.

## 5. Models (real mode)

```powershell
ollama pull qwen3:4b-instruct-2507-q4_K_M      # ~2.5 GB, once
ollama list                                    # should list the model
```

Ollama runs as a background service after installation. If `/api/health` says "Ollama not reachable", start it
with `ollama serve` in a separate terminal.

The embedding model (~65 MB, into `data/runtime/model_cache/`) and the NLI model (~280 MB, into the Hugging Face cache
in your user folder) download automatically on the first real-mode start. The first start therefore takes a few
minutes; later starts take about 10–30 seconds.

Low-memory alternatives: `ARAG_OLLAMA_MODEL=qwen2.5:1.5b` (pull it first), `ARAG_OLLAMA_NUM_GPU=0` for CPU-only,
or `ARAG_NLI_BACKEND=none` to skip the NLI model.

## 5b. Hardware profiles (8 / 16 / 32 GB RAM)

Set one variable; it only fills settings you did not set yourself.

| `ARAG_HARDWARE_PROFILE` | LLM | Context | NLI verifier | Judge pool | Question routing | Status |
|---|---|---|---|---|---|---|
| `32gb` (default) | qwen3:4b-instruct-2507 (Q4) | 4096 | on | 10 | off | used for all reported benchmarks |
| `16gb` | qwen3:4b-instruct-2507 (Q4) | 3072 | on | 8 | on | not separately benchmarked; estimated from the 32 GB measurements (Ollama ~3.5 GB, backend 0.6–1.0 GB) |
| `8gb` | qwen2.5:1.5b (~1 GB download, `ollama pull qwen2.5:1.5b`) | 2048 | off (lexical + judge checks) | 6 | on | functional check only: backend ~0.3 GB, model 1.1 GB; quality not benchmarked |

```powershell
$env:ARAG_HARDWARE_PROFILE = "8gb"; ollama pull qwen2.5:1.5b
uvicorn adaptive_rag.api.app:app --port 8000
```

On an 8 GB laptop close other heavy applications; the smaller model answers less accurately than the 4B model.

## 6. Frontend

```powershell
cd frontend
npm install
Copy-Item .env.example .env.local      # sets NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
cd ..
```

## 7. Run

Terminal 1 (backend):

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn adaptive_rag.api.app:app --port 8000
```

Check http://127.0.0.1:8000/api/health and the interactive API docs at http://127.0.0.1:8000/docs.

Terminal 2 (frontend):

```powershell
cd frontend
npm run dev                 # development server, http://localhost:3000
# or a production build:
npm run build; npm run start
```

## 8. First use

1. Open http://localhost:3000. The sidebar shows "Backend online" and the mode.
2. **Document Library** → *Choose files* → select files from `data/sample_documents/`.
3. **Ask Documents** → e.g. "How many books can an undergraduate student borrow and for how long?"
4. Click **View agent trace** to see each agent's work.

## 9. Tests and checks

```powershell
pytest                                                         # fast suite, mocked models
$env:ARAG_RUN_REAL_MODEL_TESTS = "1"; pytest -m real_model     # needs Ollama + models
Remove-Item Env:ARAG_RUN_REAL_MODEL_TESTS
ruff check src tests evaluation
cd frontend; npm run lint; npm run build; cd ..
```

## 10. Benchmarks

```powershell
python evaluation/validate_dataset.py                                   # lists datasets and usable questions

# quick smoke test (24 synthetic questions, ~5 min in real mode)
python evaluation/run_benchmark.py --dataset smoke_test --mode real

# TriviaQA samples (need the [datasets] extra and internet once)
python evaluation/prepare_triviaqa.py --n 100                                                   # development set
python evaluation/prepare_triviaqa.py --offset 100 --n 200 --out data/datasets/triviaqa_heldout_101_300 `
    --exclude-dataset data/datasets/triviaqa_unfiltered_sample                                  # held-out test

# the full held-out comparison: baseline, proposed system and ablations (~3 h on the reference laptop)
python evaluation/run_benchmark.py --dataset triviaqa_heldout_101_300 --mode real `
    --methods standard,adaptive,adaptive_no_retry,adaptive_no_filter,adaptive_v1 --repeat 20

# a quicker comparison of only the baseline and the proposed system on 50 questions
python evaluation/run_benchmark.py --dataset triviaqa_heldout_101_300 --mode real --methods standard,adaptive --limit 50

# recompute metrics and paired McNemar tests from saved per-question rows
python evaluation/analyze_results.py <run_id>
```

Results are written to `evaluation/results/<run_id>/results.json` (configuration, variant settings, model names,
git commit, per-question rows with per-attempt drafts, summaries) and `results.csv`. Runs can also be started and
downloaded on the **Evaluation** page.

> Qdrant local mode allows one process per index folder. Do not run the CLI benchmark on a dataset while the API
> server is running an evaluation of the same dataset. Avoid using the app during a benchmark: it shares the GPU and
> changes the measured latency.

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| UI says "Backend offline" | Start the backend (step 7); check `NEXT_PUBLIC_API_URL` in `frontend/.env.local` |
| "Generator not ready" | `ollama serve`, then `ollama pull <model>`; the name must match `ARAG_OLLAMA_MODEL` |
| `Storage folder ... is already accessed by another instance` | Another backend/benchmark process holds the Qdrant folder; stop it |
| Very slow answers | Model partly on CPU (normal on 4 GB GPUs); close other GPU apps, reduce `ARAG_OLLAMA_NUM_CTX` to 2048, or use a smaller model |
| "No extractable text found" for a PDF | It is a scanned/image PDF; OCR is not supported |
| Symlink warnings from Hugging Face on Windows | Harmless; enabling Windows Developer Mode removes them |
| Port already in use | `uvicorn ... --port 8001` and update `NEXT_PUBLIC_API_URL`; or `npm run dev -- -p 3001` (then add the origin to `ARAG_CORS_ORIGINS`) |

## 12. Resetting

Delete `data/runtime/` to remove all uploaded documents, indexes and saved runs (recreated on next start).
Delete `evaluation/results/<run_id>/` to remove a benchmark result.
