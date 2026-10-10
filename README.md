# RAG-Based Adaptive Framework for Multi-Agent Systems

An evidence-grounded question-answering system that compares two approaches on your own documents:

| | Standard RAG (baseline) | Adaptive Multi-Agent RAG (proposed) |
|---|---|---|
| Flow | retrieve → generate → check citations | **Planner → Evidence → draft → Verifier → Confidence** → accept / retry / abstain |
| Checks the answer? | no (only shown as post-hoc analysis) | yes, claim by claim against the evidence |
| Can retry retrieval? | no | yes, up to **2** targeted retries |
| Can refuse to answer? | only if the LLM itself says so | yes, abstains when support stays weak |

Everything runs **locally on a Windows laptop** (tested: 32 GB RAM, RTX 3050 Laptop 4 GB). No paid API keys, no Docker, no cloud.

> **Honesty note.** This system *reduces the risk* of unsupported answers; it does **not** eliminate hallucinations.
> Its confidence score is a transparent heuristic, **not** a calibrated probability. All numbers in the docs come from
> real runs that are listed with their run IDs in [docs/results.md](docs/results.md).

---

## Contents

| Document | What you learn |
|---|---|
| [docs/how_it_works.md](docs/how_it_works.md) | **Start here.** Plain-language walk-through of RAG and of one question flowing through the 4 agents |
| [docs/architecture.md](docs/architecture.md) | System diagram, modules, the four agents, confidence formula, design decisions |
| [docs/tech_stack.md](docs/tech_stack.md) | Every technology and model used, why it was chosen, and its size/licence |
| [docs/setup_guide.md](docs/setup_guide.md) | Detailed Windows setup, model downloads, configuration, troubleshooting |
| [docs/user_guide.md](docs/user_guide.md) | How to use each page of the web dashboard |
| [docs/api_reference.md](docs/api_reference.md) | All REST endpoints with examples |
| [docs/evaluation_protocol.md](docs/evaluation_protocol.md) | Datasets, metric definitions, how a fair comparison is run |
| [docs/results.md](docs/results.md) | The actual benchmark results obtained, with interpretation |
| [docs/testing.md](docs/testing.md) | Test suite: what is mocked, what uses real models, how to run |
| [docs/limitations_and_future_work.md](docs/limitations_and_future_work.md) | Known limitations and possible extensions |
| [docs/faq.md](docs/faq.md) | Frequently asked questions about the design and results, plus a demo script |

---

## Quick start (Windows PowerShell)

Prerequisites: **Python 3.12**, **Node.js 20+**, **Git**, and **Ollama** (https://ollama.com/download) for real mode.

```powershell
git clone https://github.com/rabbunimerugumala/RAG-Based-Adaptive-Framework-for-Multi-Agent-Systems.git
cd RAG-Based-Adaptive-Framework-for-Multi-Agent-Systems

# 1. Python backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU torch for the NLI verifier
pip install -e ".[nli,dev]"
Copy-Item .env.example .env

# 2. Local LLM (real mode only, ~2.5 GB download)
ollama pull qwen3:4b-instruct-2507-q4_K_M

# 3. Frontend
cd frontend
npm install
Copy-Item .env.example .env.local
cd ..
```

Run it (two terminals):

```powershell
# Terminal 1 - backend (http://127.0.0.1:8000, API docs at /docs)
.\.venv\Scripts\Activate.ps1
$env:ARAG_MODE = "real"          # or "dev" for the no-download development mode
uvicorn adaptive_rag.api.app:app --port 8000

# Terminal 2 - frontend (http://localhost:3000)
cd frontend
npm run dev
```

Then open **http://localhost:3000**, go to **Document Library**, upload files from `data/sample_documents/`, and ask
questions on **Ask Documents**. The first real-mode start downloads the embedding (~65 MB) and NLI (~280 MB) models.

Tests and benchmark:

```powershell
pytest                                                          # 98 fast tests, mocked models, no downloads
$env:ARAG_RUN_REAL_MODEL_TESTS = "1"; pytest -m real_model      # 4 integration tests with the real models
python evaluation/run_benchmark.py --dataset smoke_test --mode real
cd frontend; npm run lint; npm run build
```

Full details: [docs/setup_guide.md](docs/setup_guide.md).

---

## Two run modes

| Mode | Embeddings | Answer generator | Verifier | Use it for |
|---|---|---|---|---|
| `dev` | hashing embedder (lexical) | **extractive sentence picker, no LLM** | lexical checks | development, CI, unit tests. **Never report its numbers as results** |
| `real` | `BAAI/bge-small-en-v1.5` (fastembed, CPU) | `qwen3:4b-instruct-2507` via Ollama (GPU + CPU) | lexical + NLI `nli-deberta-v3-xsmall` (CPU) | demos and experiments |

The UI shows a **DEV MODE** banner whenever the backend runs in dev mode, and every saved result records its mode.

---

## Features

- Upload **PDF (text layer), DOCX, TXT, Markdown**: type/size validation, safe filenames, duplicate detection (SHA-256), page-aware chunking, listing and deletion.
- **Hybrid retrieval**: Qdrant (local mode) dense search + BM25 keyword search, merged by **reciprocal-rank fusion**; optional cross-encoder reranker.
- **Four agents with distinct jobs** sharing **one** LLM instance (see [architecture](docs/architecture.md)).
- **Real execution trace** of every run (inputs, outputs, timing, decisions, retries) shown in the UI.
- **Abstention** when evidence is missing; clear instructions when the knowledge base is empty.
- **Compare** both methods on the same question and knowledge base.
- **Evaluation** on a synthetic smoke-test set and on a locally prepared **TriviaQA-unfiltered** sample; CSV/JSON export with the full configuration.
- Professional **light/dark** dashboard (system theme by default, toggle, saved preference, no theme flash), responsive down to phone width.

---

## Project structure

```
├── src/adaptive_rag/           Python backend package
│   ├── agents/                 planner.py · evidence.py · verifier.py · confidence.py
│   ├── api/app.py              FastAPI app (all endpoints)
│   ├── evaluation/             datasets.py · metrics.py · runner.py
│   ├── generation/             llm.py (Ollama + dev generator) · nli.py (verifier model)
│   ├── ingestion/              extract.py (PDF/DOCX/TXT/MD) · chunking.py
│   ├── retrieval/              embeddings.py · bm25.py · fusion.py · hybrid.py · store.py (Qdrant)
│   ├── config.py               all settings (ARAG_* environment variables)
│   ├── models.py               model-loading layer + readiness status
│   ├── pipeline.py             Standard RAG and Adaptive RAG orchestration
│   ├── schemas.py              Pydantic data models
│   └── text.py                 tokenising, keywords, sentence splitting
├── frontend/                   Next.js 16 + React 19 + Tailwind 4 dashboard
│   ├── app/                    pages: / · ask · trace · documents · compare · evaluation
│   ├── components/             app shell, UI primitives, result views
│   └── lib/                    typed API client + types
├── data/
│   ├── datasets/smoke_test/    synthetic 24-question test set (committed)
│   └── sample_documents/       demo documents to upload
├── evaluation/                 run_benchmark.py · prepare_triviaqa.py · validate_dataset.py
├── tests/                      pytest suite (mocked) + opt-in real-model tests
├── docs/                       documentation (see table above)
├── pyproject.toml · .env.example · LICENSE
```

Generated data (indexes, uploads, saved runs, model cache, TriviaQA sample, result files) lives in git-ignored folders and is re-created by the commands above.

---

## Results (measured, real models)

| Dataset | Metric | Standard RAG | Adaptive RAG |
|---|---|---|---|
| Synthetic smoke test (24 q) | accuracy / correct abstention | 100 % / 100 % | 100 % / 100 % |
| TriviaQA-unfiltered sample (100 q) | accuracy (answerable, n = 76) | **78.9 %** | 68.4 % |
| | correct abstention (unanswerable, n = 24) | 54.2 % | 54.2 % |
| | wrong abstention (answerable) | 14.5 % | 23.7 % |
| | mean latency / LLM calls | 4.4 s / 1.0 | 11.5 s / 2.46 |

**Honest finding:** on the TriviaQA sample the adaptive pipeline was more cautious, slower and *less* accurate. The
failure analysis traces most lost answers to the Verifier's NLI step wrongly marking short correct answers
(e.g. "Niger") as contradicted, which then triggers abstention. The difference is not statistically significant
at n = 100 (sign test p ≈ 0.08). Details, paired analysis and proposed fixes: [docs/results.md](docs/results.md).

## Licence and credits

MIT licence (see `LICENSE`). Related work that inspired this project: *MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented
Generation* (ACL 2025). This project uses a different, four-agent design and does not reproduce MAIN-RAG.
TriviaQA: Joshi et al., ACL 2017; the dataset is downloaded locally by a script and is not redistributed here.
