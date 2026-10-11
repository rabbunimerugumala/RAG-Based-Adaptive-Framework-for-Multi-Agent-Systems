# Technology Stack

All components are free and open source and run locally. Sizes below were measured on the development laptop
(Windows 11, 32 GB RAM, RTX 3050 Laptop GPU 4 GB) unless stated otherwise.

## 1. Backend (Python 3.12)

| Technology | Role in this project | Why chosen |
|---|---|---|
| **FastAPI** | REST API + automatic OpenAPI docs at `/docs` | Simple, typed, fast; Pydantic integration |
| **Uvicorn** | ASGI server that runs FastAPI | Standard FastAPI server |
| **Pydantic v2 / pydantic-settings** | Data schemas, request validation, configuration from env vars | One typed source of truth for data and settings |
| **qdrant-client (local mode)** | Vector database for dense retrieval, stored in a folder | Real vector DB API without running a server or Docker |
| **BM25 (own code)** | Keyword retrieval | ~40 lines, transparent, no dependency |
| **fastembed (ONNX Runtime)** | Runs the embedding model (and optional reranker) on CPU | Small, no PyTorch needed, fast on CPU |
| **pypdf** | PDF text extraction (text layer, per page) | Pure Python, permissive licence |
| **python-docx** | DOCX paragraph and table extraction | Standard library for Word files |
| **httpx** | HTTP client used to call Ollama | Timeouts, simple API |
| **PyTorch (CPU) + transformers** | Runs the NLI verifier model | Standard way to run Hugging Face models (optional extra `[nli]`) |
| **pytest** | Testing | De-facto Python test framework |
| **ruff** | Linting | Fast, one tool for lint + import order |
| **datasets** (optional) | Streams the TriviaQA sample from Hugging Face | Avoids downloading the 3 GB archive |

No LangChain, LlamaIndex or LangGraph: the agent loop is short explicit Python, which is easier to explain.

## 2. Models

| Role | Model | Licence | Size on disk | Runs on | Notes |
|---|---|---|---|---|---|
| Generator **and evidence judge** (one shared instance) | `qwen3:4b-instruct-2507-q4_K_M` via **Ollama** | Apache-2.0 | 2.5 GB download | GPU + CPU | 4-bit quantised. With a 4096-token context Ollama reported 3.5 GB in memory, split ~67 % GPU / 33 % CPU on a 4 GB card. The judge reads Ollama's token log-probabilities (`logprobs`), available in the tested Ollama 0.31 |
| Small LLM for the `8gb` profile (optional) | `qwen2.5:1.5b` via Ollama | Apache-2.0 | ~1 GB download | GPU or CPU | functional check only: model 1.1 GB in memory, backend ~0.3 GB; answer quality not benchmarked |
| Embeddings | `BAAI/bge-small-en-v1.5` (fastembed ONNX) | MIT | ~65 MB | CPU | 384-dimensional vectors |
| Verifier NLI | `cross-encoder/nli-deberta-v3-xsmall` | Apache-2.0 | ~280 MB | CPU | Outputs entailment / contradiction / neutral |
| Reranker (optional, off by default) | `Xenova/ms-marco-MiniLM-L-6-v2` (fastembed) | Apache-2.0 | ~80 MB (approx.) | CPU | Enable with `ARAG_RERANKER=cross-encoder` |

**Why Qwen3-4B-Instruct-2507?** Apache-2.0 licence, strong instruction following and JSON output for its size, no
"thinking" mode (fast, predictable output), and a 4-bit build small enough for a 4 GB GPU with partial CPU offload.
Any other Ollama model can be used by changing `ARAG_OLLAMA_MODEL` (e.g. `qwen2.5:1.5b` for a slower laptop).

**CPU-only fallback:** set `ARAG_OLLAMA_NUM_GPU=0`; Ollama then runs fully on CPU (slower, roughly several seconds
to tens of seconds per answer depending on the CPU). Embeddings and NLI are always CPU.

## 3. Frontend

| Technology | Role | Why chosen |
|---|---|---|
| **Next.js 16 (App Router)** | Pages, routing, production build | Recommended React framework; static prerendering of the app shell |
| **React 19** | UI components | Industry standard |
| **TypeScript** | Types for all API data (`lib/types.ts` mirrors `schemas.py`) | Catches API mismatches at build time |
| **Tailwind CSS 4** | Styling with semantic design tokens (`app/globals.css`) | One token set drives both themes |
| **next-themes** | Light/dark theme: system default, toggle, saved preference, no flash on load | Solves the theme-flash problem correctly |
| **Recharts** | Evaluation bar chart | Simple React charting; colours use CSS variables so both themes work |
| **lucide-react** | Icons | Clean, consistent icon set |
| clsx + tailwind-merge | Conditional class names | Tiny helpers |

Instead of installing a full component library, a few small primitives (`components/ui.tsx`: Card, Button, Badge,
Notice, Meter, ...) were written in the shadcn/ui style. This keeps the frontend small and easy to read.

## 4. Infrastructure

| Item | Choice |
|---|---|
| OS | Windows 11 (PowerShell commands in all docs) |
| LLM server | Ollama (native Windows installer) |
| Containers | none (no Docker) |
| Database | Qdrant local mode folder + JSON registry |
| Auth | none: local prototype |

## 5. Approximate resource requirements

| Resource | Development mode | Real mode |
|---|---|---|
| Disk (code + venv + node_modules) | ~1.9 GB measured (Python venv 1.3 GB incl. CPU PyTorch for the NLI extra; frontend node_modules 0.56 GB) | same |
| Disk (models) | 0 | ~2.9 GB (LLM 2.5 GB + NLI 0.28 GB + embeddings 0.07 GB) |
| RAM | < 1 GB | backend process 0.6–1.0 GB observed + Ollama (part of the 3.5 GB model on CPU); 8 GB minimum, 16 GB+ comfortable |
| GPU | none | optional; 4 GB VRAM used for most model layers. Without a GPU everything still works on CPU, slower |
| Latency (observed, real mode) | milliseconds | Standard ~2.5–5 s/question; Adaptive ~4–20 s (more when it retries) |
