# Architecture

## 1. System overview

```
┌──────────────────────────── Browser (http://localhost:3000) ─────────────────────────────┐
│  Next.js dashboard: Overview · Ask · Agent Trace · Documents · Compare · Evaluation      │
└───────────────────────────────────────┬──────────────────────────────────────────────────┘
                                        │ JSON over HTTP (fetch / XHR upload)
┌───────────────────────────────────────▼──────────────────────────────────────────────────┐
│ FastAPI  (src/adaptive_rag/api/app.py)            http://127.0.0.1:8000  · /docs          │
│   RunStore (saved traces)   EvaluationManager (background benchmark thread)              │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│ RAGSystem  (pipeline.py)                                                                 │
│   run_standard():  hybrid retrieval → generator → citation check (+ post-hoc verifier)   │
│   run_adaptive():  Planner → Evidence → generator → Verifier → Confidence → loop/stop    │
├───────────────┬───────────────────┬────────────────────┬─────────────────────────────────┤
│ agents/       │ retrieval/        │ generation/        │ ingestion/                      │
│  planner      │  store (Qdrant +  │  llm  (Ollama or   │  extract (PDF/DOCX/TXT/MD)      │
│  evidence     │   registry)       │   dev extractive)  │  chunking                       │
│  verifier     │  bm25, fusion,    │  nli  (DeBERTa     │                                 │
│  confidence   │  hybrid, embed.   │   cross-encoder)   │                                 │
└───────────────┴─────────┬─────────┴─────────┬──────────┴─────────────────────────────────┘
                          │                   │
        data/runtime/kb/qdrant (vectors)   Ollama server :11434 (qwen3:4b, GPU+CPU)
```

All components run in **one Python process** plus the Ollama server. There are no microservices, queues or databases
to install.

## 2. Modules

| Module | Responsibility |
|---|---|
| `config.py` | One `Settings` class (pydantic-settings). Reads `ARAG_*` env vars / `.env`. Validates values (e.g. overlap < chunk size, max 2 retries). |
| `schemas.py` | Pydantic models shared by pipeline, API and evaluation: `Evidence`, `RetrievalPlan`, `VerificationResult`, `ConfidenceAssessment`, `TraceStep`, `QueryResponse`, ... |
| `text.py` | Tokenising, stop-words, light stemming, keywords, focus terms, sentence splitting, answer normalisation. |
| `ingestion/extract.py` | File-type validation, safe filenames, text extraction, clear errors (no OCR, encrypted PDF, corrupt file). |
| `ingestion/chunking.py` | Character chunking with overlap, preferring sentence/paragraph boundaries; never crosses pages. |
| `retrieval/embeddings.py` | `FastEmbedEmbedder` (real) and `HashEmbedder` (dev mode). |
| `retrieval/store.py` | `KnowledgeBase`: Qdrant local collection + document registry + original files; add / delete / re-index if the embedder or chunk settings change. |
| `retrieval/bm25.py` | Okapi BM25 written from scratch (~40 lines) so it can be explained. |
| `retrieval/fusion.py` | Reciprocal-rank fusion. |
| `retrieval/hybrid.py` | Dense + BM25 for every query → RRF → optional rerank → `Evidence` list. Shared by both methods. |
| `generation/llm.py` | `OllamaLLM` (JSON-schema structured output, temperature 0, seed 42) and the dev `ExtractiveGenerator`. |
| `generation/nli.py` | Optional NLI model; label indices read from the model config. |
| `models.py` | Builds **one** embedder, **one** generator, optional NLI and reranker; reports readiness. |
| `agents/*.py` | The four agents (below). |
| `pipeline.py` | Orchestration with explicit control flow and full trace recording. |
| `evaluation/*.py` | Dataset loading/validation, metric functions, benchmark runner, result files. |
| `api/app.py` | FastAPI endpoints, run storage, background evaluation jobs. |

## 3. The four agents

### Planner Agent (`agents/planner.py`)
- **Input**: the question (and, on a retry, the previous plan + the Confidence Agent's retry strategy + verifier output).
- **Does**: detects small talk (no retrieval); classifies the query (comparison, multi-part, numeric, definition,
  procedural, factoid); extracts keywords and focus terms; decomposes comparison/multi-part questions into
  sub-questions; chooses `top_k`/`candidate_k`; on retry builds new queries (unsupported-claim queries, uncovered
  keywords, relaxed keyword pairs, optional LLM reformulations).
- **Output**: `RetrievalPlan` with a short explanation.
- **LLM use**: only during a retry, one call for query reformulation (deterministic rules do the rest).

### Evidence Agent (`agents/evidence.py`)
- **Input**: `RetrievalPlan`.
- **Does**: for every search query, dense (Qdrant cosine) and BM25 rankings; fuses all lists with RRF; optional
  cross-encoder rerank; attaches filename, page, chunk index, and all scores.
- **Output**: numbered `Evidence` passages + a retrieval summary. Scores are **relevance**, never proof.

### Verifier Agent (`agents/verifier.py`)
- **Input**: question, draft answer, evidence.
- **Does**: splits the answer into atomic claims; per claim computes lexical support (with a strict number check)
  and, if available, NLI entailment/contradiction over 3-sentence windows of the cited and best passages (and the
  cited passages combined); converts very short answers to statements for NLI ("Who founded X?" + "Edda" →
  "Edda founded X"); labels the claim; computes support score and citation consistency.
- **Output**: `VerificationResult` (supported / partially_supported / unsupported / contradicted / not_applicable).
- **Never** asks the generating LLM whether its own answer is right. With NLI available, word overlap alone can
  never produce "supported".

### Confidence Agent (`agents/confidence.py`)
- **Input**: plan, evidence, draft, verification, retries left.
- **Does**: computes four components, the weighted score, and the decision with a written reason and retry strategy.
- **Output**: `ConfidenceAssessment`.

```
C = (w_r·relevance + w_c·coverage + w_v·verification + w_k·consistency) / (sum of weights of AVAILABLE components)
default weights: 0.25, 0.25, 0.35, 0.15     accept ≥ 0.65     abstain < 0.40     max retries = 2
```

Missing components (e.g. no verification because the LLM declined) are **listed** and excluded from the mean, and an
answer without a verification result can never be accepted. The score is a heuristic until calibrated on labelled data.

Decision order: no evidence → retry/abstain · generator said insufficient → retry/abstain · contradiction →
retry/abstain · score ≥ accept and verification ok → **accept** · retries left → **retry** (broaden if relevance or
coverage < 0.5, else targeted) · otherwise pick the best attempt: accept if ≥ abstain threshold and at least partially
supported, else **abstain**.

## 4. Fair comparison by design

Both methods share the same `KnowledgeBase`, `hybrid_search` function, generator instance, prompt and default
`top_k`. Adaptive RAG differs only by the agents' additional steps. Standard RAG's verification is computed **after**
its answer is final ("post-hoc analysis") so the two can be compared, and is labelled as such everywhere.

## 5. Data and storage

| Path | Content | In git? |
|---|---|---|
| `data/runtime/kb/` | user knowledge base: `registry.json`, original files, Qdrant data | no |
| `data/runtime/benchmarks/<dataset>/` | separate index per benchmark dataset | no |
| `data/runtime/runs/` | every query response with its trace (last 200) | no |
| `data/runtime/model_cache/` | fastembed models | no |
| `data/datasets/smoke_test/` | synthetic benchmark | **yes** |
| `data/datasets/triviaqa_unfiltered_sample/` | prepared locally by script | no |
| `evaluation/results/<run_id>/` | `results.json` (config + rows) and `results.csv` | no (summarised in docs/results.md) |

User documents and benchmark corpora are **never mixed**. If the embedding model or chunk settings change, the user
knowledge base is rebuilt automatically from the stored original files.

## 6. Key design decisions

| Decision | Reason |
|---|---|
| Plain Python orchestration, no LangChain/LangGraph | The loop is ~100 lines; explicit code is easier to explain and debug. |
| One shared LLM, structured JSON output | Fits 4 GB VRAM; avoids loading four model copies. |
| Ollama for the LLM | Handles quantisation (Q4_K_M), GPU/CPU split and Windows setup in one installer. |
| Embeddings and NLI on CPU | Keeps the small GPU free for the LLM. |
| Qdrant **local mode** | Real vector database API without running a server; persists to a folder. |
| BM25 written in-house | 40 lines, no edge cases with tiny corpora, easy to explain. |
| Rules where rules suffice | Planner, confidence and most of verification are deterministic and testable. |
| Dev mode | Lets anyone run the UI and all unit tests without downloads, clearly labelled. |

## 7. Security notes

Local prototype: **no authentication**. CORS allows only the local frontend origins. Uploaded filenames are sanitised,
upload size is limited, dataset and run IDs are validated against path traversal, and documents are never sent to
external services (Ollama runs locally). Do not expose the server to a network without adding authentication.
