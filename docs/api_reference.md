# API Reference

Base URL: `http://127.0.0.1:8000`. Interactive documentation (try requests in the browser): **`/docs`**
(Swagger UI) and `/redoc`. Machine-readable schema: `/openapi.json`.
No authentication (local prototype).

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Backend status, mode, model readiness, knowledge-base size, configuration |
| GET | `/api/documents` | List indexed documents |
| POST | `/api/documents` | Upload one or more files (multipart field `files`) |
| DELETE | `/api/documents/{doc_id}` | Delete a document and its chunks |
| POST | `/api/query` | Run one question with `standard` or `adaptive` |
| POST | `/api/compare` | Run both methods on the same question |
| GET | `/api/runs?limit=20` | Recent runs (summary) |
| GET | `/api/runs/{run_id}` | One full run including its trace |
| GET | `/api/datasets` | Benchmark datasets with usable-question counts |
| POST | `/api/evaluations` | Start a benchmark in the background |
| GET | `/api/evaluations` | List running and saved evaluations |
| GET | `/api/evaluations/{id}` | Progress, per-question rows and summary |
| POST | `/api/evaluations/{id}/cancel` | Cancel a running evaluation |
| GET | `/api/evaluations/{id}/export?format=csv\|json` | Download results |

## Examples (PowerShell)

```powershell
# health
Invoke-RestMethod http://127.0.0.1:8000/api/health

# upload (curl.exe ships with Windows 10/11)
curl.exe -F "files=@data/sample_documents/college_library_rules.txt" http://127.0.0.1:8000/api/documents

# ask
$body = @{ question = "What is the fine for an overdue book?"; method = "adaptive" } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/api/query -Method Post -ContentType "application/json" -Body $body

# start an evaluation of 10 questions
$body = @{ dataset = "smoke_test"; limit = 10; methods = @("standard","adaptive") } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/api/evaluations -Method Post -ContentType "application/json" -Body $body
```

## `POST /api/query`

Request:

```json
{ "question": "What is the fine for an overdue book?", "method": "adaptive" }
```

`question`: 2–2000 characters. `method`: `standard` or `adaptive` (default).

Response (`QueryResponse`, abridged):

```json
{
  "run_id": "3f9c0a1b2c4d",
  "method": "adaptive",
  "question": "What is the fine for an overdue book?",
  "answer": "The fine for an overdue book is 2 rupees per day. [1]",
  "status": "answered",            // answered | abstained | no_documents | no_retrieval_needed | error
  "citations": [{"ref": 1, "filename": "college_library_rules.txt", "page": null, "chunk_index": 0, "valid": true, "...": "..."}],
  "evidence": [{"ref": 1, "text": "...", "rrf_score": 0.0328, "dense_score": 0.79, "bm25_score": 5.1, "keyword_overlap": 1.0, "...": "..."}],
  "verification": {"status": "supported", "support_score": 1.0, "citation_consistency": 1.0, "claims": ["..."], "method": "lexical+nli(...)"},
  "verification_used_in_decision": true,
  "confidence": {"score": 0.86, "components": {"relevance": 0.83, "coverage": 0.75, "verification": 1.0, "consistency": 1.0},
                 "decision": "accept", "reason": "...", "missing_components": [], "note": "Heuristic score, not a calibrated probability."},
  "trace": [{"step": "planner", "attempt": 0, "status": "completed", "duration_ms": 0.4, "input": {}, "output": {}, "decision": "retrieve", "note": "..."}],
  "stats": {"latency_ms": 5109, "llm_calls": 1, "nli_calls": 1, "retrieval_attempts": 1, "retries": 0},
  "mode": "real",
  "models": {"generator": "ollama:qwen3:4b-instruct-2507-q4_K_M", "embedder": "BAAI/bge-small-en-v1.5", "nli": "...", "reranker": "none"}
}
```

For Standard RAG, `confidence` is `null` and `verification` is post-hoc analysis (`verification_used_in_decision: false`).

## `POST /api/documents`

Multipart upload, one or more `files` fields. Returns one result per file:

```json
[{"filename": "notes.txt", "status": "indexed", "message": "Indexed 2 chunks.", "document": {"doc_id": "a1b2c3d4e5f6", "num_chunks": 2, "...": "..."}},
 {"filename": "scan.pdf", "status": "failed", "message": "No extractable text found. Scanned/image-only PDFs need OCR, which is not implemented.", "document": null}]
```

## Errors

| Code | When |
|---|---|
| 400 | invalid evaluation id |
| 404 | unknown document, run, dataset or evaluation |
| 409 | an evaluation is already running |
| 422 | request validation failed (empty question, unknown method, bad export format, no files) |
| 503 | the language model could not be reached |

Error bodies have the FastAPI shape `{"detail": "..."}`.
