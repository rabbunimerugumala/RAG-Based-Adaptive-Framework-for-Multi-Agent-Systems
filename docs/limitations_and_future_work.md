# Limitations and Future Work

## What is implemented

Upload and indexing of PDF (text layer) / DOCX / TXT / Markdown; hybrid retrieval (Qdrant + BM25 + RRF, optional
reranker); the Standard RAG baseline; the four-agent Multi-Agent RAG v2 with a MAIN-RAG-inspired judged evidence
filter (log-odds score, adaptive bar, ordering), grounded claim verification, confidence-based answer / retry /
partial answer / abstain decisions, bounded retries; full execution traces; comparison page; benchmark runner with
ablation variants, metrics, McNemar analysis and CSV/JSON export; light/dark dashboard; development mode; mocked and
real-model tests.

## Known limitations

1. **Not a MAIN-RAG reproduction.** Default judging uses an adaptation (Document–Question pairs), filtering runs only
   on escalation, the pool is 10–30 passages, and the model is Qwen3-4B; see [main_rag_alignment.md](main_rag_alignment.md).
2. **Hallucinations are reduced, not eliminated.** The verifier can miss unsupported paraphrases and the judge and
   NLI model can both be wrong. A citation does not prove a claim.
3. **Confidence is a heuristic**, not a calibrated probability; weights and thresholds were not tuned on data
   (the v2 design changes were chosen on the development set; thresholds were left at their original values).
4. **The judge is the same LLM that writes the answer**, so its errors are not independent of the generator's.
5. **Cost.** Escalated questions use many LLM calls (one judge call per candidate passage) and take tens of seconds
   on a 4 GB GPU; see the latency and call counts in [results.md](results.md).
6. **Rule-based helpers** (query types, focus terms, question-to-statement conversion) cover common English question
   shapes only.
7. **Hardware.** The 4-bit 4B model does not fit entirely in 4 GB VRAM at 4096 context; part runs on CPU.
8. **No OCR**, English only, single user, no authentication, one process per index folder.
9. **Evaluation scope.** One dataset family (TriviaQA-unfiltered) with truncated evidence, 199 held-out questions,
   alias-match accuracy, one model. Differences of a few points can be noise; see the McNemar tests.
10. Browser cancel does not stop a query already running on the server.

## Future work

- Calibrate the confidence score on labelled data (e.g. logistic regression; reliability diagrams) and tune
  thresholds on a validation split.
- Use a separate (smaller or different) model as the evidence judge to make its errors independent of the generator.
- Batch or cache judge calls; streaming answers; server-side cancellation.
- Evaluate on more datasets (PopQA, Natural Questions, HotpotQA) and more questions, with confidence intervals.
- Paper-faithful `triplet` mode and `always` filtering stage as further ablations.
- OCR for scanned PDFs; table-aware parsing; multilingual support.
