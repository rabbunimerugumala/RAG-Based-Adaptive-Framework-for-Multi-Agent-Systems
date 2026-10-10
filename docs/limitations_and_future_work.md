# Limitations and Future Work

## What is implemented

Upload and indexing of PDF (text layer) / DOCX / TXT / Markdown; hybrid retrieval (Qdrant + BM25 + RRF, optional
reranker); Standard RAG; the four-agent Adaptive RAG with verification, confidence scoring, up to two retries and
abstention; full execution traces; comparison; benchmark runner with metrics and CSV/JSON export; light/dark web
dashboard; development mode; mocked and real-model tests.

## Known limitations

0. **Measured weakness:** on the 100-question TriviaQA sample the verifier's NLI step marked several short correct
   answers as *contradicted*, so Adaptive RAG abstained more often and was less accurate than Standard RAG
   (68.4 % vs 78.9 %). See [results.md](results.md) for the analysis and the proposed fix.
1. **Hallucinations are not eliminated.** The verifier can miss unsupported claims (e.g. paraphrases that
   share words with the evidence) and can reject correct ones.
2. **Confidence is a heuristic**, not a calibrated probability; weights and thresholds were set by reasoning, not tuned
   or calibrated on labelled data.
3. **Rule-based planner and verifier helpers** (query types, focus terms, question-to-statement conversion) cover
   common English question shapes only. Example: for "Which direction should rooftop panels face?" with the answer
   "South", the statement conversion fails, the claim is only *partially supported*, and the system needs retries
   before accepting with moderate confidence.
4. **Small NLI model** (DeBERTa-v3-xsmall) on 3-sentence windows: limited multi-hop reasoning; long passages are
   truncated to 384 tokens.
5. **Cost**: adaptive runs use more LLM calls and time, especially when they retry or abstain (each retry adds a
   generation and, with a real LLM, a query-rewrite call).
6. **Hardware**: the 4-bit 4B model does not fit entirely in 4 GB VRAM at 4096 context; part runs on CPU.
7. **No OCR**: scanned PDFs are rejected. Tables in PDFs are extracted as plain text only.
8. **English only** (stop-words, stemming, models).
9. **Evaluation scale**: 24 synthetic questions and a 100-question TriviaQA sample; differences of a few points are
   within noise. Alias matching can misjudge paraphrased answers. The support score uses the project's own verifier.
10. **Single user, local**: no authentication, one process per index folder (Qdrant local mode), one evaluation at a time.
11. Cancelling a query in the browser does not stop the computation already running on the server.

## Future work

- Calibrate the confidence score (e.g. logistic regression on labelled correct/incorrect answers; reliability diagrams).
- Tune weights/thresholds on a validation split and report on a separate test split.
- Larger evaluations (more TriviaQA questions, other datasets such as Natural Questions or HotpotQA) with confidence intervals.
- Stronger verifier: a larger NLI model or an LLM judge cross-checked against NLI; claim extraction with an LLM.
- LLM-based planner for complex multi-hop questions; query decomposition learned from data.
- OCR (e.g. Tesseract) for scanned PDFs; table-aware PDF parsing.
- Streaming answers and server-side cancellation.
- Multi-user deployment with authentication and a Qdrant server.
