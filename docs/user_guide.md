# User Guide — the web dashboard

Open http://localhost:3000 after starting the backend and frontend ([setup_guide.md](setup_guide.md)).
The left sidebar (a menu button on phones) links to six pages. The bottom of the sidebar shows whether the backend
is online, the run mode (**DEV MODE** in yellow, or the local model name), and the **light/dark theme toggle**.
The theme follows your operating system until you toggle it; your choice is remembered.

## 1. Overview

- Short description of the project and the four agents.
- **Knowledge base** size (documents, chunks), **run mode**, and the configured thresholds.
- **Model readiness**: embedder, generator (LLM), NLI verifier and reranker, reported live by the backend.
- **Quick actions** to Upload, Ask and Evaluate.
- **Recent query activity**: the last saved runs; click one to open its trace.

## 2. Document Library

1. Drag files onto the upload box or click **Choose files** (PDF with a text layer, DOCX, TXT, Markdown).
2. A progress bar shows the upload; then the backend extracts, chunks, embeds and indexes the files.
3. Each file reports **indexed**, **duplicate** (identical content already present) or **failed** with the reason
   (unsupported type, too large, scanned PDF without text, corrupt file).
4. The table lists every document with type, size, pages, chunk count and date.
5. The bin icon deletes a document after a confirmation dialog; its chunks are removed from the index.

## 3. Ask Documents

1. Type a question (Ctrl + Enter submits).
2. Choose **Adaptive RAG** (four agents) or **Standard RAG** (baseline).
3. Press **Ask**. **Cancel** stops waiting in the browser (a run already started on the server still finishes).
4. The result shows:
   - **Answer** with citation chips `[1]`; click one to jump to its passage.
   - **Citations**: filename, page (PDFs) and chunk number.
   - **Confidence Agent** card (adaptive only): score, the four components, decision and reason.
   - **Verifier Agent** card: every claim with its label, lexical support and NLI entailment.
   - **Retrieved evidence**: click a passage to expand it and see its RRF, dense, BM25 and keyword-overlap scores.
   - If evidence was insufficient you see an **abstention** message instead of an answer.
   - If the library is empty you see instructions to upload documents; no answer is generated.

## 4. Agent Trace

Open from **View agent trace** or from the recent-runs list. It shows the **real trace returned by the backend**:

- Summary: method, outcome, latency, LLM calls, NLI calls, retrieval attempts.
- A timeline grouped by attempt (*Attempt 1 (initial)*, *Attempt 2 (adaptive retry 1)*, ...).
- Each step shows status (**completed / skipped / failed**), decision (accept / retry / abstain ...), duration and a
  note; **Show inputs and outputs** reveals the exact JSON the step received and produced (plan, passages,
  claims, confidence components, retry reasons).

## 5. Compare RAG

Enter one question and press **Compare**. Both pipelines run on the same knowledge base. You get:

- A table of measured values: outcome, latency, LLM calls, NLI calls, retrieval attempts, citation count,
  verification label and support score, confidence (adaptive only).
- Both answers side by side with their evidence, links to their traces, and the reasons for any adaptive retries.

One question is an example, not proof; use the Evaluation page for measured comparisons.

## 6. Evaluation

1. Pick a dataset. *Synthetic* (smoke test) and *research benchmark sample* (TriviaQA) are labelled.
2. Optionally enter a sample size (fixed-seed random sample) and choose the methods.
3. **Run evaluation**. A progress bar and per-question rows update live; **Cancel** stops the run (no result files
   are written for a cancelled run).
4. When complete: number of questions, failed cases, runtime, mode, a bar chart of rate metrics, the full metrics
   table, and per-question outcomes (correct / incorrect / abstained correctly / abstained wrongly).
5. **CSV** and **JSON** buttons download the result; the JSON includes the configuration and model names.
6. **Previous runs** reopens earlier results (also those produced with the command-line benchmark).

Metric definitions: [evaluation_protocol.md](evaluation_protocol.md).
