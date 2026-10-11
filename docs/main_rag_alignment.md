# MAIN-RAG and This Project: Paper-to-Code Alignment

Reference: C.-Y. Chang et al., *MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation*, ACL 2025
(Volume 1: Long Papers), pp. 2607–2622. The paper is read in full (method §3, experiments §4, Tables 1–2, case
studies, limitations, Appendices A–D). The PDF is not redistributed in this repository.

**Summary.** MAIN-RAG is a three-agent *document filter*. This project is a four-agent *adaptive answer pipeline*
(plan → gather evidence → verify claims → decide). Since v2, the Evidence Agent contains a MAIN-RAG-style judge and
adaptive filtering stage. This project does **not** reproduce MAIN-RAG and does not report or compare against the
paper's numbers.

## 1. What MAIN-RAG does (from the paper)

| Paper element | Description |
|---|---|
| Agent-1 Predictor (§3.1) | For each retrieved document, the LLM answers the query from that document alone, forming Doc-Q-A triplets. |
| Agent-2 Judge (§3.1–3.2) | For each triplet, the LLM answers "Yes"/"No": does the document provide specific information for the question **and** does the answer directly answer it from the document (Appendix C instruction). |
| Relevance score (§3.2) | `s_i = log P("Yes") − log P("No")` of the judge's output token. |
| Adaptive judge bar (§3.3) | `τ_q = mean(s)` over the query's candidate documents; optionally relaxed to `τ_q − n·σ`. `n` is the only hyperparameter (default 0). |
| Filtering and ordering (§3.3, Table 2) | Keep documents with `s_i ≥ τ_q`, order them by descending score (better than ascending in the paper's ablation). |
| Agent-3 Final-Predictor | Answers the query from the filtered, ordered list. |
| Setup (§4.3) | Up to 20 retrieved documents (Contriever-MS MARCO), Mistral-7B or Llama-3-8B, greedy decoding; accuracy = gold answer contained in the generation. |
| Not in MAIN-RAG | Claim-level answer verification, a confidence score, retries, abstention. |

## 2. Mechanism-by-mechanism comparison

| MAIN-RAG mechanism | This project | Status | Code |
|---|---|---|---|
| A. Per-document Predictor | `doc_judge_mode="triplet"`: `OllamaLLM.predict()` with the Appendix C Predictor instruction, one call per candidate | Implemented (optional) | `generation/llm.py` `predict()`; `agents/evidence.py` |
| B. Judge on Doc-Q-A triplets | Triplet mode uses the **verbatim Appendix C Judge instruction**. Default **pair mode** judges (Document, Question) only: condition (1) of the paper, without the Predictor. This is an **adaptation** to save one LLM call per document. | Implemented; default is an adaptation | `JUDGE_TRIPLET_SYSTEM`, `JUDGE_PAIR_SYSTEM`, `OllamaLLM.judge()` |
| C. Score = log P(Yes) − log P(No) | Exact. Ollama's `logprobs`/`top_logprobs` (top 20) for the first output token; case/space variants of Yes/No merged; if a token is absent from the top-20 its log-probability is bounded by (lowest returned − 1) and flagged as `floor_used`. | Implemented | `yes_no_score()` |
| D. Adaptive bar τ_q = mean − n·σ | Exact (population std), `ARAG_DOC_FILTER_N_SIGMA`, default 0 | Implemented | `adaptive_threshold()` |
| E. Filter + descending order | Keep `s_i ≥ τ_q`; descending score, ties in retrieval order. **Additions not in the paper:** near-duplicate removal (word Jaccard ≥ 0.8), a minimum of 1 kept passage, and a cap at the evidence budget (top-k = 5). | Implemented + safeguards | `select_by_threshold()` |
| F. Final-Predictor on filtered docs | The shared generator answers from the kept, ordered passages with citations | Implemented | `pipeline._generate()` |
| Candidate pool | 10 passages (pool grows by 5 on a "broaden" retry, max 30) instead of 20 documents, because each judgment is one LLM call on a 4 GB GPU | Different (documented) | `ARAG_DOC_POOL_K` |
| When filtering runs | **On escalation only** (default `doc_filter_stage=on_retry`): round 1 uses plain hybrid top-k; filtering runs when the Confidence Agent asks for more evidence. `always` mode filters on every round like the paper. | Different (design decision, see §4) | `agents/planner.py` `_retry_pool()` |
| Model | Qwen3-4B-Instruct (Q4) via Ollama, temperature 0, seed 42 | Different | `.env` |

## 3. How the four agents use the MAIN-RAG ideas

| Agent | Role | MAIN-RAG idea used |
|---|---|---|
| Planner | query analysis, evidence budget, retry strategy; decides when the Evidence Agent escalates to judged filtering | — |
| Evidence | hybrid retrieval → dedup → **judge scores** → **adaptive bar** → **descending order** → budget | Predictor (optional), Judge, log-odds score, τ_q, ordering |
| Verifier | claim-level check of the *answer*; short answers need the answer text in a passage **and** a positive triplet judgment; an NLI contradiction must be corroborated by the judge | the Appendix C triplet judge reused as a second, independent support check |
| Confidence | relevance (share of passages the judge rated relevant), coverage, verification, citation consistency → accept / retry / abstain | judge scores as an evidence-quality signal independent of the verifier |

## 4. Why filtering is not applied on the first round (development evidence)

On the development set (TriviaQA validation questions 1–100), filtering on every round kept only 2–4 of 10
passages for many questions and sometimes removed the passage that contained the answer, so the generator answered
"insufficient" where plain top-5 retrieval had worked (e.g. *"Which musical featured the song The Street Where You
Live?"*). It also cost about 14 LLM calls per question. The escalation design keeps the round-1 evidence identical
to the Standard RAG baseline and spends judge calls only on questions the Confidence Agent could not accept. This
decision was made on the development set only; the held-out test set was not used for it.

## 5. What is *not* claimed

- No reproduction of MAIN-RAG's results; no comparison with its numbers (different model, retriever, corpus, sample).
- Pair mode is an adaptation, and the escalation stage is this project's design, not the paper's.
- Judge scores measure relevance/support as judged by the same LLM; they are not calibrated probabilities and are
  not proof that a claim is true.
