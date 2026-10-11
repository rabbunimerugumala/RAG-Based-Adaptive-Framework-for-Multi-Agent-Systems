# Frequently Asked Questions

Short, honest answers to common questions about the design and results. Read [how_it_works.md](how_it_works.md) first.

## A. Concept

**1. What is your project in one sentence?**
An adaptive Multi-Agent RAG system: four cooperating agents plan the search, gather and filter evidence, verify the
answer claim by claim, and then answer, retrieve again or abstain; Standard RAG is kept as the comparison baseline.

**2. What is RAG and why is it needed?**
Retrieval-Augmented Generation first retrieves relevant passages from a document collection and then asks an LLM to
answer using only those passages. It gives the LLM knowledge it was not trained on and lets the answer cite sources.

**3. What problem remains with Standard RAG?**
It trusts whatever the retriever returns and whatever the LLM writes. Nobody checks whether the answer is supported,
and it rarely says "I don't know".

**4. What are your four agents?**
Planner (analyse the question, plan the search and the evidence budget), Evidence (hybrid retrieval; on escalation,
MAIN-RAG-style LLM judging, adaptive-threshold filtering and ranking), Verifier (check each claim with lexical checks,
an NLI model and the judge), Confidence (score and decide: answer, retrieve again, or abstain).

**5. Are they really separate agents or just prompts?**
They are separate Python classes with different inputs, outputs and logic (`src/adaptive_rag/agents/`). The LLM is
used for the draft answer, as the Evidence Agent's judge (only on escalation), to confirm short answers or cross-check
contradictions in the Verifier, and for the Planner's retry queries. All roles share one LLM instance (4 GB VRAM).

**6. How is this different from MAIN-RAG (ACL 2025)?**
MAIN-RAG (Predictor, Judge, Final-Predictor) scores every retrieved document with log P(Yes) − log P(No), keeps those
above the per-query mean score and orders them. We use exactly that score and threshold inside our Evidence Agent, but
(a) by default the judge sees (document, question) pairs instead of the paper's (document, question, answer) triplets,
(b) filtering runs only when the Confidence Agent asks for more evidence, and (c) we add claim verification, a
confidence score, retries and abstention, which MAIN-RAG does not have. It is not a reproduction; see
[main_rag_alignment.md](main_rag_alignment.md).

## B. Techniques

**7. Why hybrid retrieval?**
Dense embeddings capture meaning ("tall" ≈ "high"); BM25 captures exact words, names and numbers ("Kestrel-9", "7425").
Combining them is more robust than either alone.

**8. What is Reciprocal Rank Fusion?**
Each passage gets `Σ 1/(k + rank)` over all ranked lists (k = 60). It needs only ranks, not comparable scores, so it can
merge cosine similarities and BM25 scores fairly.

**9. Explain BM25 briefly.**
A keyword score: for each query term, rarer terms (higher IDF) count more, repeated terms count more with diminishing
returns (k1), and long documents are normalised (b). We implemented it ourselves in `retrieval/bm25.py`.

**10. What is NLI?**
Natural Language Inference: given a premise and a hypothesis, predict entailment, contradiction or neutral. We use the
evidence as premise and each answer claim as hypothesis.

**11. How is the confidence score calculated?**
`C = 0.25·relevance + 0.25·coverage + 0.35·verification + 0.15·consistency`, each in [0,1]. Missing components are
listed and excluded from the weighted mean. Accept at ≥ 0.65 with a supported verdict, retry otherwise (max 2), and
after the last retry accept only if ≥ 0.40, else abstain.

**12. Is the confidence a probability?**
No. It is a transparent heuristic; it has not been calibrated against labelled data, and we say so in the UI and docs.

**13. Why these weights and thresholds?**
Verification is the most direct evidence of correctness, so it has the largest weight. The values are experimental
settings, configurable in `.env`, not universal truths. Tuning them on a validation set is future work.

**14. How do you prevent infinite retries?**
`max_retries` is validated to be at most 2 in `config.py`, and the loop runs at most `max_retries + 1` times.

**15. How do retries differ from the first attempt?**
The Confidence Agent picks a strategy. *Broaden* (weak retrieval): larger top-k, double the candidates, relaxed
keyword-pair queries. *Targeted* (unsupported claims): queries built from the unsupported claims and missing keywords,
plus up to 3 reformulations from the LLM.

## C. Implementation

**16. Why Ollama and Qwen3-4B?**
Free, local, Apache-2.0, good JSON instruction following, and a 4-bit build (~2.5 GB) that fits a 4 GB GPU with partial
CPU offload. The model name is one setting, so it can be swapped.

**17. Why Qdrant local mode?**
A real vector database API that stores data in a folder, with no server or Docker to install.

**18. What is dev mode?**
A no-download mode with a hashing embedder and an extractive sentence picker instead of an LLM, used for development
and unit tests. Its outputs are labelled and never reported as results.

**19. How do you keep the comparison fair?**
Same knowledge base, same retrieval function, same LLM and prompt, same default top-k, same questions in the same
order, temperature 0, fixed seed.

**20. How did you test it?**
156 fast tests with a scripted mock LLM and a fake NLI model covering every agent, routing and decision branch, plus 4 opt-in
tests with the real models, frontend lint/build, and live end-to-end runs. See [testing.md](testing.md).

## D. Results and limitations

**21. What did you find?**
On 199 held-out TriviaQA questions, Multi-Agent RAG v2 answered more questions correctly than Standard RAG in two
separate runs: 73.9 % vs 67.9 % (13 vs 3 discordant questions, exact McNemar p = 0.021) and 72.7 % vs 69.1 % (9 vs 3,
p = 0.146). The direction is consistent (fewer wrong refusals), but significance at 5 % appeared in only one run, so
the honest conclusion is "promising, not yet conclusive". Adaptive retries carry the gain; the MAIN-RAG-style filter
adds little here; question routing keeps accuracy with ~21 % fewer LLM calls. The cost is ~4× the latency.

**22. Does your system eliminate hallucination?**
No. It reduces the chance of returning unsupported answers by verifying and abstaining, but the verifier itself can make
mistakes.

**23. Main limitations?**
Heuristic, uncalibrated confidence; rule-based helpers for English questions; small NLI model; extra latency; small
evaluation samples; no OCR. See [limitations_and_future_work.md](limitations_and_future_work.md).

**24. What would you do next?**
Calibrate confidence on labelled data, run larger benchmarks with confidence intervals, use a stronger verifier, add OCR.

## Demo script (5 minutes)

1. Overview page: models ready, four agents.
2. Document Library: upload `data/sample_documents/college_library_rules.txt` and `solar_energy_basics.md`.
3. Ask (Adaptive): "How many books can an undergraduate student borrow and for how long?" → citations, supported claim, confidence → **View agent trace**.
4. Ask: "What is the WiFi password of the library?" → retries and **abstains**; show the retry reasons in the trace.
5. Compare: "What is the fine for an overdue book?" → side-by-side cost (LLM calls, latency).
6. Evaluation: open the saved TriviaQA run → chart, metrics table, per-question outcomes, CSV download; explain the honest result.
7. Toggle dark mode.


## E. The v2 upgrade

**25. What changed between v1 and v2, and why?**
The original v1 lost accuracy against Standard RAG on TriviaQA because the verifier wrongly rejected correct short
answers (NLI "contradictions" from ill-formed hypotheses and off-topic passages). v2 fixes the verifier (grounded
contradictions, judge-confirmed short answers, corroborated contradictions), adds the MAIN-RAG-style evidence filter
on escalation, and selects verified attempts first. All changes are switchable, so v1 can still be reproduced.

**26. How do you make sure the improvement is not just tuning to the test?**
All design decisions were made on a development set (TriviaQA questions 1–100). The code was frozen (git commit
`808ef33`) and then evaluated once on 199 different questions (101–300). Thresholds and weights were not tuned.

**27. Why is filtering not applied on every round like in the paper?**
On the development set, filtering every round kept only 2–4 passages and sometimes removed the one with the answer,
and it cost ~14 LLM calls per question. Running it only when confidence is low kept the round-1 evidence identical to
Standard RAG and roughly halved the cost.

**28. Is the confidence score a probability?**
No. It is a weighted heuristic of four normalised signals; the evidence-relevance part comes from the LLM judge, the
other parts from coverage and the Verifier. It has not been calibrated.
