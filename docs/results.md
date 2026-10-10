# Results

All numbers below come from actual runs on 2026-10-10 in **real mode** on the development laptop
(Windows 11, 32 GB RAM, RTX 3050 Laptop GPU 4 GB). Configuration: default `.env.example` values
(top-k 5, chunk 900/150, weights 0.25/0.25/0.35/0.15, accept 0.65, abstain 0.40, max 2 retries, no reranker).

| Component | Model |
|---|---|
| Generator (shared) | `qwen3:4b-instruct-2507-q4_K_M` via Ollama, temperature 0, seed 42 |
| Embeddings | `BAAI/bge-small-en-v1.5` (fastembed) |
| Verifier NLI | `cross-encoder/nli-deberta-v3-xsmall` |

Re-create any table with `python evaluation/run_benchmark.py --dataset <name> --mode real`; the run IDs below are
the folder names written to `evaluation/results/` (git-ignored; the JSON contains every per-question answer).
Results on the two datasets must **not** be compared with each other.

---

## 1. Synthetic smoke test (pipeline verification, not a research benchmark)

Run `20261010-212100-3c2e8d` · 24 questions (18 answerable, 6 unanswerable) · runtime 220 s.

| Metric | Standard RAG | Adaptive RAG |
|---|---|---|
| Accuracy (answerable) | 100 % (18/18) | 100 % (18/18) |
| Exact match | 44.4 % | 44.4 % |
| Token F1 | 0.680 | 0.680 |
| Correct abstention (unanswerable) | 100 % (6/6) | 100 % (6/6) |
| Wrong abstention (answerable) | 0 % | 0 % |
| Cites a gold document | 100 % | 100 % |
| Mean latency | 2.8 s | 6.0 s |
| Mean LLM calls | 1.00 | 2.33 |
| Mean retrieval attempts | 1.00 | 1.67 |

**Reading:** the pipeline works end to end. The set is easy: the evidence is short and clean, and the LLM itself
reported "insufficient" for every unanswerable question, so both methods behave identically. Adaptive RAG only
added cost here (its retries happened on the unanswerable questions before abstaining).

---

## 2. TriviaQA-unfiltered sample (research benchmark sample)

Run `20261010-223619-6d2293` · first 100 validation questions · 475 pooled evidence documents ·
76 answerable / 24 unanswerable given the retained evidence ([protocol](evaluation_protocol.md)) · runtime 31 min.

| Metric | Standard RAG | Adaptive RAG |
|---|---|---|
| **Accuracy** (answerable, abstain = wrong) | **78.9 %** (60/76) | 68.4 % (52/76) |
| Exact match | 38.2 % | 29.0 % |
| Token F1 | 0.554 | 0.459 |
| Answer rate on answerable | 85.5 % | 76.3 % |
| Selective accuracy (answered only) | 92.3 % | 89.7 % |
| **Correct abstention** (unanswerable) | 54.2 % (13/24) | 54.2 % (13/24) |
| **Wrong abstention** (answerable) | 14.5 % | 23.7 % |
| Cites a gold document (when answered) | 100 % | 100 % |
| Evidence recall (answer in retrieved passages) | 94.7 % | 97.4 % |
| Mean support score (own verifier, see caveat) | 0.763 | 0.929 |
| Mean / median latency | 4.4 s / 4.0 s | 11.5 s / 7.5 s |
| Mean LLM calls | 1.00 | 2.46 |
| Mean retrieval attempts | 1.00 | 1.73 |
| Failed (error) cases | 0 | 0 |

### Paired comparison (same 76 answerable questions)

| | Adaptive correct | Adaptive abstained | Adaptive wrong |
|---|---|---|---|
| **Standard correct** | 48 | 11 | 1 |
| **Standard abstained** | 4 | 6 | 1 |
| **Standard wrong** | 0 | 1 | 4 |

Standard RAG was right where Adaptive was not on 12 questions; the reverse happened on 4. An exact two-sided sign
test on these 16 discordant questions gives p ≈ 0.08, so with 100 questions even this difference is not
statistically significant at the 5 % level.

On the 24 unanswerable questions both methods abstained on the same 13 and answered the same 11 (none of those 11
answers contained a gold alias).

### What the results mean (honest interpretation)

1. **On this sample the proposed Adaptive RAG did not improve accuracy; it was lower (68.4 % vs 78.9 %)**, mainly
   because it abstained on more answerable questions (wrong abstention 23.7 % vs 14.5 %).
2. **Cause (failure analysis).** Of the 11 questions Standard answered correctly but Adaptive abstained on, the final attempt had a
   verifier label of *contradicted* in 9 cases (e.g. correct one-word answers "Niger", "Brunei", "Seymour Hersh"); in the other 2 the LLM itself reported insufficient evidence on every attempt. TriviaQA
   answers are usually bare names; for many question shapes the Verifier's question-to-statement heuristic does not
   apply, the NLI model then receives an ill-formed hypothesis (question + name) and often outputs "contradiction".
   The Confidence Agent correctly treats a contradiction as a reason to retry and finally abstain, so a verifier
   error becomes a lost answer. This is a limitation of the verifier design, not of the retry logic.
3. **Where Adaptive helped:** 4 answerable questions that Standard RAG gave up on were answered correctly after
   adaptive re-retrieval, and evidence recall rose from 94.7 % to 97.4 %.
4. **Abstention on unanswerable questions was not better** (54 % for both): when the LLM produced an answer from
   loosely related evidence, both pipelines kept it for the same questions.
5. **Cost:** about 2.5× the LLM calls and 2.6× the mean latency.
6. The higher mean support score of Adaptive RAG is expected by construction (it filters on the same verifier) and
   is **not** independent evidence of better faithfulness.

### What would likely fix the main failure (future work, not applied to these results)

- Do not use the NLI *contradiction* signal for hypotheses that could not be converted into a proper statement; use
  lexical support plus NLI *entailment* only in that case.
- Ask the LLM to answer in a full sentence that restates the question (the prompt already requests this; the 4B model
  often ignores it), or let the LLM rewrite the short answer into a statement before NLI.
- Then evaluate on a **different** set of TriviaQA questions (e.g. validation questions 101–300), because tuning on
  these 100 questions and re-reporting on them would overstate the improvement.

The code was intentionally **not** changed after seeing these results, so the reported numbers reflect the system as
designed before testing.

---

## 3. Summary for the report

- The four-agent pipeline, traces, retries and abstention work as designed (verified by tests and the smoke set).
- On a 100-question TriviaQA-unfiltered sample with a 4B local model, the adaptive pipeline was **more cautious and
  more expensive, and less accurate**, because its verifier mislabels short correct answers as contradicted.
- This is a useful, honest finding: answer verification is only as good as the verifier, and an over-strict verifier
  turns into unnecessary abstentions. Improving the verifier is the most promising next step.
