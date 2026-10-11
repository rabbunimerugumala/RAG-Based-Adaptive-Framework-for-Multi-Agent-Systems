# Results

This file has three parts, kept separate on purpose:

* **Part A — original baseline (v1)**, 2026-10-10: unchanged from the first report (with audit notes appended).
* **Part B — v2 development** on the development set (TriviaQA questions 1–100), used to make design decisions.
* **Part C — held-out test** of the frozen v2 code on 199 new questions (101–300): the final comparison.

---

# Part A — Original baseline (v1)

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

## 3. Summary (Part A)

- The four-agent pipeline, traces, retries and abstention work as designed (verified by tests and the smoke set).
- On a 100-question TriviaQA-unfiltered sample with a 4B local model, the adaptive pipeline was **more cautious and
  more expensive, and less accurate**, because its verifier mislabels short correct answers as contradicted.
- This is a useful, honest finding: answer verification is only as good as the verifier, and an over-strict verifier
  turns into unnecessary abstentions. Improving the verifier is the most promising next step.


### Audit notes on Part A (added 2026-10-11; no number above was changed)

An audit recomputed every Part A metric from the saved per-question rows: all numbers match. Corrections to wording:

* The verifier label quoted for the 11 questions "Standard correct, Adaptive abstained" is the label of the
  **best-scoring attempt selected by the Confidence Agent**, not necessarily the final attempt; v1 benchmark runs did
  not save per-attempt drafts, so "the LLM reported insufficient evidence on every attempt" cannot be verified.
* The short-answer examples ("Niger", "Brunei", "Seymour Hersh") are **Standard RAG's** saved answers; re-verifying
  them with the v1 verifier reproduced the *contradicted* label in 5 of 6 cases (failure mechanism: ill-formed
  "question? answer." NLI hypotheses on 43 of 100 questions, and maximum-contradiction over off-topic windows).
* Adaptive's 18 wrong abstentions (23.7 %) = 11 where Standard was correct + 6 where both abstained + 1 where Standard
  was wrong; Standard's 11 (14.5 %) = 6 + 4 + 1. The "11" in the failure analysis is the paired subset, not a total.
* The p ≈ 0.08 is an **exact two-sided McNemar test** (binomial test on the 16 discordant questions: 12 vs 4,
  p = 0.0768; mid-p 0.049). It is now produced by `evaluation/analyze_results.py`.
* Evidence conditions were not identical even in round 1: v1 Adaptive searched with 2 query variants, Standard with 1.

---

# Part B — Multi-Agent RAG v2: development (TriviaQA questions 1–100)

These runs informed design decisions, so they are **not** evidence of improvement. The Standard RAG reference is the
Part A run `20261010-223619-6d2293` (same questions, model, settings; Standard RAG code unchanged).

| Questions 1–100 | Standard (Part A) | MA v1 (Part A) | MA v2 iteration 1 | MA v2 iteration 2 (frozen) |
|---|---|---|---|---|
| Run | `20261010-223619-6d2293` | same | `20261011-110725-f69cff` | `20261011-115334-b96d4c` |
| Accuracy (of 76 answerable) | 78.9 % (60) | 68.4 % (52) | 78.9 % (60) | 82.9 % (63) |
| Wrong abstention | 14.5 % | 23.7 % | 13.2 % | 7.9 % |
| Correct abstention (of 24) | 54.2 % | 54.2 % | 54.2 % | 54.2 % |
| Mean LLM calls / question | 1.0 | 2.46 | 14.4 | 7.2 |
| Mean / median latency | 4.4 / 4.0 s | 11.5 / 7.5 s | 25.4 / 18.4 s | 13.8 / 5.9 s |
| Paired vs Standard (only-MA correct : only-Standard correct) | – | 4 : 12 | 5 : 5 | 5 : 2 (p = 0.45) |

* **Iteration 1** (judge filtering on every round, grounded verifier, Confidence v2): recovered v1's accuracy loss but
  first-round filtering kept only 2–4 of 10 passages and sometimes removed the answer passage; ~14 LLM calls/question.
* **Iteration 2** (filtering only on escalation, NLI contradictions must be corroborated by the judge): the version
  frozen at commit `808ef33` for Part C. No thresholds or weights were changed in either iteration.

---

# Part C — Held-out test of the frozen v2 (TriviaQA questions 101–300)

Run `20261011-122005-0a2d3c` · 2026-10-11 · 199 questions (165 answerable, 34 unanswerable given the evidence) ·
978 pooled documents · 5 systems × 199 questions, interleaved per question · 0 failed cases · runtime 2 h 44 min.
Code: frozen commit `808ef33` (the results file records `a703ed7` because the commit was read when the file was saved;
the commits in between changed only documentation, UI and a trace-note string, not decision logic — the runner now
records the commit at start). Same model, embedder, NLI model, corpus, prompt and top-k for every system.
Reproduce: see [evaluation_protocol.md](evaluation_protocol.md) §4.

| Metric | Standard RAG | **Multi-Agent v2 (full)** | v2 without retries | v2 without filtering | v1 (original) |
|---|---|---|---|---|---|
| **Accuracy** (of 165 answerable) | 67.9 % (112) | **73.9 % (122)** | 63.0 % (104) | 72.7 % (120) | 64.2 % (106) |
| Exact match | 32.7 % | 37.6 % | 32.7 % | 37.0 % | 31.5 % |
| Token F1 | 0.469 | 0.520 | 0.444 | 0.504 | 0.437 |
| **Wrong abstention** (answerable) | 19.4 % (32) | **13.9 % (23)** | 27.3 % (45) | 17.6 % (29) | 29.1 % (48) |
| **Correct abstention** (of 34 unanswerable) | 79.4 % (27) | 79.4 % (27) | 82.4 % (28) | 76.5 % (26) | 79.4 % (27) |
| Abstention rate (all) | 29.6 % | 25.1 % | 36.7 % | 27.6 % | 37.7 % |
| Selective accuracy (answered) | 84.2 % | 85.9 % | 86.7 % | 88.2 % | 90.6 % |
| Unsupported-answer rate (answered) | 20.0 % | 18.1 % | 17.5 % | 16.7 % | 14.5 % |
| Citation correctness (cites a gold doc) | 94.7 % | 95.1 % | 95.8 % | 96.3 % | 96.6 % |
| Evidence precision / recall | 54.8 % / 92.1 % | 58.9 % / 95.2 % | 53.8 % / 92.1 % | 53.5 % / 95.8 % | 52.9 % / 96.4 % |
| Grounded answers (lexical) | 82.1 % | 85.2 % | 86.5 % | 86.1 % | 83.9 % |
| Response consistency (20 repeats, same answer) | 85 % | 100 % | 100 % | 85 % | 90 % |
| Mean / median latency | 3.6 / 3.3 s | 15.2 / 6.8 s | 3.9 / 3.2 s | 9.4 / 6.6 s | 10.5 / 10.5 s |
| Mean LLM calls | 1.00 | 9.27 | 1.31 | 3.00 | 2.97 |
| Mean retrieval rounds | 1.00 | 1.71 | 1.00 | 1.75 | 1.98 |

### Paired significance (exact two-sided McNemar on the 165 answerable questions)

| Comparison | Only first correct | Only second correct | p |
|---|---|---|---|
| Standard vs **Multi-Agent v2** | 3 | **13** | **0.021** |
| Standard vs v2 without filtering | 5 | 13 | 0.096 |
| Standard vs v2 without retries | 12 | 4 | 0.077 |
| Standard vs v1 | 20 | 14 | 0.39 |
| v2 vs v2 without retries | 18 | 0 | < 0.001 |
| v2 vs v2 without filtering | 7 | 5 | 0.77 |
| v2 vs v1 | 23 | 7 | 0.005 |

Abstention on the 34 unanswerable questions: Standard and v2 differ on only 2 questions (1 : 1, p = 1.0).

### Findings (measured)

1. **Multi-Agent RAG v2 answered more answerable questions correctly than Standard RAG on held-out data**:
   73.9 % vs 67.9 % (+6.1 points, 13 vs 3 discordant questions, p = 0.021). The gain comes from fewer wrong
   abstentions (23 vs 32) while keeping the same correct-abstention rate on unanswerable questions (27/34 each) and a
   similar wrong-answer count (20 vs 21).
2. **Adaptive retrieval is the component that matters.** Without retries, the same pipeline is *below* Standard RAG
   (63.0 %; its round-1 verification turns "insufficient" drafts into abstentions); with retries it is above.
3. **The MAIN-RAG-style evidence filter adds little on this data**: v2 vs v2-without-filtering is 7 vs 5 (p = 0.77).
   It raises evidence precision (58.9 % vs 53.5 %) but costs ~6 extra LLM calls per question on average.
4. **The v2 verifier fixes the v1 failure**: v2 beats v1 by 23 vs 7 questions (p = 0.005); v1 again had the most
   wrong abstentions (48).
5. **Cost:** 59 % of questions (117/199) were answered in one round (1.4 LLM calls, 4.8 s on average); the 82
   escalated questions averaged 20.5 LLM calls and 30 s. Mean latency is 4.2× Standard RAG's.
6. Standard RAG's repeat consistency (17/20 identical answers) shows that Ollama generation is not perfectly
   deterministic even at temperature 0 with a fixed seed.

### What this does **not** show

* It is one dataset family, one 4B model and 199 questions; the confidence interval of a 6-point difference at this
  size is wide, and p = 0.021 is not a guarantee of the same gain elsewhere.
* "Unanswerable" means the answer is absent from the truncated evidence; correct abstention did not improve.
* Support/verified rates come from this project's own verifier; selective accuracy is *lower* for v2 than for v1 or
  the no-filter variant because v2 answers more of the hard questions.
* Results are not comparable with the MAIN-RAG paper's numbers (different model, retriever, corpus and sample).
