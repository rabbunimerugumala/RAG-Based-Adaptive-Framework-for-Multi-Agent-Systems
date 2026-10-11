# Evaluation Protocol

## 1. Separation of data

| Role | Dataset | Use |
|---|---|---|
| Smoke test | `smoke_test` (24 synthetic questions, committed) | checks that the pipeline runs; **not** a benchmark |
| Development / validation | `triviaqa_unfiltered_sample` = TriviaQA-unfiltered validation questions 1–100 | the **original baseline** (run `20261010-223619-6d2293`, kept unchanged) and all design decisions for v2 |
| Held-out test | `triviaqa_heldout_101_300` = validation questions 101–300, minus 1 question shared with the development sample → **199 questions** | run **once**, after the v2 code was frozen; never used for tuning |

User-uploaded documents (`data/runtime/kb/`) are never used for evaluation; each benchmark corpus has its own index
(`data/runtime/benchmarks/<dataset>/`). Results from different datasets are never compared with each other.

## 2. Datasets

### 2.1 TriviaQA-unfiltered samples (research benchmark samples)

* Source: TriviaQA (Joshi et al., ACL 2017), https://nlp.cs.washington.edu/triviaqa/, via the Hugging Face mirror
  `mandarjoshi/trivia_qa`, config `unfiltered`, split `validation`, streamed (the full archive is not downloaded).
* Licence: the TriviaQA repository states Apache-2.0; the evidence documents are third-party web/Wikipedia text.
  Prepared files are git-ignored and not redistributed; the scripts below re-create them.
* Evidence per question: Wikipedia entity pages (first 6,000 characters) + first 3 web results (first 3,000
  characters). Truncation keeps the leading text and never looks for the answer. All questions' documents form one
  pooled corpus, so other questions' documents act as distractors.
* `answerable` = a gold alias (TriviaQA `value` + `aliases`) occurs in at least one retained document;
  `gold_doc_ids` = those documents. Unanswerable-given-evidence questions measure abstention.
* Counts: development 100 (76 answerable / 24 not), 475 documents; held-out 199 (165 / 34), 978 documents.

```powershell
python evaluation/prepare_triviaqa.py --n 100                                          # development
python evaluation/prepare_triviaqa.py --offset 100 --n 200 --out data/datasets/triviaqa_heldout_101_300 `
    --exclude-dataset data/datasets/triviaqa_unfiltered_sample                          # held-out test
python evaluation/validate_dataset.py
```

### 2.2 Smoke test

8 fictional documents, 24 questions (18 answerable, 6 unanswerable). Verifies grounding without memorised knowledge.

## 3. Systems compared

All systems share the knowledge base, retriever (bge-small + BM25 + RRF), generator (Qwen3-4B Q4 via Ollama,
temperature 0, seed 42), prompt, NLI model, evidence budget (top-k = 5), questions and question order.

| Name | System | Differences from the full system |
|---|---|---|
| `standard` | Standard RAG baseline | single retrieval (1 query, top-5), single draft, no verification in the decision |
| `adaptive` | **Multi-Agent RAG v2 (proposed)** | – |
| `adaptive_no_retry` | ablation | `max_retries = 0` (no additional retrieval) |
| `adaptive_no_filter` | ablation | `doc_filter = off` (retries without judged evidence filtering) |
| `adaptive_v1` | original pipeline | `config.LEGACY_V1`: no filtering, v1 verifier, v1 confidence |

Unavoidable differences: Multi-Agent RAG searches with 2 query variants in round 1 (question + keywords) and may
retrieve more candidates on retries; this is part of the method being evaluated.

## 4. Procedure

```powershell
python evaluation/run_benchmark.py --dataset triviaqa_heldout_101_300 --mode real `
    --methods standard,adaptive,adaptive_no_retry,adaptive_no_filter,adaptive_v1 --repeat 20
python evaluation/analyze_results.py <run_id>
```

1. Load and validate the dataset; build or reuse its separate index.
2. For each question, run every system in the same order (interleaved, so they share warm-up and hardware state).
3. Score each response; errors become failed rows (`n_errors`), never dropped.
4. `--repeat N` re-runs the first N questions per system to measure response consistency.
5. Write `results.json` (configuration, variant settings, model names, software version + git commit, dataset
   manifest, question ids, per-question rows with per-attempt drafts and decisions, summaries, consistency) and
   `results.csv`.

## 5. Metric definitions

Normalisation (SQuAD/TriviaQA): lower-case, remove punctuation and a/an/the. *Abstained* = status is not `answered`.

| # | Metric | Definition | Over |
|---|---|---|---|
| 1 | **Accuracy** | answered and a normalised gold alias occurs in the normalised answer (whole words); abstention counts as wrong | answerable |
| 2 | Evidence relevance – **precision** | mean share of passages given to the generator that contain a gold alias | answerable |
| 2b | Evidence relevance – **recall** | some passage given to the generator contains a gold alias | answerable |
| 3 | **Grounded answers** (faithfulness, verifier-independent) | ≥ 80 % of the answer's content terms occur in its cited passages (all passages if none cited) | answered |
| 3b | Mean support score / verified rate | from this project's own verifier – descriptive only, not independent | answered |
| 4 | **Citation correctness** | at least one valid citation points to a gold document | answered answerable with gold docs |
| 5 | Answer completeness | not applicable to TriviaQA (single-answer questions) – not reported | – |
| 6 | **Response consistency** | share of repeat runs whose normalised answer equals the first run's (and same status) | first N questions |
| 7 | **Latency** | wall-clock pipeline time per question; reported as mean, median, cold (first question) and warm mean | all |
| 8 | **LLM calls** | drafts + judge calls + per-passage predictions + query rewrites + verifier judge calls | all |
| 9 | **Retrieval rounds** | 1 + adaptive retries | all |
| 10 | **Abstention rate** | abstained / all | all |
| 11 | **False (wrong) abstention rate** | abstained / answerable | answerable |
| 11b | Correct abstention rate | abstained / unanswerable | unanswerable |
| 12 | **Unsupported-answer rate** | answered but not correct (wrong answers to answerable + any answer to unanswerable) / answered | answered |
| – | Selective accuracy | correct / answered answerable | answered answerable |
| – | Exact match, token F1 | strict string metrics, reported for completeness | answerable |

## 6. Statistics

* Paired comparisons on the same questions use the **exact two-sided McNemar test** (binomial test with p = 0.5 on
  the discordant pairs; questions both systems got right, or both wrong, carry no information about the difference).
  Assumptions: binary outcome per question, questions independent.
* Reported with the discordant counts, never alone. No superiority is claimed unless p < 0.05 on the held-out
  test, and even then only for this model, corpus and sample.
* `evaluation/analyze_results.py` recomputes all metrics from the rows and runs the tests (also across two runs).

## 7. Caveats

* Alias matching misses paraphrases and can accept an alias used in a wrong statement.
* Support score and "verified" use the project's own verifier; the grounded-answer metric is lexical.
* Unanswerable labels come from truncated evidence; a model may still know the answer from pre-training.
* The first question of a run includes model warm-up (reported as cold latency).
* Confidence scores are heuristics; no calibration study was performed.
