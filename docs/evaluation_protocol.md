# Evaluation Protocol

## 1. Two separate workflows

1. **User documents**: uploaded in the UI, stored in `data/runtime/kb/`. Used for demos; never used for metrics.
2. **Benchmarks**: fixed datasets in `data/datasets/<name>/`, each indexed into its **own** knowledge base
   (`data/runtime/benchmarks/<name>/`). Results from different datasets are never compared with each other.

## 2. Dataset format

```
data/datasets/<name>/
  dataset.json      name, title, description, source, license, synthetic (true/false)
  corpus.jsonl      {"doc_id", "title", "text"}                          one evidence document per line
  questions.jsonl   {"id", "question", "answers": [aliases], "answerable": bool, "gold_doc_ids": [...], "type"}
```

`python evaluation/validate_dataset.py` checks the files and prints the number of **usable** questions
(rows with an id, a question, answers when answerable, and gold documents that exist in the corpus).

## 3. Datasets

### 3.1 `smoke_test` — synthetic (committed)

- 8 short documents about **invented** entities (a town, a battery, a moth, a protocol, ...), written for this project.
- 24 questions: **18 answerable** (factoid, numeric, multi-part, comparison) and **6 unanswerable** (plausible
  questions whose answer is deliberately absent).
- Invented entities stop the LLM from answering from memory, so it tests grounding.
- Purpose: verify the pipeline end to end. **It is not a research benchmark** and the UI labels it "synthetic".

### 3.2 `triviaqa_unfiltered_sample` — research benchmark sample (prepared locally)

- Source: TriviaQA (Joshi et al., ACL 2017), https://nlp.cs.washington.edu/triviaqa/, accessed through the Hugging
  Face mirror `mandarjoshi/trivia_qa`, config `unfiltered`, split `validation`, in streaming mode (the 3 GB archive is
  not downloaded).
- Licence: the TriviaQA repository states Apache-2.0 for code and data; the website notes the authors do not own the
  copyright of the evidence documents (Wikipedia and web pages). The prepared files are **git-ignored** and not
  redistributed; anyone can re-create them with the script.
- Preparation (`python evaluation/prepare_triviaqa.py --n 100`):
  - the first 100 validation questions in dataset order;
  - each question keeps its Wikipedia entity pages (first 6,000 characters) and its first 3 web search results
    (first 3,000 characters). Truncation keeps the **leading** text and never looks for the answer, so it is unbiased;
  - all documents are pooled into one corpus (475 documents), so other questions' documents act as distractors;
  - gold aliases = TriviaQA `value` + `aliases`; `gold_doc_ids` = retained documents containing an alias;
  - `answerable` = at least one retained document contains an alias. In our sample **76** questions are
    answerable and **24** are not ("unanswerable given the evidence"), which lets us measure abstention.

## 4. Procedure

```powershell
python evaluation/run_benchmark.py --dataset <name> --mode real [--limit N --seed 0]
```

1. Load and validate the dataset; if `--limit` is given, take a fixed-seed random sample (kept in file order).
2. Build (or reuse) the benchmark index with the configured embedder and chunk settings.
3. For **each question**, run every selected method **in the same order**, with the same knowledge base, retriever,
   generator, prompt and default `top_k`. The LLM uses temperature 0 and a fixed seed.
4. Score each response, aggregate per method, and write `results.json` (configuration, model names, dataset
   manifest, question ids, per-question rows, summary, runtime) and `results.csv`.
5. Errors are recorded as failed rows and counted (`n_errors`); they are never dropped silently.
6. Results are written only after the run finishes successfully.

## 5. Metric definitions

Answer normalisation (SQuAD/TriviaQA style): lower-case, remove punctuation and the articles a/an/the, collapse spaces.
*Abstained* means the response status is not `answered`.

| Metric | Definition | Computed over |
|---|---|---|
| **Accuracy** | answered **and** some normalised gold alias occurs in the normalised answer as a whole-word sequence; abstention counts as wrong | answerable questions |
| Exact match | normalised answer == some normalised alias (strict; low for sentence-style answers, reported for completeness) | answerable |
| Token F1 | max over aliases of token-overlap F1 between answer and alias | answerable |
| Answer rate (answerable) | fraction of answerable questions that were answered | answerable |
| **Selective accuracy** | accuracy among the answered answerable questions | answered answerable |
| **Correct abstention** | fraction of unanswerable questions where the system abstained | unanswerable |
| **Wrong abstention** | fraction of answerable questions where the system abstained | answerable |
| Cites gold document | at least one valid citation points to a gold document | answered answerable questions with gold docs |
| Evidence recall | some retrieved passage contains a gold alias (retrieval quality) | answerable |
| Mean support score | mean verifier support score of answered questions | answered |
| Latency | wall-clock pipeline time per question (excludes Standard RAG's post-hoc verification) | all |
| LLM calls | answer generations + planner query reformulations | all |
| Retrieval attempts | 1 + number of adaptive retries | all |

## 6. Known measurement caveats

- Alias matching can miss correct paraphrases ("fourteen days" vs "14 days") and can accept an answer that mentions
  the alias in a wrong statement.
- **Mean support score uses this project's own verifier**, so it is not independent evidence of faithfulness and
  favours the method that optimises it (Adaptive). Treat it as descriptive.
- Unanswerable labels in the TriviaQA sample are derived from truncated evidence; the model could still know the
  answer from pre-training. Answering such a question correctly from memory is counted as "not abstaining".
- 24 or 100 questions give wide confidence intervals: with n = 76, one question is 1.3 percentage points.
- Confidence scores are not calibrated probabilities; no calibration study has been done.
- Dev mode results only test the software and must never be reported as experimental results (they are labelled).
