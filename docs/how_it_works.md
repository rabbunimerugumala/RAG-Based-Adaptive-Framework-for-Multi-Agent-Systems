# How It Works — a plain-language guide

This page explains the whole project step by step, assuming only basic computer-science knowledge.

---

## 1. The problem

Large Language Models (LLMs) such as ChatGPT or Qwen write fluent answers, but:

1. they **do not know your private documents** (college rules, a company manual, your notes), and
2. they sometimes **make things up** ("hallucinate") with full confidence.

## 2. What is RAG?

**Retrieval-Augmented Generation (RAG)** fixes problem 1:

```
Your question ──► search your documents ──► give the best passages to the LLM ──► LLM writes the answer
```

The LLM is told: *"answer only from these passages and cite them"*. This is **Standard RAG**, our baseline.

Standard RAG still has weaknesses:

- If the search returns the wrong passages, the LLM may still answer (wrongly).
- Nobody checks whether each sentence of the answer is really supported by the passages.
- It never says "I don't know" unless the LLM itself decides to.

## 3. Our idea: Adaptive Multi-Agent RAG

We split the work among **four agents**. Each is a small, separate Python class with one clear job.
They all share **one** local LLM (loaded once), and only two steps actually use it.

| # | Agent | Job in one line | Uses the LLM? |
|---|---|---|---|
| 1 | **Planner** | Understand the question and plan the search | only on a retry (to suggest new search queries) |
| 2 | **Evidence** | Find the best passages (keyword + meaning search) | no |
| – | *Answer generator* | Write a draft answer with citations | **yes** |
| 3 | **Verifier** | Check every claim in the draft against the passages | no (uses a small NLI model) |
| 4 | **Confidence** | Score the answer and decide: accept, retry, or abstain | no |

```
             ┌───────────────────────────── retry (max 2) ─────────────────────────────┐
             ▼                                                                         │
Question → Planner → Evidence → Draft answer → Verifier → Confidence ── accept ──► Answer + citations
                                                                    └── abstain ──► "Not enough evidence"
```

## 4. Before any question: indexing documents

When you upload a file on the **Document Library** page:

1. **Validate**: only PDF/DOCX/TXT/MD, max 20 MB; the filename is cleaned (no `../` tricks).
2. **Duplicate check**: a SHA-256 hash of the content; identical files are not indexed twice.
3. **Extract text**: `pypdf` for PDFs (per page), `python-docx` for Word (paragraphs + tables), plain decode for TXT/MD.
   Scanned PDFs (images only) are rejected with a message, because OCR is not implemented.
4. **Chunk**: split into pieces of about 900 characters with 150 characters of overlap, cutting at sentence or
   paragraph ends where possible. Chunks never cross PDF pages, so every citation keeps its page number.
5. **Embed**: each chunk becomes a 384-number vector (its "meaning") using the `bge-small-en-v1.5` model.
6. **Store**: vectors go into **Qdrant** (a vector database running inside our Python process); the text is also
   added to a **BM25** keyword index.

## 5. Walk-through of one real question

Question (asked against the sample file `college_library_rules.txt`):

> **"How many books can an undergraduate student borrow and for how long?"**

### Step 1 — Planner Agent (rules, no LLM)

- Is it small talk like "hello"? No, so retrieval is needed.
- **Type**: `numeric` (it starts with "how many"), using simple patterns (comparison, multi-part, numeric, definition, ...).
- **Keywords**: `many, book, undergraduate, student, borrow, long` (stop-words removed, light stemming).
- **Focus terms**: what is actually being asked for: here `book` (in "What is the retail price of X?" it would be `retail, price`).
- **Sub-questions**: comparison and two-part questions are split so each part gets its own search.
- **Plan**: search queries = the question itself + a keyword-only query, `top_k = 5`.

### Step 2 — Evidence Agent (no LLM)

For every planned query it runs **two searches**:

- **Dense search** in Qdrant: finds chunks with similar *meaning* (cosine similarity of vectors).
- **BM25 search**: finds chunks containing the same *words*, weighted by how rare the words are.

The ranked lists are merged with **Reciprocal Rank Fusion**: each chunk gets `Σ 1/(60 + rank)` over all lists.
A chunk ranked high by both methods wins. The top 5 passages are numbered `[1]...[5]` with filename, page and chunk id.

> Important: a high retrieval score means *"related to the question"*, not *"the claim is true"*.

### Step 3 — Draft answer (the only LLM call in a normal run)

The LLM (Qwen3-4B, running in Ollama) receives the numbered passages and strict rules: use only these passages,
cite `[n]`, and say "insufficient" if the answer is not there. It returns JSON:

```json
{"answer": "An undergraduate student can borrow up to 4 books for a period of 14 days. [1]",
 "citations": [1], "insufficient": false}
```

### Step 4 — Verifier Agent (no LLM)

1. Split the answer into **claims** (sentences; compound sentences such as "X ..., while Y ..." are split further).
2. For each claim compute:
   - **Lexical support**: what fraction of the claim's important words appear in a passage. Any **number** in the claim
     must appear in that passage (so "5 books" vs "4 books" fails).
   - **NLI entailment**: a small Natural Language Inference model (`nli-deberta-v3-xsmall`) reads
     *passage → claim* and outputs probabilities for *entailment / contradiction / neutral*.
3. Label each claim: **supported** (NLI entailment ≥ 0.5), **contradicted** (NLI contradiction ≥ 0.6 on a related
   passage), **partially supported** (words match but NLI does not confirm), or **unsupported**.

Result for our example: 1/1 claims supported, citation consistent.

### Step 5 — Confidence Agent (no LLM)

It combines four numbers, each between 0 and 1:

```
C = 0.25 × relevance + 0.25 × coverage + 0.35 × verification + 0.15 × consistency
```

| Component | Meaning |
|---|---|
| relevance | how well the top 3 passages match the question's keywords |
| coverage | are all keywords / focus terms / sub-questions found in the evidence? |
| verification | supported = 1, partially = 0.5, unsupported = 0 (averaged over claims) |
| consistency | do the cited passages really support the claims they are attached to? |

Decision rules (thresholds are configurable experiment settings):

- **accept** if C ≥ 0.65 and the verifier says supported / partially supported;
- otherwise **retry** (if retries are left) — back to the Planner with a strategy:
  - *broaden* (weak retrieval): more candidates, a larger top-k, relaxed keyword queries;
  - *targeted* (unsupported claims): search specifically for the unsupported claims and missing keywords;
    with a real LLM, the Planner also asks it for up to 3 reformulated search queries;
- after the 2nd retry, take the best attempt: **accept** if C ≥ 0.40 and it is at least partially supported,
  otherwise **abstain**.

Our example scored **C = 0.83 → accept** on the first attempt (1 LLM call, about 5 s).

### Step 6 — What you see

The UI shows the answer with clickable citations `[1]`, the evidence passages and their scores, the verifier's
claim table, the confidence components, and the **Agent Trace** page with every step's real input/output and timing.

## 6. Example of abstention

> **"What is the WiFi password of the library?"**

The documents never mention WiFi. The LLM reports "insufficient", the Confidence Agent retries twice with broader and
targeted searches, still finds nothing, and the system **abstains** (3 attempts, about 19 s) instead of guessing.

## 7. How do we know if it is better? (Evaluation)

Both methods answer **the same questions** from **the same document collection**, and we measure accuracy,
correct abstention on unanswerable questions, wrong abstention on answerable ones, citation correctness,
latency and number of LLM calls. See [evaluation_protocol.md](evaluation_protocol.md) and the actual numbers in
[results.md](results.md).

## 8. Key takeaways

- RAG grounds answers in documents; our framework adds **checking** (Verifier), **scoring** (Confidence) and
  **adaptive behaviour** (retry/abstain).
- The agents are real, separate modules with different responsibilities, not four names for one LLM call.
- Everything is local and free: Ollama, Qdrant local mode, fastembed, Hugging Face transformers.
- The trade-off: extra checking costs extra time and LLM calls. In our measurements it did **not** improve accuracy on
  TriviaQA, because the verifier wrongly rejected some short correct answers; see [results.md](results.md). Knowing
  and explaining this honestly is part of the research result.
