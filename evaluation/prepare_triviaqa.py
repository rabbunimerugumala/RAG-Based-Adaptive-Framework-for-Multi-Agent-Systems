"""Prepare a local TriviaQA-unfiltered sample in this project's dataset format.

Source: TriviaQA (Joshi et al., ACL 2017), https://nlp.cs.washington.edu/triviaqa/ , accessed through the
Hugging Face mirror ``mandarjoshi/trivia_qa`` (config ``unfiltered``) in streaming mode, so the full
~3 GB archive is not downloaded. The TriviaQA repository states Apache-2.0 for code and data, while the
website notes the authors do not own the copyright of the evidence documents (Wikipedia/web pages).
The prepared files are therefore written to a git-ignored folder and are NOT redistributed.

Evidence handling (deliberately unbiased): each question keeps its Wikipedia entity pages and its first
``--web-docs`` web search results, each truncated to its LEADING ``--wiki-chars`` / ``--web-chars``
characters. The answer location is never used for truncation. A question is labelled ``answerable`` only
if a gold alias occurs in its retained evidence; otherwise it is kept as an unanswerable-given-evidence
case (useful for measuring abstention). All questions share one pooled corpus, so other questions'
documents act as distractors.

Usage (PowerShell, from the project root):
    pip install -e ".[datasets]"
    python evaluation/prepare_triviaqa.py --n 100
    python evaluation/prepare_triviaqa.py --offset 100 --n 200 --out data/datasets/triviaqa_heldout_101_300
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_rag.evaluation.metrics import contains_answer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100, help="number of questions (taken in dataset order)")
    ap.add_argument("--offset", type=int, default=0,
                    help="skip this many questions first (e.g. 100 for a held-out set disjoint from the first 100)")
    ap.add_argument("--split", default="validation", choices=["train", "validation"])
    ap.add_argument("--wiki-chars", type=int, default=6000)
    ap.add_argument("--web-chars", type=int, default=3000)
    ap.add_argument("--web-docs", type=int, default=3)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "datasets" / "triviaqa_unfiltered_sample")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit('The "datasets" package is required: pip install -e ".[datasets]"')

    stream = load_dataset("mandarjoshi/trivia_qa", "unfiltered", split=args.split, streaming=True)
    corpus, questions = [], []
    seen = 0
    for row in stream:
        seen += 1
        if seen <= args.offset:
            continue
        if len(questions) >= args.n:
            break
        qid = row["question_id"]
        aliases = list(dict.fromkeys([row["answer"]["value"], *row["answer"]["aliases"]]))
        docs = []
        ep = row["entity_pages"]
        for i, (title, text) in enumerate(zip(ep["title"], ep["wiki_context"])):
            if text and text.strip():
                docs.append({"doc_id": f"{qid}_wiki{i}", "title": f"Wikipedia: {title}", "text": text[: args.wiki_chars]})
        sr = row["search_results"]
        for i, (title, text) in enumerate(zip(sr["title"], sr["search_context"])):
            if i >= args.web_docs:
                break
            if text and text.strip():
                docs.append({"doc_id": f"{qid}_web{i}", "title": f"Web: {title}"[:150], "text": text[: args.web_chars]})
        if not docs:
            continue
        gold = [d["doc_id"] for d in docs if contains_answer(d["text"], aliases)]
        corpus.extend(docs)
        questions.append({"id": qid, "question": row["question"], "answers": aliases, "answerable": bool(gold),
                          "gold_doc_ids": gold, "type": "trivia"})
        print(f"\r{len(questions)}/{args.n} questions, {len(corpus)} documents", end="", flush=True)
    print()

    args.out.mkdir(parents=True, exist_ok=True)
    n_ans = sum(q["answerable"] for q in questions)
    manifest = {
        "name": args.out.name,
        "title": (f"TriviaQA-unfiltered sample ({args.split}, first {len(questions)} questions)" if not args.offset else
                  f"TriviaQA-unfiltered held-out sample ({args.split}, questions {args.offset + 1}-{args.offset + len(questions)})"),
        "description": (f"{len(questions)} TriviaQA-unfiltered {args.split} questions with truncated evidence "
                        f"(wiki {args.wiki_chars} chars, top-{args.web_docs} web results {args.web_chars} chars). "
                        f"{n_ans} have a gold alias in the retained evidence (answerable); the rest test abstention."),
        "source": "https://nlp.cs.washington.edu/triviaqa/ via huggingface.co/datasets/mandarjoshi/trivia_qa (config: unfiltered)",
        "citation": "Joshi et al. TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension. ACL 2017.",
        "license": "Repository states Apache-2.0; evidence documents are third-party content. Local research use only - not redistributed.",
        "synthetic": False,
        "preparation": vars(args) | {"out": str(args.out.relative_to(ROOT))},
    }
    (args.out / "dataset.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for name, rows in (("corpus.jsonl", corpus), ("questions.jsonl", questions)):
        with (args.out / name).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(questions)} questions ({n_ans} answerable given evidence) and {len(corpus)} documents to {args.out}")


if __name__ == "__main__":
    main()
