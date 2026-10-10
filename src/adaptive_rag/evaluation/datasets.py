"""Benchmark dataset loading and validation.

A dataset is a directory ``data/datasets/<name>/`` containing::

    dataset.json     manifest: name, title, description, source, license, synthetic (bool)
    corpus.jsonl     one evidence document per line: {"doc_id", "title", "text"}
    questions.jsonl  one question per line: {"id", "question", "answers": [aliases], "answerable": bool,
                     "gold_doc_ids": [doc ids that contain the answer], "type": str}

Benchmark corpora are indexed into their own knowledge base, never into the user's document library.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Dataset:
    name: str
    manifest: dict
    corpus: list[dict]
    questions: list[dict]
    path: Path


@dataclass
class ValidationReport:
    name: str
    ok: bool
    num_documents: int = 0
    num_questions: int = 0
    num_usable: int = 0
    num_answerable: int = 0
    num_unanswerable: int = 0
    issues: list[str] = field(default_factory=list)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path.name} line {n}: invalid JSON ({exc})") from exc
    return rows


def list_datasets(datasets_dir: Path) -> list[dict]:
    out = []
    if not datasets_dir.exists():
        return out
    for d in sorted(datasets_dir.iterdir()):
        mf = d / "dataset.json"
        if d.is_dir() and mf.exists():
            manifest = json.loads(mf.read_text(encoding="utf-8"))
            report = validate_dataset(d)
            out.append({**manifest, "name": d.name, "num_questions": report.num_usable,
                        "num_documents": report.num_documents, "num_answerable": report.num_answerable,
                        "num_unanswerable": report.num_unanswerable, "valid": report.ok, "issues": report.issues[:5]})
    return out


def validate_dataset(path: Path) -> ValidationReport:
    rep = ValidationReport(name=path.name, ok=False)
    for fname in ("dataset.json", "corpus.jsonl", "questions.jsonl"):
        if not (path / fname).exists():
            rep.issues.append(f"missing {fname}")
    if rep.issues:
        return rep
    try:
        corpus = _read_jsonl(path / "corpus.jsonl")
        questions = _read_jsonl(path / "questions.jsonl")
    except ValueError as exc:
        rep.issues.append(str(exc))
        return rep
    doc_ids = set()
    for i, d in enumerate(corpus):
        if not d.get("doc_id") or not isinstance(d.get("text"), str) or not d["text"].strip():
            rep.issues.append(f"corpus row {i + 1}: needs non-empty doc_id and text")
        elif d["doc_id"] in doc_ids:
            rep.issues.append(f"corpus row {i + 1}: duplicate doc_id {d['doc_id']}")
        else:
            doc_ids.add(d["doc_id"])
    rep.num_documents = len(doc_ids)
    seen = set()
    for i, q in enumerate(questions):
        problems = []
        if not q.get("id") or q["id"] in seen:
            problems.append("missing or duplicate id")
        if not isinstance(q.get("question"), str) or len(q["question"].strip()) < 3:
            problems.append("missing question")
        if not isinstance(q.get("answerable"), bool):
            problems.append("'answerable' must be true/false")
        if q.get("answerable") and not q.get("answers"):
            problems.append("answerable question without answers")
        missing = [g for g in q.get("gold_doc_ids", []) if g not in doc_ids]
        if missing:
            problems.append(f"gold_doc_ids not in corpus: {missing[:3]}")
        if problems:
            rep.issues.append(f"question row {i + 1}: " + "; ".join(problems))
            continue
        seen.add(q["id"])
        rep.num_usable += 1
        rep.num_answerable += bool(q["answerable"])
    rep.num_questions = len(questions)
    rep.num_unanswerable = rep.num_usable - rep.num_answerable
    rep.ok = rep.num_usable > 0 and rep.num_documents > 0
    return rep


def load_dataset(datasets_dir: Path, name: str) -> Dataset:
    path = (datasets_dir / name).resolve()
    if path.parent != datasets_dir.resolve() or not path.is_dir():
        raise FileNotFoundError(f"Unknown dataset '{name}'")
    report = validate_dataset(path)
    if not report.ok:
        raise ValueError(f"Dataset '{name}' is invalid: {report.issues[:3]}")
    manifest = json.loads((path / "dataset.json").read_text(encoding="utf-8"))
    corpus = _read_jsonl(path / "corpus.jsonl")
    doc_ids = {d["doc_id"] for d in corpus}
    questions = []
    seen = set()
    for q in _read_jsonl(path / "questions.jsonl"):
        ok = (q.get("id") and q["id"] not in seen and isinstance(q.get("question"), str)
              and isinstance(q.get("answerable"), bool) and (q.get("answers") or not q["answerable"])
              and all(g in doc_ids for g in q.get("gold_doc_ids", [])))
        if ok:
            seen.add(q["id"])
            questions.append(q)
    return Dataset(name=name, manifest=manifest, corpus=corpus, questions=questions, path=path)
