"""Answer generation backends. ONE shared generator instance serves every pipeline and agent role.

* OllamaLLM          - real mode: a local instruction-tuned model served by Ollama (GPU or CPU).
* ExtractiveGenerator - DEV MODE ONLY: deterministic sentence extraction, no language model at all.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

import httpx

from adaptive_rag.schemas import Evidence, GeneratedAnswer
from adaptive_rag.text import citation_refs, content_terms, focus_terms, keywords, split_sentences

INSUFFICIENT_ANSWER = "The provided documents do not contain enough information to answer this question."

ANSWER_SYSTEM_PROMPT = """You are a careful research assistant. Answer the question using ONLY the numbered \
evidence passages provided. Rules:
1. Every factual sentence must end with citations to the passages that support it, e.g. [1] or [2][3].
2. Do not use outside knowledge. Do not guess.
3. If the passages do not contain the answer, set "insufficient" to true and the answer to: \
"%s"
4. Be concise: one to three complete sentences (never just a bare name or number). Begin with the direct answer.
Return JSON with keys: answer (string), citations (list of passage numbers used), insufficient (boolean).""".replace(
    "%s", INSUFFICIENT_ANSWER
)

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "integer"}},
        "insufficient": {"type": "boolean"},
    },
    "required": ["answer", "citations", "insufficient"],
}

REWRITE_SYSTEM_PROMPT = """You help a retrieval system search a document collection. Given a question and \
what was missing from the previous attempt, write up to 3 short, different keyword search queries that \
could find the missing information. Return JSON: {"queries": ["...", "..."]}."""

REWRITE_SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
    "required": ["queries"],
}


class LLMError(RuntimeError):
    pass


def format_evidence(evidence: list[Evidence], max_chars: int = 1400) -> str:
    blocks = []
    for e in evidence:
        loc = f"{e.filename}" + (f", page {e.page}" if e.page else "")
        blocks.append(f"[{e.ref}] (source: {loc})\n{e.text[:max_chars]}")
    return "\n\n".join(blocks)


class Generator(Protocol):
    name: str
    is_llm: bool

    def answer(self, question: str, evidence: list[Evidence]) -> GeneratedAnswer: ...

    def rewrite_queries(self, question: str, missing: str) -> list[str]: ...


class OllamaLLM:
    is_llm = True

    def __init__(self, url: str, model: str, num_ctx: int, max_tokens: int, timeout_s: float,
                 num_gpu: int | None = None):
        self.url, self.model, self.name = url.rstrip("/"), model, f"ollama:{model}"
        self.options = {"temperature": 0.0, "seed": 42, "num_ctx": num_ctx, "num_predict": max_tokens}
        if num_gpu is not None:
            self.options["num_gpu"] = num_gpu
        self.timeout = timeout_s

    def status(self) -> dict:
        try:
            r = httpx.get(f"{self.url}/api/tags", timeout=3)
            r.raise_for_status()
            names = [m["name"] for m in r.json().get("models", [])]
            ok = self.model in names or f"{self.model}:latest" in names
            return {"ready": ok, "detail": "model available" if ok else
                    f"Ollama is running but '{self.model}' is not pulled (run: ollama pull {self.model})"}
        except httpx.HTTPError as exc:
            return {"ready": False, "detail": f"Ollama not reachable at {self.url}: {exc.__class__.__name__}"}

    def chat_json(self, system: str, user: str, schema: dict) -> dict:
        payload = {
            "model": self.model, "stream": False, "format": schema, "options": self.options,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        try:
            r = httpx.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        content = r.json().get("message", {}).get("content", "")
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", content, re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
            raise LLMError(f"Model did not return valid JSON: {content[:200]!r}") from None

    def answer(self, question: str, evidence: list[Evidence]) -> GeneratedAnswer:
        user = f"Evidence passages:\n\n{format_evidence(evidence)}\n\nQuestion: {question}"
        data = self.chat_json(ANSWER_SYSTEM_PROMPT, user, ANSWER_SCHEMA)
        text = str(data.get("answer", "")).strip()
        insufficient = bool(data.get("insufficient", False)) or not text
        cited = [int(c) for c in data.get("citations", []) if isinstance(c, (int, float))]
        for ref in citation_refs(text):  # inline markers count too
            if ref not in cited:
                cited.append(ref)
        if insufficient:
            text = INSUFFICIENT_ANSWER
            cited = []
        return GeneratedAnswer(answer=text, cited_refs=cited, insufficient=insufficient, raw=json.dumps(data))

    def rewrite_queries(self, question: str, missing: str) -> list[str]:
        data = self.chat_json(REWRITE_SYSTEM_PROMPT, f"Question: {question}\nMissing: {missing}", REWRITE_SCHEMA)
        return [q.strip() for q in data.get("queries", []) if isinstance(q, str) and q.strip()][:3]


class ExtractiveGenerator:
    """DEV MODE: picks the evidence sentences with the highest keyword overlap with the question.

    Not a language model. Its answers are copied from evidence, so verification scores in dev mode are
    optimistic by construction and must not be reported as model results.
    """

    is_llm = False
    name = "extractive-dev"

    def __init__(self, min_overlap: float = 0.5, max_sentences: int = 2):
        self.min_overlap, self.max_sentences = min_overlap, max_sentences

    def answer(self, question: str, evidence: list[Evidence]) -> GeneratedAnswer:
        q_terms = keywords(question)
        if not q_terms:
            return GeneratedAnswer(answer=INSUFFICIENT_ANSWER, cited_refs=[], insufficient=True)
        scored: list[tuple[float, int, int, str]] = []
        for e in evidence:
            for pos, sent in enumerate(split_sentences(e.text)):
                terms = set(content_terms(sent))
                overlap = sum(1 for t in q_terms if t in terms) / len(q_terms)
                scored.append((overlap, -e.ref, -pos, sent))
        focus = focus_terms(question)
        if focus:  # a sentence must mention what is being asked for (e.g. 'price')
            scored = [x for x in scored if set(focus) <= set(content_terms(x[3]))]
        scored.sort(reverse=True)
        chosen = [s for s in scored if s[0] >= self.min_overlap][: self.max_sentences]
        if not chosen:
            return GeneratedAnswer(answer=INSUFFICIENT_ANSWER, cited_refs=[], insufficient=True)
        parts, refs = [], []
        for _, neg_ref, _, sent in chosen:
            parts.append(f"{sent.rstrip()} [{-neg_ref}]")
            if -neg_ref not in refs:
                refs.append(-neg_ref)
        return GeneratedAnswer(answer=" ".join(parts), cited_refs=refs, insufficient=False)

    def rewrite_queries(self, question: str, missing: str) -> list[str]:
        return []  # dev mode relies on the planner's deterministic reformulation

