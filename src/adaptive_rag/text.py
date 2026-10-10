"""Small, dependency-free text helpers shared by retrieval, agents and metrics."""

from __future__ import annotations

import re
import string

STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has have
    having he her here hers herself him himself his how i if in into is it its itself just me more most my
    myself no nor not now of off on once only or other our ours ourselves out over own same she should so
    some such than that the their theirs them themselves then there these they this those through to too
    under until up very was we were what when where which while who whom why will with would you your
    yours yourself yourselves also may might must shall s t don didn doesn isn wasn weren tell explain
    describe give list show please according document documents text passage file""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")
_CITATION_RE = re.compile(r"\[(\d+)(?:\s*,\s*\d+)*\]")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def stem(token: str) -> str:
    """Very light suffix stripping so 'agents' matches 'agent'. Deliberately conservative."""
    for suffix in ("ies", "es", "s"):
        if len(token) > 4 and token.endswith(suffix) and not token.endswith("ss"):
            return token[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return token


def content_terms(text: str) -> list[str]:
    """Lower-cased, stemmed, stopword-free tokens (order preserved, duplicates kept)."""
    return [stem(t) for t in tokenize(text) if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


def keywords(text: str) -> list[str]:
    """Unique content terms in order of first appearance."""
    seen: dict[str, None] = {}
    for t in content_terms(text):
        seen.setdefault(t, None)
    return list(seen)


def numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER_RE.findall(text)}


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]


def citation_refs(text: str) -> list[int]:
    refs: list[int] = []
    for m in re.finditer(r"\[([\d,\s]+)\]", text):
        for part in m.group(1).split(","):
            part = part.strip()
            if part.isdigit() and int(part) not in refs:
                refs.append(int(part))
    return refs


def strip_citations(text: str) -> str:
    return re.sub(r"\s*\[[\d,\s]+\]", "", text).strip()


# ---- SQuAD/TriviaQA-style answer normalisation (used by metrics) ----
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


_FOCUS_PATTERNS = [
    re.compile(r"^(?:what|which)\s+(?:is|are|was|were)\s+(?:the\s+|a\s+|an\s+)?((?:[\w-]+\s+){0,3}[\w-]+?)\s+(?:of|for|in|on|at|does|do|did|that|between|to)\b"),
    re.compile(r"^(?:what|which)\s+((?:[\w-]+\s+){0,2}[\w-]+?)\s+(?:does|do|did|is|are|was|were|has|have|can)\b"),
    re.compile(r"^which\s+([\w-]+\s+[\w-]+)\s+[\w-]+(?:ed|s)\b"),
    re.compile(r"^how\s+many\s+((?:[\w-]+\s+){0,2}[\w-]+?)\s+(?:does|do|did|can|are|were|is|have|has)\b"),
]
_GENERIC_FOCUS = {"name", "type", "kind", "difference", "main", "thing", "way", "role", "purpose", "year"}


def focus_terms(question: str) -> list[str]:
    """Content terms naming *what is being asked for* (e.g. 'retail price' in 'What is the retail price of X?').

    A rule-based heuristic: returns [] when no pattern matches (who/when/where questions etc.).
    """
    q = question.strip().lower()
    for pat in _FOCUS_PATTERNS:
        m = pat.search(q)
        if m:
            terms = [t for t in keywords(m.group(1)) if t not in _GENERIC_FOCUS]
            if terms and len(terms) <= 4:
                return terms
    return []
