"""Agent 1 - Planner.

Deterministic rules decide whether retrieval is needed, classify the query, extract keywords, derive
sub-questions and set retrieval parameters. On a retry, the planner builds a *different* plan using the
Confidence Agent's retry strategy; only then (and only with a real LLM) is the shared model used to
suggest reformulated search queries.
"""

from __future__ import annotations

import re

from adaptive_rag.config import Settings
from adaptive_rag.generation.llm import Generator, LLMError
from adaptive_rag.schemas import Evidence, RetrievalPlan, VerificationResult
from adaptive_rag.text import content_terms, focus_terms, keywords

_SMALL_TALK = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|good (morning|evening|afternoon)|who are you|what can you do|bye)\b[\s!.?]*$",
    re.I,
)
_WH = r"(what|who|whom|which|when|where|why|how|is|are|does|do|did|can|list|name)"


def classify(query: str) -> str:
    q = query.lower()
    if re.search(r"\b(compare|comparison|difference|differ|versus|vs\.?|contrast)\b", q):
        return "comparison"
    if q.count("?") > 1 or re.search(rf"\band\s+{_WH}\b", q):
        return "multi_part"
    if re.search(r"\b(how many|how much|when|what year|what date|how long|percentage|number of)\b", q):
        return "numeric"
    if re.search(r"^\s*(what is|what are|define|definition of|meaning of)\b", q):
        return "definition"
    if re.search(r"\b(how (to|do|does|can)|steps|procedure|process)\b", q):
        return "procedural"
    return "factoid"


_PRONOUN_START = re.compile(r"^\s*(it|this|that|they|he|she|these|those|its|their)\b", re.I)


def route_question(query: str, qtype: str, subs: list[str], kws: list[str]) -> tuple[str, str]:
    """Rule-based question analysis (no LLM call): simple, complex or ambiguous, with a readable reason."""
    if len(kws) < 2:
        return "ambiguous", "too few content words to search reliably"
    if _PRONOUN_START.match(query) and not re.search(r"\b[A-Z][a-z]+", query[1:]):
        return "ambiguous", "starts with a pronoun and names no entity"
    if qtype in ("comparison", "multi_part") or subs:
        return "complex", f"{qtype.replace('_', '-')} question"
    n_words = len(query.split())
    if n_words > 20:
        return "complex", f"long question ({n_words} words)"
    return "simple", "single factual question"


def subquestions(query: str, qtype: str) -> list[str]:
    q = query.strip()
    if qtype == "comparison":
        m = re.search(r"between\s+(.+?)\s+and\s+(.+?)[?.!]*$", q, re.I) or re.search(
            r"compare\s+(.+?)\s+(?:and|with|to|vs\.?|versus)\s+(.+?)[?.!]*$", q, re.I
        ) or re.search(r"(.+?)\s+(?:vs\.?|versus)\s+(.+?)[?.!]*$", q, re.I)
        if m:
            return [f"What is {m.group(1).strip()}?", f"What is {m.group(2).strip()}?"]
    if qtype == "multi_part":
        parts = [p.strip() for p in re.split(r"\?\s*|\band\s+(?=" + _WH + r"\b)", q, flags=re.I) if p and p.strip()]
        parts = [p for p in parts if len(content_terms(p)) >= 1 and not re.fullmatch(_WH, p, re.I)]
        if len(parts) > 1:
            return [p.rstrip("?") + "?" for p in parts]
    return []


class PlannerAgent:
    def __init__(self, settings: Settings, generator: Generator):
        self.settings = settings
        self.generator = generator

    def plan(self, query: str) -> RetrievalPlan:
        s = self.settings
        if _SMALL_TALK.match(query) or not keywords(query):
            return RetrievalPlan(
                query=query, needs_retrieval=False, query_type="small_talk", keywords=[], subquestions=[],
                search_queries=[], top_k=0, candidate_k=0, strategy="none",
                explanation="Greeting/small talk with no information need - no document retrieval required.",
            )
        qtype = classify(query)
        kws = keywords(query)
        subs = subquestions(query, qtype)
        top_k = min(s.top_k + (2 if subs else 0), 10)
        queries = [query, *subs]
        kw_query = " ".join(kws)
        if len(kws) >= 2 and kw_query.lower() != query.lower():
            queries.append(kw_query)
        pool_k = max(s.doc_pool_k, top_k) if s.doc_filter == "judge" and s.doc_filter_stage == "always" else 0
        route, route_reason, budget = "full", "question routing disabled", s.max_retries
        if s.question_routing:
            route, route_reason = route_question(query, qtype, subs, kws)
            budget = min(s.simple_route_retries, s.max_retries) if route == "simple" else s.max_retries
        expl = f"Query classified as '{qtype}'. Hybrid search with {len(queries)} query variant(s), evidence budget top_k={top_k}."
        if pool_k:
            expl += f" Evidence Agent will judge {pool_k} candidate passages and keep those above the adaptive bar."
        elif s.doc_filter == "judge":
            expl += " Evidence filtering is held back and used only if more evidence is needed."
        if subs:
            expl += f" Decomposed into {len(subs)} sub-question(s) so each part gets its own evidence."
        return RetrievalPlan(
            query=query, needs_retrieval=True, query_type=qtype, keywords=kws, focus_terms=focus_terms(query),
            subquestions=subs,
            search_queries=queries, top_k=top_k, candidate_k=s.candidate_k, pool_k=pool_k, strategy="initial",
            explanation=expl + (f" Route: {route} ({route_reason}); up to {budget} escalation round(s) with the full "
                                f"four-agent workflow." if s.question_routing else ""),
            route=route, route_reason=route_reason, retry_budget=budget,
        )

    def _retry_pool(self, previous: RetrievalPlan, strategy: str) -> int:
        """Evidence budget for a retry: escalate to (or widen) judge-based filtering of a larger candidate pool."""
        s = self.settings
        if s.doc_filter != "judge":
            return 0
        extra = 5 if strategy == "broaden" else 0
        base = previous.pool_k or max(s.doc_pool_k, previous.top_k)
        return min(base + extra, 30)

    def replan(
        self, previous: RetrievalPlan, strategy: str, evidence: list[Evidence],
        verification: VerificationResult | None, attempt: int,
    ) -> tuple[RetrievalPlan, int]:
        """Return a new plan for the given retry strategy and the number of LLM calls it used."""
        seen_terms: set[str] = set()
        for e in evidence:
            seen_terms.update(content_terms(e.text))
        uncovered = [k for k in previous.keywords if k not in seen_terms]
        new_queries: list[str] = []
        notes: list[str] = []
        llm_calls = 0

        if strategy == "targeted" and verification and verification.unsupported_claims:
            for claim in verification.unsupported_claims[:2]:
                new_queries.append(" ".join(keywords(claim)[:12]))
            notes.append(f"searching for evidence on {len(verification.unsupported_claims[:2])} unsupported claim(s)")
        if uncovered:
            new_queries.append(" ".join(uncovered + [k for k in previous.keywords if k not in uncovered][:3]))
            notes.append(f"targeting uncovered keywords {uncovered}")
        if strategy == "broaden":
            # individual keyword pairs relax the conjunction implied by the full question
            kws = previous.keywords
            for i in range(0, max(len(kws) - 1, 0), 2):
                new_queries.append(" ".join(kws[i : i + 2]))
            notes.append("broadening: more candidates and relaxed keyword sub-queries")
        if self.generator.is_llm:
            missing = "; ".join(verification.unsupported_claims[:2]) if verification and verification.unsupported_claims else (
                f"keywords not found: {', '.join(uncovered)}" if uncovered else "the answer was not well supported")
            try:
                llm_calls = 1
                rewrites = self.generator.rewrite_queries(previous.query, missing)
                new_queries.extend(rewrites)
                notes.append(f"LLM suggested {len(rewrites)} reformulation(s)")
            except LLMError as exc:
                notes.append(f"LLM reformulation failed ({exc}); using rule-based queries only")

        queries: list[str] = []
        for q in [*previous.search_queries, *new_queries]:
            q = q.strip()
            if q and q.lower() not in {x.lower() for x in queries}:
                queries.append(q)
        bump = 3 if strategy == "broaden" else 2
        plan = previous.model_copy(update={
            "search_queries": queries,
            "top_k": min(previous.top_k + bump, 12),
            "candidate_k": min(previous.candidate_k * (2 if strategy == "broaden" else 1), 100),
            "pool_k": self._retry_pool(previous, strategy),
            "strategy": f"retry-{attempt}:{strategy}",
            "explanation": f"Retry {attempt} ({strategy}): " + "; ".join(notes or ["re-running with a larger top_k"]) + ".",
        })
        return plan, llm_calls
