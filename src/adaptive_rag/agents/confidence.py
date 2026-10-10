"""Agent 4 - Confidence. Combines four normalised signals into a transparent heuristic score and decides
whether to ACCEPT the answer, RETRY retrieval (at most ``max_retries`` times) or ABSTAIN.

    C = (w_r*relevance + w_c*coverage + w_v*verification + w_k*consistency) / (sum of weights of AVAILABLE components)

  relevance    - mean over the top-3 passages of passage relevance (keyword overlap with the query;
                 averaged with sigmoid(rerank score) when a reranker is enabled).
  coverage     - fraction of query keywords found anywhere in the evidence; if the planner found focus
                 terms (what is asked for, e.g. 'retail price'), averaged with the best single-passage
                 focus-term overlap; with sub-questions, further averaged with the fraction of
                 sub-questions whose keywords are >= 50% covered.
  verification - Verifier support score (supported=1, partial=0.5, unsupported/contradicted=0).
  consistency  - Verifier citation consistency (claims whose citations point to supporting passages).

Missing components (e.g. verification when the generator declined to answer) are listed explicitly
and excluded from the weighted mean; an answer can never be accepted without a verification result.
C is a heuristic, not a calibrated probability. Thresholds are experimental settings.
"""

from __future__ import annotations

from dataclasses import dataclass

from adaptive_rag.config import Settings
from adaptive_rag.retrieval.hybrid import keyword_overlap, sigmoid
from adaptive_rag.schemas import (
    ConfidenceAssessment,
    Evidence,
    GeneratedAnswer,
    RetrievalPlan,
    VerificationResult,
)
from adaptive_rag.text import content_terms, keywords


@dataclass
class AttemptRecord:
    attempt: int
    plan: RetrievalPlan
    evidence: list[Evidence]
    answer: GeneratedAnswer
    verification: VerificationResult
    confidence: ConfidenceAssessment


class ConfidenceAgent:
    def __init__(self, settings: Settings):
        self.s = settings

    def components(self, plan: RetrievalPlan, evidence: list[Evidence], verification: VerificationResult) -> dict:
        relevance = coverage = None
        if evidence:
            rels = []
            for e in evidence[:3]:
                r = e.keyword_overlap
                if e.rerank_score is not None:
                    r = (r + sigmoid(e.rerank_score)) / 2
                rels.append(r)
            relevance = sum(rels) / len(rels)
            if plan.keywords:
                terms: set[str] = set()
                for e in evidence:
                    terms.update(content_terms(e.text))
                coverage = sum(1 for k in plan.keywords if k in terms) / len(plan.keywords)
                if plan.focus_terms:  # is the thing being asked for mentioned in any single passage?
                    focus_cov = max(keyword_overlap(plan.focus_terms, e.text) for e in evidence)
                    coverage = (coverage + focus_cov) / 2
                if plan.subquestions:
                    joined = " ".join(e.text for e in evidence)
                    sub_cov = sum(keyword_overlap(keywords(q), joined) >= 0.5 for q in plan.subquestions)
                    coverage = (coverage + sub_cov / len(plan.subquestions)) / 2
        return {
            "relevance": relevance,
            "coverage": coverage,
            "verification": verification.support_score,
            "consistency": verification.citation_consistency,
        }

    def score(self, comps: dict) -> tuple[float | None, list[str]]:
        w = self.weights
        missing = [k for k, v in comps.items() if v is None]
        avail = {k: v for k, v in comps.items() if v is not None and w[k] > 0}
        total = sum(w[k] for k in avail)
        if not avail or total == 0:
            return None, missing
        return round(sum(w[k] * v for k, v in avail.items()) / total, 4), missing

    @property
    def weights(self) -> dict[str, float]:
        s = self.s
        return {"relevance": s.w_relevance, "coverage": s.w_coverage,
                "verification": s.w_verification, "consistency": s.w_consistency}

    def assess(
        self, plan: RetrievalPlan, evidence: list[Evidence], answer: GeneratedAnswer,
        verification: VerificationResult, retries_left: int,
    ) -> ConfidenceAssessment:
        s = self.s
        comps = self.components(plan, evidence, verification)
        score, missing = self.score(comps)
        rounded = {k: (None if v is None else round(v, 4)) for k, v in comps.items()}
        weak_retrieval = (comps["relevance"] or 0) < 0.5 or (comps["coverage"] or 0) < 0.5
        strategy = "broaden" if weak_retrieval else "targeted"

        def out(decision, reason, strat=None):
            return ConfidenceAssessment(score=score, components=rounded, weights=self.weights,
                                        missing_components=missing, decision=decision, reason=reason,
                                        retry_strategy=strat)

        if not evidence:
            return (out("retry", "No evidence retrieved; broadening the search.", "broaden") if retries_left
                    else out("abstain", "No relevant evidence was found in the knowledge base."))
        if answer.insufficient:
            return (out("retry", "Generator reported the evidence is insufficient; retrieving more.", strategy)
                    if retries_left else
                    out("abstain", "Evidence remained insufficient after the maximum number of retries."))
        if verification.status == "contradicted":
            return (out("retry", "Verifier found claim(s) contradicted by the evidence.", "targeted")
                    if retries_left else
                    out("abstain", "Answer contains claims contradicted by the evidence."))
        if (score is not None and score >= s.accept_threshold
                and verification.status in ("supported", "partially_supported")
                and (verification.support_score or 0) >= 0.5):
            return out("accept", f"Confidence {score:.2f} >= accept threshold {s.accept_threshold} "
                                 f"and verification is '{verification.status}'.")
        if retries_left:
            why = f"Confidence {score:.2f} < accept threshold {s.accept_threshold}" if score is not None else \
                "Confidence could not be computed"
            if verification.unsupported_claims:
                why += f"; {len(verification.unsupported_claims)} unsupported claim(s)"
            return out("retry", why + f". Retry strategy: {strategy}.", strategy)
        return self._fallback(score, verification, out)

    def _fallback(self, score, verification, out) -> ConfidenceAssessment:
        s = self.s
        if score is not None and score >= s.abstain_threshold and verification.status in (
            "supported", "partially_supported"
        ):
            return out("accept", f"Retries exhausted; confidence {score:.2f} >= abstain threshold "
                                 f"{s.abstain_threshold} - accepted with moderate confidence.")
        return out("abstain", f"Retries exhausted; confidence {score if score is None else round(score, 2)} "
                              f"or verification '{verification.status}' is too weak to answer.")

    def select_final(self, attempts: list[AttemptRecord]) -> tuple[AttemptRecord, ConfidenceAssessment]:
        """After the retry budget is spent, pick the best-scoring attempt and apply the fallback rule to it."""
        best = max(attempts, key=lambda a: (a.confidence.score or 0.0, -a.attempt))
        c = best.confidence
        if best.answer.insufficient or not best.evidence or best.verification.status == "contradicted":
            final = c.model_copy(update={"decision": "abstain",
                                         "reason": f"Best attempt ({best.attempt}) still lacks supported evidence."})
            return best, final

        def out(decision, reason, strat=None):
            return c.model_copy(update={"decision": decision, "retry_strategy": strat,
                                        "reason": f"Selected attempt {best.attempt} of {len(attempts)}. " + reason})

        return best, self._fallback(c.score, best.verification, out)
