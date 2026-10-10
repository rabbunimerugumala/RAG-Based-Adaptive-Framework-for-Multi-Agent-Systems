"""Agent 3 - Verifier. Checks every claim (sentence) of the draft answer against the retrieved evidence.

Two independent signals per claim:
  * lexical support - fraction of the claim's content terms found in the best passage; any number in the
    claim must appear in that passage, otherwise lexical support is capped at 0.4.
  * NLI (optional)  - max entailment / contradiction probability of the claim against 3-sentence windows
    of the evidence passages, from a local NLI cross-encoder.

Labelling rules (thresholds are configurable):
  with NLI:   entail >= t_e -> supported; contradiction >= t_c -> contradicted;
              otherwise lexical >= t_partial -> partially_supported; else unsupported.
              (High lexical overlap alone never yields 'supported' when NLI is available.)
  lexical only: lexical >= t_support -> supported; >= t_partial -> partially_supported; else unsupported.
              Contradictions cannot be detected in this mode, which is reported in `method`.

The verifier never asks the generating LLM whether its own answer is correct.
"""

from __future__ import annotations

import re

from adaptive_rag.config import Settings
from adaptive_rag.generation.nli import NLIModel
from adaptive_rag.schemas import ClaimCheck, Evidence, GeneratedAnswer, VerificationResult
from adaptive_rag.text import citation_refs, content_terms, numbers, split_sentences, strip_citations

_LABEL_SCORE = {"supported": 1.0, "partially_supported": 0.5, "unsupported": 0.0, "contradicted": 0.0}


def lexical_support(claim: str, passage: str) -> float:
    terms = set(content_terms(claim))
    if not terms:
        return 0.0
    p_terms = set(content_terms(passage))
    score = len(terms & p_terms) / len(terms)
    if not numbers(claim) <= numbers(passage):
        score = min(score, 0.4)
    return score


def _windows(text: str, size: int = 3, stride: int = 2) -> list[str]:
    sents = split_sentences(text)
    if len(sents) <= size:
        return [" ".join(sents)] if sents else []
    return [" ".join(sents[i : i + size]) for i in range(0, len(sents) - size + stride, stride)]


_CLAUSE_SPLIT = re.compile(r";\s+|,\s+(?:while|whereas|but)\s+|\s+whereas\s+|,\s+and\s+(?=the\s|it\s|its\s|they\s)", re.I)


def atomic_claims(sentence: str) -> list[str]:
    """Split a compound sentence ('X is A, while Y is B') into clauses that can be checked one by one.
    A split is kept only if every part still has at least three content terms."""
    parts = [p.strip(" ,.") for p in _CLAUSE_SPLIT.split(sentence) if p and p.strip(" ,.")]
    if len(parts) > 1 and all(len(content_terms(p)) >= 3 for p in parts):
        return parts
    return [sentence.strip()]


def declarative(question: str, answer: str) -> str:
    """Heuristic question+short-answer -> statement conversion for NLI (e.g. 'Who founded X?' + 'Edda Larsk'
    -> 'Edda Larsk founded X'). Falls back to 'question answer' when no pattern applies."""
    q = question.strip().rstrip("?. ").strip()
    a = answer.strip().rstrip(".")
    aux = r"(is|are|was|were)"
    if m := re.match(rf"^(?:in\s+)?(?:what|which)\s+year\s+{aux}\s+(.+?)\s+(\w+)$", q, re.I):
        return f"{m.group(2)} {m.group(1)} {m.group(3)} in {a}."
    if m := re.match(rf"^(?:who|what|which(?:\s+\w+)?)\s+{aux}\s+(.+)$", q, re.I):
        return f"{m.group(2)} {m.group(1)} {a}."
    if m := re.match(r"^how\s+many\s+(\w+(?:\s+\w+)?)\s+(?:does|do|did|can)\s+(.+?)\s+(\w+)$", q, re.I):
        return f"{m.group(2)} {m.group(3)} {a} {m.group(1)}."
    if m := re.match(r"^(?:who|which\s+\w+|what\s+\w+)\s+(?!(?:is|are|was|were|do|does|did))(.+)$", q, re.I):
        return f"{a} {m.group(1)}."
    return f"{q}? {a}."


class VerifierAgent:
    def __init__(self, settings: Settings, nli: NLIModel | None):
        self.settings, self.nli = settings, nli

    @property
    def method(self) -> str:
        return f"lexical+nli({self.nli.name})" if self.nli else "lexical-only (no NLI model; contradictions undetectable)"

    def verify(self, question: str, answer: GeneratedAnswer, evidence: list[Evidence]) -> tuple[VerificationResult, int]:
        """Return the verification result and the number of NLI model invocations (batched calls)."""
        if answer.insufficient:
            return VerificationResult(
                status="not_applicable", method=self.method,
                summary="The generator reported insufficient evidence, so there are no claims to verify.",
            ), 0
        sentences = [c for c in split_sentences(answer.answer) if content_terms(strip_citations(c))]
        if not sentences or not evidence:
            return VerificationResult(
                status="unsupported", support_score=0.0, citation_consistency=0.0 if sentences else None,
                method=self.method, unsupported_claims=sentences,
                summary="No evidence passages were available to support the answer." if sentences else "Empty answer.",
            ), 0

        by_ref = {e.ref: e for e in evidence}
        s = self.settings
        claims: list[tuple[str, list[int]]] = []
        for raw in sentences:
            cited = [r for r in citation_refs(raw) if r in by_ref] or [r for r in answer.cited_refs if r in by_ref]
            claims.extend((c, cited) for c in atomic_claims(strip_citations(raw)))
        checks: list[ClaimCheck] = []
        nli_calls = 0
        for claim, cited in claims:
            lex = {e.ref: lexical_support(claim, e.text) for e in evidence}
            best_ref = max(lex, key=lambda r: (lex[r], r in cited))
            best_lex = lex[best_ref]
            if len(cited) > 1:  # a claim may combine facts from several cited passages
                best_lex = max(best_lex, lexical_support(claim, " ".join(by_ref[r].text for r in cited)))
            entail = contra = None
            if self.nli:
                hypothesis = claim if len(content_terms(claim)) >= 4 else declarative(question, claim)
                refs = list(dict.fromkeys(cited + sorted(lex, key=lex.get, reverse=True)[:3]))
                premises = [(r, w) for r in refs for w in _windows(by_ref[r].text)]
                if len(cited) > 1:
                    premises.append((cited[0], " ".join(by_ref[r].text for r in cited)))
                preds = self.nli.predict([(w, hypothesis) for _, w in premises])
                nli_calls += 1
                i_best = max(range(len(preds)), key=lambda i: preds[i]["entailment"])
                entail = preds[i_best]["entailment"]
                # contradictions only count from premises that talk about the same thing as the claim
                related = [p["contradiction"] for (_, w), p in zip(premises, preds)
                           if lexical_support(claim, w) >= s.lexical_partial_threshold]
                contra = max(related, default=0.0)
                if entail >= s.nli_entail_threshold:
                    best_ref = premises[i_best][0]
            label, reason = self._label(best_lex, entail, contra)
            checks.append(ClaimCheck(
                claim=claim, label=label, lexical_support=round(best_lex, 3),
                nli_entailment=None if entail is None else round(entail, 3),
                nli_contradiction=None if contra is None else round(contra, 3),
                best_evidence_ref=best_ref, cited_refs=cited, reason=reason,
            ))

        support = sum(_LABEL_SCORE[c.label] for c in checks) / len(checks)
        consistent = 0
        for c in checks:
            if c.label in ("supported", "partially_supported") and c.cited_refs and (
                c.best_evidence_ref in c.cited_refs
                or any(lexical_support(c.claim, by_ref[r].text) >= s.lexical_partial_threshold for r in c.cited_refs)
            ):
                consistent += 1
        consistency = consistent / len(checks)
        unsupported = [c.claim for c in checks if c.label == "unsupported"]
        contradicted = [c.claim for c in checks if c.label == "contradicted"]
        if contradicted:
            status = "contradicted"
        elif all(c.label == "supported" for c in checks):
            status = "supported"
        elif support > 0:
            status = "partially_supported"
        else:
            status = "unsupported"
        n_sup = sum(c.label == "supported" for c in checks)
        summary = (f"{n_sup}/{len(checks)} claim(s) supported, {len(unsupported)} unsupported, "
                   f"{len(contradicted)} contradicted; citation consistency {consistency:.2f}.")
        return VerificationResult(
            status=status, support_score=round(support, 4), citation_consistency=round(consistency, 4),
            claims=checks, unsupported_claims=unsupported, contradicted_claims=contradicted,
            method=self.method, summary=summary,
        ), nli_calls

    def _label(self, lex: float, entail: float | None, contra: float | None) -> tuple[str, str]:
        s = self.settings
        if entail is not None:
            if entail >= s.nli_entail_threshold:
                return "supported", f"NLI entailment {entail:.2f} >= {s.nli_entail_threshold}"
            if contra is not None and contra >= s.nli_contradict_threshold:
                return "contradicted", f"NLI contradiction {contra:.2f} >= {s.nli_contradict_threshold}"
            if lex >= s.lexical_partial_threshold:
                return "partially_supported", f"terms found (lexical {lex:.2f}) but NLI did not confirm entailment"
            return "unsupported", f"low lexical support {lex:.2f} and no NLI entailment"
        if lex >= s.lexical_support_threshold:
            return "supported", f"lexical support {lex:.2f} >= {s.lexical_support_threshold} (lexical-only check)"
        if lex >= s.lexical_partial_threshold:
            return "partially_supported", f"lexical support {lex:.2f} (partial)"
        return "unsupported", f"lexical support {lex:.2f} < {s.lexical_partial_threshold}"
