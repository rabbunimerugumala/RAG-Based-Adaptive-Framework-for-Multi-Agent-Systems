"""Standard RAG baseline and the four-agent Adaptive RAG pipeline.

Standard RAG:  query -> hybrid retrieval -> answer generation -> citation validation -> response
Adaptive RAG:  query -> Planner -> Evidence -> draft answer -> Verifier -> Confidence
                     -> accept | retry (back to Planner with a new strategy, max 2) | abstain

Both methods share the same knowledge base, retriever, generator and default top_k, so comparisons are
made under the same evidence conditions. Every step is recorded in the returned trace.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

from adaptive_rag.agents.confidence import AttemptRecord, ConfidenceAgent
from adaptive_rag.agents.evidence import EvidenceAgent, EvidencePackage
from adaptive_rag.agents.planner import PlannerAgent
from adaptive_rag.agents.verifier import VerifierAgent
from adaptive_rag.config import Settings
from adaptive_rag.generation.llm import INSUFFICIENT_ANSWER, Generator, LLMError
from adaptive_rag.generation.nli import NLIModel
from adaptive_rag.retrieval.hybrid import CrossEncoderReranker, hybrid_search
from adaptive_rag.retrieval.store import KnowledgeBase
from adaptive_rag.schemas import (
    Citation,
    Evidence,
    GeneratedAnswer,
    QueryResponse,
    RunStats,
    TraceStep,
)
from adaptive_rag.text import keywords

EMPTY_KB_MESSAGE = ("The knowledge base is empty. Upload PDF, DOCX, TXT or Markdown documents in the "
                    "Document Library, then ask your question again.")
SMALL_TALK_MESSAGE = ("Hello! I answer questions using the documents in the knowledge base. "
                      "Ask a question about your uploaded documents.")
ABSTAIN_PREFIX = "I could not find sufficiently supported evidence in the documents to answer this reliably."


class _Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = round((time.perf_counter() - self.t) * 1000, 2)


def _citations(answer: GeneratedAnswer, evidence: list[Evidence]) -> list[Citation]:
    by_ref = {e.ref: e for e in evidence}
    out = []
    for r in answer.cited_refs:
        e = by_ref.get(r)
        out.append(Citation(ref=r, chunk_id=e.chunk_id if e else "", doc_id=e.doc_id if e else "",
                            filename=e.filename if e else "(invalid reference)", page=e.page if e else None,
                            chunk_index=e.chunk_index if e else -1, valid=e is not None))
    return out


class RAGSystem:
    def __init__(self, settings: Settings, kb: KnowledgeBase, generator: Generator,
                 nli: NLIModel | None = None, reranker: CrossEncoderReranker | None = None):
        self.settings, self.kb, self.generator = settings, kb, generator
        self.reranker = reranker
        self.planner = PlannerAgent(settings, generator)
        self.evidence_agent = EvidenceAgent(settings, kb, reranker, generator)
        self.verifier = VerifierAgent(settings, nli, generator)
        self.confidence = ConfidenceAgent(settings)

    @property
    def models(self) -> dict[str, str]:
        return {
            "generator": self.generator.name,
            "embedder": self.kb.embedder.name,
            "nli": self.verifier.nli.name if self.verifier.nli else "none",
            "reranker": self.reranker.name if self.reranker else "none",
            "evidence_judge": (f"{self.generator.name} log-odds ({self.settings.doc_judge_mode})"
                               if self.settings.doc_filter == "judge" else "off"),
        }

    def _response(self, method, question, answer, status, evidence, citations, verification, used, confidence,
                  trace, t0, llm_calls, nli_calls, attempts, analysis_ms=0.0, **extra) -> QueryResponse:
        return QueryResponse(**extra,
            run_id=uuid.uuid4().hex[:12], method=method, question=question, answer=answer, status=status,
            citations=citations, evidence=evidence, verification=verification,
            verification_used_in_decision=used, confidence=confidence, trace=trace,
            stats=RunStats(latency_ms=round((time.perf_counter() - t0) * 1000 - analysis_ms, 2),
                           llm_calls=llm_calls, nli_calls=nli_calls, retrieval_attempts=attempts,
                           retries=max(attempts - 1, 0), analysis_latency_ms=round(analysis_ms, 2)),
            mode=self.settings.mode, models=self.models,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    def _generate(self, question, evidence, trace, attempt) -> tuple[GeneratedAnswer, int]:
        """Draft-answer step. Returns the answer and LLM calls used (0 for the dev extractive generator)."""
        if not evidence:
            trace.append(TraceStep(step="generator", title="Answer generation", attempt=attempt, status="skipped",
                                   duration_ms=0, note="No evidence passages - generation skipped."))
            return GeneratedAnswer(answer=INSUFFICIENT_ANSWER, cited_refs=[], insufficient=True), 0
        calls = 1 if self.generator.is_llm else 0
        with _Timer() as t:
            try:
                ans = self.generator.answer(question, evidence)
                status, note = "completed", None
            except LLMError as exc:
                ans = GeneratedAnswer(answer=INSUFFICIENT_ANSWER, cited_refs=[], insufficient=True)
                status, note = "failed", str(exc)
        trace.append(TraceStep(
            step="generator", title=f"Answer generation ({self.generator.name})", attempt=attempt, status=status,
            duration_ms=t.ms, input={"question": question, "passages": [e.ref for e in evidence]},
            output={"answer": ans.answer, "cited_refs": ans.cited_refs, "insufficient": ans.insufficient},
            note=note,
        ))
        return ans, calls

    def _early_exit(self, method, question, t0) -> QueryResponse | None:
        if self.kb.num_chunks == 0:
            step = TraceStep(step="retrieval" if method == "standard" else "planner", title="Knowledge base check",
                             status="skipped", duration_ms=0, note="Knowledge base is empty; nothing to retrieve.")
            return self._response(method, question, EMPTY_KB_MESSAGE, "no_documents", [], [], None, False, None,
                                  [step], t0, 0, 0, 0)
        return None

    # ------------------------------------------------------------------ Standard RAG
    def run_standard(self, question: str, analyse: bool = True) -> QueryResponse:
        t0 = time.perf_counter()
        if (early := self._early_exit("standard", question, t0)) is not None:
            return early
        s = self.settings
        trace: list[TraceStep] = []
        with _Timer() as t:
            evidence = hybrid_search(self.kb, [question], keywords(question), s.top_k, s.candidate_k, s.rrf_k,
                                     self.reranker)
        trace.append(TraceStep(step="retrieval", title="Hybrid retrieval (single pass)", status="completed",
                               duration_ms=t.ms, input={"query": question, "top_k": s.top_k},
                               output={"returned": len(evidence),
                                       "chunk_ids": [e.chunk_id for e in evidence]}))
        ans, llm_calls = self._generate(question, evidence, trace, 0)
        with _Timer() as t:
            citations = _citations(ans, evidence)
            invalid = [c.ref for c in citations if not c.valid]
            if invalid:
                ans = ans.model_copy(update={"cited_refs": [r for r in ans.cited_refs if r not in invalid]})
                citations = [c for c in citations if c.valid]
        trace.append(TraceStep(step="citation_check", title="Citation validation", status="completed",
                               duration_ms=t.ms, output={"valid": [c.ref for c in citations], "removed_invalid": invalid}))
        verification, nli_calls, analysis_ms = None, 0, 0.0
        if analyse:
            # Post-hoc analysis so both methods can be compared; it does NOT change the standard answer.
            with _Timer() as t:
                verification, nli_calls = self.verifier.verify(question, ans, evidence)
            analysis_ms = t.ms
            trace.append(TraceStep(step="verifier", title="Post-hoc verification (analysis only)",
                                   status="completed", duration_ms=t.ms,
                                   output={"status": verification.status, "support_score": verification.support_score},
                                   note="Not part of the Standard RAG decision path; shown for comparison only."))
        status = "abstained" if ans.insufficient else "answered"
        return self._response("standard", question, ans.answer, status, evidence, citations, verification, False,
                              None, trace, t0, llm_calls, nli_calls, 1, analysis_ms)

    # ------------------------------------------------------------------ Adaptive RAG
    def run_adaptive(self, question: str) -> QueryResponse:
        t0 = time.perf_counter()
        trace: list[TraceStep] = []
        with _Timer() as t:
            plan = self.planner.plan(question)
        trace.append(TraceStep(step="planner", title="Planner Agent", status="completed", duration_ms=t.ms,
                               input={"query": question}, output=plan.model_dump(),
                               decision="retrieve" if plan.needs_retrieval else "no retrieval needed",
                               note=plan.explanation))
        if not plan.needs_retrieval:
            return self._response("adaptive", question, SMALL_TALK_MESSAGE, "no_retrieval_needed", [], [], None,
                                  False, None, trace, t0, 0, 0, 0)
        if (early := self._early_exit("adaptive", question, t0)) is not None:
            early.trace = trace + early.trace
            return early

        llm_calls = nli_calls = 0
        judge_cache: dict = {}  # passage judgments are reused across retries of the same run
        packages: dict[int, EvidencePackage] = {}
        attempts: list[AttemptRecord] = []
        final: AttemptRecord | None = None
        final_conf = None
        for attempt in range(self.settings.max_retries + 1):
            if attempt > 0:
                prev = attempts[-1]
                with _Timer() as t:
                    plan, used = self.planner.replan(prev.plan, prev.confidence.retry_strategy or "broaden",
                                                     prev.evidence, prev.verification, attempt)
                llm_calls += used
                trace.append(TraceStep(step="planner", title="Planner Agent (re-plan)", attempt=attempt,
                                       status="completed", duration_ms=t.ms,
                                       input={"retry_strategy": prev.confidence.retry_strategy,
                                              "reason": prev.confidence.reason},
                                       output={"search_queries": plan.search_queries, "top_k": plan.top_k,
                                               "candidate_k": plan.candidate_k, "llm_calls": used},
                                       decision=plan.strategy, note=plan.explanation))
            with _Timer() as t:
                try:
                    package = self.evidence_agent.retrieve(plan, judge_cache)
                    step_status = "completed"
                    step_note = ("Judge scores rank and filter passages; they do not prove claims." if package.report
                                 else "Retrieval scores rank passages; they do not prove claims.")
                except Exception as exc:  # retrieval failure: recorded, then handled like "no evidence"
                    package = EvidencePackage(evidence=[], summary={"error": f"{exc.__class__.__name__}: {exc}"})
                    step_status, step_note = "failed", f"Retrieval failed: {exc.__class__.__name__}: {exc}"
            packages[attempt] = package
            evidence = package.evidence
            llm_calls += package.llm_calls
            rep = package.report
            decision = (f"kept {rep.retained} of {rep.pool_size} (tau_q={rep.threshold})" if rep and rep.threshold
                        is not None else ("retrieved" if step_status == "completed" else "failed"))
            trace.append(TraceStep(step="evidence", title="Evidence Agent", attempt=attempt, status=step_status,
                                   duration_ms=t.ms, input={"search_queries": plan.search_queries, "top_k": plan.top_k,
                                                            "pool_k": plan.pool_k},
                                   output=package.summary, decision=decision, note=step_note))
            ans, calls = self._generate(question, evidence, trace, attempt)
            llm_calls += calls
            with _Timer() as t:
                verification, n = self.verifier.verify(question, ans, evidence)
            nli_calls += n
            llm_calls += verification.llm_calls
            trace.append(TraceStep(step="verifier", title="Verifier Agent", attempt=attempt, status="completed",
                                   duration_ms=t.ms, input={"method": verification.method},
                                   output=verification.model_dump(), decision=verification.status,
                                   note=verification.summary))
            retries_left = self.settings.max_retries - attempt
            with _Timer() as t:
                conf = self.confidence.assess(plan, evidence, ans, verification, retries_left)
            trace.append(TraceStep(step="confidence", title="Confidence Agent", attempt=attempt, status="completed",
                                   duration_ms=t.ms, output=conf.model_dump(), decision=conf.decision,
                                   note=conf.reason))
            rec = AttemptRecord(attempt, plan, evidence, ans, verification, conf)
            attempts.append(rec)
            if conf.decision == "retry":
                continue
            final, final_conf = rec, conf
            if conf.decision != "accept" and len(attempts) > 1:
                with _Timer() as t:
                    final, final_conf = self.confidence.select_final(attempts)
                trace.append(TraceStep(step="confidence", title="Confidence Agent (final selection)",
                                       attempt=attempt, status="completed", duration_ms=t.ms,
                                       input={"attempt_scores": {a.attempt: a.confidence.score for a in attempts}},
                                       output=final_conf.model_dump(), decision=final_conf.decision,
                                       note=final_conf.reason))
            break

        assert final is not None and final_conf is not None
        unresolved: list[str] = []
        if final_conf.decision == "accept":
            answer, status = final.answer.answer, "answered"
            citations = _citations(final.answer, final.evidence)
            citations = [c for c in citations if c.valid]
            v = final.verification
            support_level = {"supported": "verified", "partially_supported": "partial"}.get(v.status, "unverified")
            unresolved = [c.claim for c in v.claims if c.label != "supported"]
        else:
            answer, status, citations, support_level = f"{ABSTAIN_PREFIX} {final_conf.reason}", "abstained", [], "none"
        package = packages.get(final.attempt)
        return self._response("adaptive", question, answer, status, final.evidence, citations, final.verification,
                              True, final_conf, trace, t0, llm_calls, nli_calls, len(attempts),
                              support_level=support_level, unresolved_claims=unresolved,
                              filtered_evidence=package.filtered if package else [],
                              evidence_filter=package.report if package else None)

    def run(self, question: str, method: str) -> QueryResponse:
        return self.run_adaptive(question) if method == "adaptive" else self.run_standard(question)

