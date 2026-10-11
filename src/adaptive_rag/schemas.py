"""Pydantic data schemas shared by the pipeline, the API and the evaluation code."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Method = Literal["standard", "adaptive"]


# --------------------------------------------------------------------------- documents
class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    size_bytes: int
    sha256: str
    num_chunks: int
    num_pages: int | None = None
    num_characters: int
    status: Literal["indexed"] = "indexed"
    created_at: str


class UploadResult(BaseModel):
    filename: str
    status: Literal["indexed", "duplicate", "failed"]
    message: str
    document: DocumentInfo | None = None


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    filename: str
    text: str
    chunk_index: int
    page: int | None = None
    char_start: int
    char_end: int


# --------------------------------------------------------------------------- evidence
class Evidence(BaseModel):
    """A retrieved passage. Scores describe *retrieval relevance*, never truth of a claim."""

    ref: int = Field(description="1-based number used for citations like [1]")
    chunk_id: str
    doc_id: str
    filename: str
    page: int | None = None
    chunk_index: int
    text: str
    rrf_score: float
    dense_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None
    keyword_overlap: float = Field(0.0, description="Fraction of query keywords present in this passage")
    matched_queries: list[str] = []
    retrieval_rank: int | None = Field(None, description="Rank after hybrid retrieval, before judging")
    judge_score: float | None = Field(
        None, description="Evidence Agent judge score: log P(Yes) - log P(No). Relevance, not proof of truth")
    candidate_answer: str | None = Field(None, description="Per-passage Predictor answer (triplet mode only)")
    filter_reason: str | None = None


# --------------------------------------------------------------------------- agents
class RetrievalPlan(BaseModel):
    query: str
    needs_retrieval: bool
    query_type: str
    keywords: list[str]
    focus_terms: list[str] = Field([], description="Terms naming what is asked for, e.g. 'retail price'")
    subquestions: list[str]
    search_queries: list[str]
    top_k: int
    candidate_k: int
    pool_k: int = Field(0, description="Passages judged by the Evidence Agent per round (0 = no judging)")
    route: Literal["simple", "complex", "ambiguous", "full"] = Field(
        "full", description="Planner's route; 'full' when question routing is disabled")
    route_reason: str = ""
    retry_budget: int = Field(2, description="Escalation rounds allowed for this question")
    strategy: str
    explanation: str


class ClaimCheck(BaseModel):
    claim: str
    label: Literal["supported", "partially_supported", "unsupported", "contradicted"]
    lexical_support: float
    nli_entailment: float | None = None
    nli_contradiction: float | None = None
    best_evidence_ref: int | None = None
    cited_refs: list[int] = []
    reason: str


class VerificationResult(BaseModel):
    status: Literal["supported", "partially_supported", "unsupported", "contradicted", "not_applicable"]
    support_score: float | None = Field(None, description="Weighted fraction of supported claims, 0-1")
    citation_consistency: float | None = None
    claims: list[ClaimCheck] = []
    unsupported_claims: list[str] = []
    contradicted_claims: list[str] = []
    method: str
    summary: str
    llm_calls: int = Field(0, description="Judge calls made by the verifier (grounded mode, short answers)")


class ConfidenceAssessment(BaseModel):
    score: float | None
    components: dict[str, float | None]
    weights: dict[str, float]
    missing_components: list[str]
    decision: Literal["accept", "retry", "abstain"]
    reason: str
    retry_strategy: str | None = None
    note: str = "Heuristic score, not a calibrated probability."


class GeneratedAnswer(BaseModel):
    answer: str
    cited_refs: list[int]
    insufficient: bool
    raw: str | None = None


# --------------------------------------------------------------------------- trace / response
class TraceStep(BaseModel):
    step: str  # planner | evidence | generator | verifier | confidence | retrieval | citation_check
    title: str
    attempt: int = 0
    status: Literal["completed", "skipped", "failed"]
    duration_ms: float
    input: dict[str, Any] = {}
    output: dict[str, Any] = {}
    decision: str | None = None
    note: str | None = None


class EvidenceFilterReport(BaseModel):
    mode: str
    pool_size: int
    duplicates_removed: int
    threshold: float | None
    mean: float | None
    std: float | None
    n_sigma: float
    retained: int
    filtered: int
    fallback: str | None = None
    judge_calls: int


class Citation(BaseModel):
    ref: int
    chunk_id: str
    doc_id: str
    filename: str
    page: int | None
    chunk_index: int
    valid: bool


class RunStats(BaseModel):
    latency_ms: float
    llm_calls: int
    nli_calls: int
    retrieval_attempts: int
    retries: int
    analysis_latency_ms: float = 0.0


class QueryResponse(BaseModel):
    run_id: str
    method: Method
    question: str
    answer: str
    status: Literal["answered", "abstained", "no_documents", "no_retrieval_needed", "error"]
    citations: list[Citation]
    evidence: list[Evidence]
    verification: VerificationResult | None
    verification_used_in_decision: bool
    confidence: ConfidenceAssessment | None
    route: str | None = Field(None, description="Planner route of the Multi-Agent run (simple/complex/ambiguous/full)")
    support_level: Literal["verified", "partial", "unverified", "none"] = Field(
        "none", description="verified: all claims supported; partial: some claims unresolved; none: no answer")
    unresolved_claims: list[str] = []
    filtered_evidence: list[Evidence] = Field([], description="Retrieved passages the Evidence Agent removed")
    evidence_filter: EvidenceFilterReport | None = None
    trace: list[TraceStep]
    stats: RunStats
    mode: str
    models: dict[str, str]
    created_at: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    method: Method = "adaptive"


class CompareRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)


class CompareResponse(BaseModel):
    question: str
    standard: QueryResponse
    adaptive: QueryResponse
    note: str


class EvaluationRequest(BaseModel):
    dataset: str
    limit: int | None = Field(None, ge=1, le=2000)
    methods: list[str] = Field(["standard", "adaptive"], min_length=1,
                               description="standard | adaptive | adaptive_no_retry | adaptive_no_filter | adaptive_v1")
    repeat: int = Field(0, ge=0, le=50, description="Re-run the first N questions to measure response consistency")
