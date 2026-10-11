"""Central configuration.

All settings are read from environment variables prefixed with ``ARAG_`` (or a ``.env`` file in the
project root). There is exactly one configuration system: this module.

Two run modes exist:

* ``dev``  - lightweight development/testing mode. Uses a hashing embedder (lexical, *not* semantic)
             and a deterministic extractive answer generator instead of an LLM. Nothing is downloaded.
             Results produced in this mode must NOT be reported as real-model experiments.
* ``real`` - local models: Ollama-served instruction-tuned LLM, fastembed (ONNX) embeddings,
             optional NLI verifier and optional cross-encoder reranker.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Hardware profiles: defaults for laptops with 8 / 16 / 32 GB RAM. A profile only fills settings that were not set
# explicitly (env or .env), so any value can still be overridden. Only "32gb" was used for the reported benchmarks.
HARDWARE_PROFILES: dict[str, dict] = {
    "8gb": {"ollama_model": "qwen2.5:1.5b", "ollama_num_ctx": 2048, "llm_max_tokens": 256, "nli_backend": "none",
            "doc_pool_k": 6, "question_routing": True},
    "16gb": {"ollama_num_ctx": 3072, "doc_pool_k": 8, "question_routing": True},
    "32gb": {},
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ARAG_", env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    mode: Literal["dev", "real"] = "dev"

    # --- storage -------------------------------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    runtime_dir: Path = PROJECT_ROOT / "data" / "runtime"  # indexes, uploads, logs (git-ignored)
    results_dir: Path = PROJECT_ROOT / "evaluation" / "results"

    # --- ingestion -----------------------------------------------------------------------------
    max_upload_mb: int = Field(20, ge=1, le=200)
    chunk_size: int = Field(900, ge=200, le=4000, description="Target chunk size in characters")
    chunk_overlap: int = Field(150, ge=0, le=1000)

    # --- models --------------------------------------------------------------------------------
    embedding_backend: Literal["auto", "hash", "fastembed"] = "auto"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    llm_backend: Literal["auto", "extractive", "ollama"] = "auto"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3:4b-instruct-2507-q4_K_M"
    ollama_num_ctx: int = Field(4096, ge=1024, le=32768)
    ollama_num_gpu: int | None = Field(None, description="Layers to offload to GPU; 0 forces CPU; None = auto")
    llm_timeout_s: float = Field(180.0, ge=5)
    llm_max_tokens: int = Field(384, ge=32, le=2048)
    nli_backend: Literal["auto", "none", "transformers"] = "auto"
    nli_model: str = "cross-encoder/nli-deberta-v3-xsmall"
    reranker: Literal["none", "cross-encoder"] = "none"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # --- retrieval -----------------------------------------------------------------------------
    top_k: int = Field(5, ge=1, le=20, description="Passages given to the generator")
    candidate_k: int = Field(20, ge=1, le=100, description="Candidates per retriever before fusion")
    rrf_k: int = Field(60, ge=1, description="Reciprocal-rank-fusion constant")

    # --- confidence agent (experimental parameters, not universal truths) ----------------------
    w_relevance: float = Field(0.25, ge=0)
    w_coverage: float = Field(0.25, ge=0)
    w_verification: float = Field(0.35, ge=0)
    w_consistency: float = Field(0.15, ge=0)
    accept_threshold: float = Field(0.65, ge=0, le=1)
    abstain_threshold: float = Field(0.40, ge=0, le=1)
    max_retries: int = Field(2, ge=0, le=2, description="Adaptive retries; hard-capped at 2")

    # --- question routing (Planner): simple questions take a lightweight route ---------------------
    # Default ON since the routing experiment (held-out run 20261011-171730-9902e7): same accuracy as v2 with ~21 % fewer
    # LLM calls and ~26 % lower mean latency. Disable with ARAG_QUESTION_ROUTING=false to run the evaluated v2.
    question_routing: bool = Field(
        True, description="Planner routes simple questions to a lightweight round with a smaller escalation budget")
    simple_route_retries: int = Field(1, ge=0, le=2, description="Escalation rounds allowed for simple questions")
    hardware_profile: Literal["8gb", "16gb", "32gb"] = "32gb"

    # --- verifier ------------------------------------------------------------------------------
    nli_entail_threshold: float = Field(0.5, ge=0, le=1)
    nli_contradict_threshold: float = Field(0.6, ge=0, le=1)
    lexical_support_threshold: float = Field(0.75, ge=0, le=1)
    lexical_partial_threshold: float = Field(0.45, ge=0, le=1)

    # --- multi-agent v2 (each switch can be turned off to reproduce the original v1 pipeline) -----
    # Evidence Agent: MAIN-RAG-inspired document judging and filtering (Chang et al., ACL 2025, Sec. 3.2-3.3)
    doc_filter: Literal["off", "judge"] = "judge"
    # always: judge-filter on every round; on_retry: round 1 uses plain hybrid top-k (like Standard RAG) and the
    # Evidence Agent escalates to judge filtering only when the Confidence Agent requests more evidence
    doc_filter_stage: Literal["always", "on_retry"] = "on_retry"
    doc_judge_mode: Literal["pair", "triplet"] = Field(
        "pair", description="pair: Judge(doc, question) - adaptation; triplet: paper's Doc-Q-A with per-doc Predictor")
    doc_pool_k: int = Field(10, ge=2, le=30, description="Candidate passages judged per retrieval round")
    doc_filter_n_sigma: float = Field(0.0, ge=0, le=3, description="tau_q = mean - n*std (paper default n=0)")
    doc_min_keep: int = Field(1, ge=1, le=10, description="Fallback: never keep fewer passages than this")
    dedup_jaccard: float = Field(0.8, ge=0.5, le=1.0, description="Drop passages this similar to a better one")
    # Verifier: v1 = original rules; grounded = contradiction only from related, well-formed premises,
    # plus a judge-backed support check for short answers
    verifier_mode: Literal["v1", "grounded"] = "grounded"
    # grounded mode: an NLI contradiction must be corroborated - if the evidence judge confirms the claim on the
    # same evidence, the claim is marked unresolved (partially supported) instead of contradicted
    contradiction_cross_check: bool = True
    # Confidence: v1 = original decision rules; v2 = judge-based evidence relevance, partial answers,
    # attempt selection that prefers verified answers
    confidence_mode: Literal["v1", "v2"] = "v2"

    # --- api -----------------------------------------------------------------------------------
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    @model_validator(mode="after")
    def _check(self) -> Settings:
        for key, value in HARDWARE_PROFILES[self.hardware_profile].items():
            if key not in self.model_fields_set:
                object.__setattr__(self, key, value)
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.abstain_threshold > self.accept_threshold:
            raise ValueError("abstain_threshold must not exceed accept_threshold")
        if self.weight_sum <= 0:
            raise ValueError("at least one confidence weight must be positive")
        return self

    @property
    def weight_sum(self) -> float:
        return self.w_relevance + self.w_coverage + self.w_verification + self.w_consistency

    @property
    def resolved_embedding_backend(self) -> str:
        if self.embedding_backend != "auto":
            return self.embedding_backend
        return "hash" if self.mode == "dev" else "fastembed"

    @property
    def resolved_llm_backend(self) -> str:
        if self.llm_backend != "auto":
            return self.llm_backend
        return "extractive" if self.mode == "dev" else "ollama"

    @property
    def resolved_nli_backend(self) -> str:
        if self.nli_backend != "auto":
            return self.nli_backend
        return "none" if self.mode == "dev" else "transformers"

    def public_dict(self) -> dict:
        """Configuration snapshot stored next to results and shown in the UI (no secrets exist)."""
        d = self.model_dump(mode="json", exclude={"data_dir", "runtime_dir", "results_dir", "cors_origins"})
        d["resolved"] = {
            "embedding_backend": self.resolved_embedding_backend,
            "llm_backend": self.resolved_llm_backend,
            "nli_backend": self.resolved_nli_backend,
        }
        return d


# Switches that reproduce the original (v1) Adaptive RAG pipeline exactly - used for baseline comparisons.
LEGACY_V1 = {"doc_filter": "off", "verifier_mode": "v1", "confidence_mode": "v1", "question_routing": False}


@lru_cache
def get_settings() -> Settings:
    return Settings()
