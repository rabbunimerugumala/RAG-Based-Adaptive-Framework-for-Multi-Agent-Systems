"""Model-loading layer. Builds the shared embedder, generator, optional NLI model and optional reranker
from Settings, and records a readable status for each so the UI can show real model readiness.

Exactly one generator instance is created and shared by every pipeline/agent role.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from adaptive_rag.config import Settings
from adaptive_rag.generation.llm import ExtractiveGenerator, Generator, OllamaLLM
from adaptive_rag.generation.nli import NLIModel
from adaptive_rag.retrieval.embeddings import Embedder, FastEmbedEmbedder, HashEmbedder
from adaptive_rag.retrieval.hybrid import CrossEncoderReranker
from adaptive_rag.retrieval.store import KnowledgeBase

log = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    embedder: Embedder
    generator: Generator
    nli: NLIModel | None
    reranker: CrossEncoderReranker | None
    status: dict[str, dict] = field(default_factory=dict)


def build_models(settings: Settings) -> ModelBundle:
    status: dict[str, dict] = {}
    cache = str(settings.runtime_dir / "model_cache")

    if settings.resolved_embedding_backend == "fastembed":
        embedder: Embedder = FastEmbedEmbedder(settings.embedding_model, cache_dir=cache)
        status["embedder"] = {"name": embedder.name, "ready": True, "detail": f"fastembed ONNX on CPU, dim={embedder.dim}"}
    else:
        embedder = HashEmbedder()
        status["embedder"] = {"name": embedder.name, "ready": True,
                              "detail": "DEV MODE hashing embedder (lexical, not semantic)"}

    if settings.resolved_llm_backend == "ollama":
        generator: Generator = OllamaLLM(settings.ollama_url, settings.ollama_model, settings.ollama_num_ctx,
                                         settings.llm_max_tokens, settings.llm_timeout_s, settings.ollama_num_gpu)
        status["generator"] = {"name": generator.name, **generator.status()}
    else:
        generator = ExtractiveGenerator()
        status["generator"] = {"name": generator.name, "ready": True,
                               "detail": "DEV MODE extractive generator (not a language model)"}

    nli = None
    if settings.resolved_nli_backend == "transformers":
        try:
            nli = NLIModel(settings.nli_model)
            status["nli"] = {"name": nli.name, "ready": True, "detail": "transformers on CPU"}
        except Exception as exc:  # optional component: degrade to lexical verification, but say so
            log.warning("NLI model unavailable: %s", exc)
            status["nli"] = {"name": settings.nli_model, "ready": False,
                             "detail": f"unavailable ({exc.__class__.__name__}); verifier uses lexical checks only"}
    else:
        status["nli"] = {"name": "none", "ready": False, "detail": "disabled; verifier uses lexical checks only"}

    reranker = None
    if settings.reranker == "cross-encoder":
        try:
            reranker = CrossEncoderReranker(settings.reranker_model)
            status["reranker"] = {"name": reranker.name, "ready": True, "detail": "fastembed ONNX cross-encoder"}
        except Exception as exc:
            log.warning("Reranker unavailable: %s", exc)
            status["reranker"] = {"name": settings.reranker_model, "ready": False, "detail": str(exc)[:200]}
    else:
        status["reranker"] = {"name": "none", "ready": False, "detail": "disabled (ARAG_RERANKER=none)"}
    return ModelBundle(embedder, generator, nli, reranker, status)


def open_kb(settings: Settings, models: ModelBundle, root: Path) -> KnowledgeBase:
    return KnowledgeBase(root, models.embedder, settings.chunk_size, settings.chunk_overlap)
