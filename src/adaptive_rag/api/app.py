"""FastAPI application. Interactive OpenAPI docs: http://127.0.0.1:8000/docs

Local prototype: there is no authentication. Do not expose this server to untrusted networks.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from adaptive_rag import __version__
from adaptive_rag.config import Settings, get_settings
from adaptive_rag.evaluation.datasets import list_datasets
from adaptive_rag.evaluation.runner import (
    VARIANT_LABELS,
    VARIANTS,
    EvaluationCancelled,
    run_evaluation,
    save_results,
)
from adaptive_rag.generation.llm import LLMError
from adaptive_rag.models import ModelBundle, build_models
from adaptive_rag.pipeline import RAGSystem
from adaptive_rag.retrieval.store import KnowledgeBase
from adaptive_rag.schemas import (
    CompareRequest,
    CompareResponse,
    DocumentInfo,
    EvaluationRequest,
    QueryRequest,
    QueryResponse,
    UploadResult,
)

log = logging.getLogger("adaptive_rag.api")


class RunStore:
    """Keeps every query response (with its full trace) on disk so traces survive restarts."""

    def __init__(self, root: Path, keep: int = 200):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.keep = keep
        self._lock = threading.Lock()

    def save(self, resp: QueryResponse) -> None:
        with self._lock:
            (self.root / f"{resp.run_id}.json").write_text(resp.model_dump_json(), encoding="utf-8")
            files = sorted(self.root.glob("*.json"), key=lambda p: p.stat().st_mtime)
            for old in files[: max(len(files) - self.keep, 0)]:
                old.unlink(missing_ok=True)

    def get(self, run_id: str) -> QueryResponse | None:
        if not run_id.isalnum():
            return None
        p = self.root / f"{run_id}.json"
        return QueryResponse.model_validate_json(p.read_text(encoding="utf-8")) if p.exists() else None

    def recent(self, limit: int) -> list[dict]:
        files = sorted(self.root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
        out = []
        for p in files:
            r = QueryResponse.model_validate_json(p.read_text(encoding="utf-8"))
            out.append({"run_id": r.run_id, "method": r.method, "question": r.question, "status": r.status,
                        "confidence": r.confidence.score if r.confidence else None,
                        "verification": r.verification.status if r.verification else None,
                        "latency_ms": r.stats.latency_ms, "created_at": r.created_at, "mode": r.mode})
        return out


class EvaluationManager:
    """Runs one benchmark at a time in a background thread and exposes live progress."""

    def __init__(self, settings: Settings, models: ModelBundle):
        self.settings, self.models = settings, models
        self.jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def start(self, req: EvaluationRequest) -> dict:
        with self._lock:
            if any(j["status"] == "running" for j in self.jobs.values()):
                raise HTTPException(409, "An evaluation is already running; wait for it or cancel it.")
            job_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
            job = {"job_id": job_id, "dataset": req.dataset, "methods": req.methods, "limit": req.limit,
                   "status": "running", "done": 0, "total": None, "rows": [], "error": None, "result": None,
                   "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "cancel": threading.Event()}
            self.jobs[job_id] = job
        threading.Thread(target=self._run, args=(job, req), daemon=True).start()
        return self.public(job)

    def _run(self, job: dict, req: EvaluationRequest) -> None:
        def progress(done, total, row):
            job["done"], job["total"] = done, total
            job["rows"].append(row)

        try:
            result = run_evaluation(self.settings, self.models, req.dataset, req.methods, req.limit,
                                    progress=progress, cancel=job["cancel"], run_id=job["job_id"], repeat=req.repeat)
            save_results(result, self.settings.results_dir)
            job.update(status="completed", result={k: v for k, v in result.items() if k != "rows"})
        except EvaluationCancelled:
            job.update(status="cancelled", error="Cancelled by user; no result files were written.")
        except Exception as exc:
            log.exception("evaluation failed")
            job.update(status="failed", error=f"{exc.__class__.__name__}: {exc}")

    @staticmethod
    def public(job: dict, with_rows: bool = True) -> dict:
        d = {k: v for k, v in job.items() if k != "cancel"}
        if not with_rows:
            d.pop("rows", None)
        return d

    def get(self, job_id: str) -> dict:
        if job_id in self.jobs:
            return self.public(self.jobs[job_id])
        path = self._result_path(job_id)
        if not path.exists():
            raise HTTPException(404, "Evaluation not found")
        res = json.loads(path.read_text(encoding="utf-8"))
        return {"job_id": job_id, "dataset": res["dataset"]["name"], "methods": res["methods"],
                "limit": res["sample"]["limit"], "status": "completed", "done": len(res["rows"]),
                "total": len(res["rows"]), "rows": res["rows"], "error": None, "started_at": res["started_at"],
                "result": {k: v for k, v in res.items() if k != "rows"}}

    def list(self) -> list[dict]:
        out = {j: self.public(job, with_rows=False) for j, job in self.jobs.items()}
        rd = self.settings.results_dir
        if rd.exists():
            for p in rd.glob("*/results.json"):
                if p.parent.name not in out:
                    res = json.loads(p.read_text(encoding="utf-8"))
                    out[p.parent.name] = {"job_id": p.parent.name, "dataset": res["dataset"]["name"],
                                          "methods": res["methods"], "limit": res["sample"]["limit"],
                                          "status": "completed", "done": len(res["rows"]), "total": len(res["rows"]),
                                          "started_at": res["started_at"], "mode": res["mode"], "error": None}
        return sorted(out.values(), key=lambda j: j["started_at"], reverse=True)

    def cancel(self, job_id: str) -> dict:
        job = self.jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Evaluation not found")
        job["cancel"].set()
        return {"job_id": job_id, "status": "cancelling"}

    def _result_path(self, job_id: str) -> Path:
        if not all(c.isalnum() or c == "-" for c in job_id):
            raise HTTPException(400, "Invalid evaluation id")
        return self.settings.results_dir / job_id / "results.json"


class AppState:
    def __init__(self, settings: Settings, models: ModelBundle | None = None):
        self.settings = settings
        self.models = models or build_models(settings)
        self.kb = KnowledgeBase(settings.runtime_dir / "kb", self.models.embedder,
                                settings.chunk_size, settings.chunk_overlap)
        self.system = RAGSystem(settings, self.kb, self.models.generator, self.models.nli, self.models.reranker)
        self.runs = RunStore(settings.runtime_dir / "runs")
        self.evals = EvaluationManager(settings, self.models)

    def close(self) -> None:
        self.kb.close()


def create_app(settings: Settings | None = None, models: ModelBundle | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.s = AppState(settings, models)
        yield
        app.state.s.close()

    app = FastAPI(
        title="RAG-Based Adaptive Framework for Multi-Agent Systems",
        description="Standard RAG vs. a four-agent Adaptive RAG (Planner, Evidence, Verifier, Confidence). "
                    "Local research prototype - no authentication; do not expose publicly.",
        version=__version__, lifespan=lifespan,
    )
    app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
                       allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(LLMError)
    async def llm_error(_: Request, exc: LLMError):
        return JSONResponse(status_code=503, content={"detail": f"Language model unavailable: {exc}"})

    def st(request: Request) -> AppState:
        return request.app.state.s

    # ------------------------------------------------------------------ health
    @app.get("/api/health", tags=["system"], summary="Backend status, model readiness and knowledge-base size")
    def health(request: Request):
        s = st(request)
        status = dict(s.models.status)
        gen = s.models.generator
        if hasattr(gen, "status"):
            status["generator"] = {"name": gen.name, **gen.status()}
        return {
            "status": "ok", "version": __version__, "mode": settings.mode,
            "dev_mode_warning": ("Development/testing mode: hashing embedder and extractive generator, no LLM. "
                                 "Outputs are not real-model results.") if settings.mode == "dev" else None,
            "models": status,
            "knowledge_base": {"documents": s.kb.num_documents, "chunks": s.kb.num_chunks,
                               "embedder": s.kb.embedder.name, "reindexed_on_startup": s.kb.reindexed_on_startup},
            "config": settings.public_dict(),
        }

    @app.get("/api/ready", tags=["system"], summary="Readiness: 200 when the generator can answer, else 503")
    def ready(request: Request):
        gen = st(request).models.generator
        status = gen.status() if hasattr(gen, "status") else {"ready": True, "detail": "dev generator"}
        if not status["ready"]:
            raise HTTPException(503, f"Generator not ready: {status['detail']}")
        return {"ready": True, "generator": gen.name}

    # ------------------------------------------------------------------ documents
    @app.get("/api/documents", tags=["documents"], response_model=list[DocumentInfo])
    def list_documents(request: Request):
        return st(request).kb.list_documents()

    @app.post("/api/documents", tags=["documents"], response_model=list[UploadResult],
              summary="Upload and index PDF (text layer), DOCX, TXT or Markdown files")
    def upload_documents(request: Request, files: list[UploadFile] = File(...)):  # noqa: B008
        if not files:
            raise HTTPException(400, "No files were uploaded.")
        limit = settings.max_upload_mb * 1024 * 1024
        results = []
        for f in files:
            data = f.file.read(limit + 1)
            if len(data) > limit:
                results.append(UploadResult(filename=f.filename or "file", status="failed",
                                            message=f"File exceeds the {settings.max_upload_mb} MB limit."))
                continue
            results.append(st(request).kb.add_bytes(f.filename or "document", data))
        return results

    @app.delete("/api/documents/{doc_id}", tags=["documents"])
    def delete_document(request: Request, doc_id: str):
        if not st(request).kb.delete_document(doc_id):
            raise HTTPException(404, "Document not found")
        return {"deleted": doc_id}

    # ------------------------------------------------------------------ queries
    @app.post("/api/query", tags=["query"], response_model=QueryResponse)
    def query(request: Request, req: QueryRequest):
        s = st(request)
        resp = s.system.run(req.question.strip(), req.method)
        s.runs.save(resp)
        return resp

    @app.post("/api/compare", tags=["query"], response_model=CompareResponse,
              summary="Run Standard and Adaptive RAG on the same question and knowledge base")
    def compare(request: Request, req: CompareRequest):
        s = st(request)
        q = req.question.strip()
        std = s.system.run_standard(q)
        ada = s.system.run_adaptive(q)
        s.runs.save(std)
        s.runs.save(ada)
        return CompareResponse(question=q, standard=std, adaptive=ada, note=(
            "Both methods used the same knowledge base, retriever, generator and default top_k. Standard RAG "
            "verification is post-hoc analysis only and did not influence its answer."))

    @app.get("/api/runs", tags=["query"], summary="Recent query runs")
    def recent_runs(request: Request, limit: int = 20):
        return st(request).runs.recent(min(max(limit, 1), 100))

    @app.get("/api/runs/{run_id}", tags=["query"], response_model=QueryResponse)
    def get_run(request: Request, run_id: str):
        r = st(request).runs.get(run_id)
        if r is None:
            raise HTTPException(404, "Run not found")
        return r

    # ------------------------------------------------------------------ evaluation
    @app.get("/api/datasets", tags=["evaluation"])
    def datasets():
        return list_datasets(settings.data_dir / "datasets")

    @app.post("/api/evaluations", tags=["evaluation"], summary="Start a benchmark run in the background")
    def start_evaluation(request: Request, req: EvaluationRequest):
        names = {d["name"] for d in list_datasets(settings.data_dir / "datasets") if d["valid"]}
        if req.dataset not in names:
            raise HTTPException(404, f"Dataset '{req.dataset}' not found or invalid")
        unknown = [m for m in req.methods if m not in VARIANTS]
        if unknown:
            raise HTTPException(422, f"Unknown method(s) {unknown}; choose from {list(VARIANTS)}")
        return st(request).evals.start(req)

    @app.get("/api/variants", tags=["evaluation"], summary="Systems that can be evaluated (baseline and ablations)")
    def variants():
        return [{"name": k, "label": VARIANT_LABELS[k], "pipeline": v[0], "settings": v[1]} for k, v in VARIANTS.items()]

    @app.get("/api/evaluations", tags=["evaluation"])
    def list_evaluations(request: Request):
        return st(request).evals.list()

    @app.get("/api/evaluations/{job_id}", tags=["evaluation"])
    def get_evaluation(request: Request, job_id: str):
        return st(request).evals.get(job_id)

    @app.post("/api/evaluations/{job_id}/cancel", tags=["evaluation"])
    def cancel_evaluation(request: Request, job_id: str):
        return st(request).evals.cancel(job_id)

    @app.get("/api/evaluations/{job_id}/export", tags=["evaluation"], summary="Download results as JSON or CSV")
    def export_evaluation(request: Request, job_id: str, format: str = "json"):
        if format not in ("json", "csv"):
            raise HTTPException(422, "format must be 'json' or 'csv'")
        path = st(request).evals._result_path(job_id).with_name(f"results.{format}")
        if not path.exists():
            raise HTTPException(404, "No result file (the run may still be in progress or failed)")
        return FileResponse(path, filename=f"evaluation-{job_id}.{format}",
                            media_type="application/json" if format == "json" else "text/csv")

    return app


app = create_app()
