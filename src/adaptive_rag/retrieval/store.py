"""Local knowledge base: Qdrant (embedded/local mode) for dense vectors + an in-memory BM25 index.

Layout of a knowledge-base directory::

    <root>/registry.json   document metadata + the embedder the index was built with
    <root>/files/          original uploaded files (used to re-index if the embedder changes)
    <root>/qdrant/         Qdrant local-mode storage (vectors + chunk payloads)

The user knowledge base and each benchmark dataset use *separate* roots, so benchmark corpora never
mix with user documents. Qdrant local mode locks its directory: only one process may open a root.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from adaptive_rag.ingestion.chunking import chunk_pages
from adaptive_rag.ingestion.extract import (
    ExtractionError,
    PageText,
    extract_pages,
    file_type_for,
    safe_filename,
)
from adaptive_rag.retrieval.bm25 import BM25Index
from adaptive_rag.retrieval.embeddings import Embedder
from adaptive_rag.schemas import Chunk, DocumentInfo, UploadResult

COLLECTION = "chunks"


class KnowledgeBase:
    def __init__(self, root: Path, embedder: Embedder, chunk_size: int, chunk_overlap: int):
        self.root = Path(root)
        self.embedder = embedder
        self.chunk_size, self.chunk_overlap = chunk_size, chunk_overlap
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "files").mkdir(exist_ok=True)
        self._lock = threading.RLock()
        self._registry_path = self.root / "registry.json"
        self._defer_save = False
        self._registry = self._load_registry()
        self._client = QdrantClient(path=str(self.root / "qdrant"))
        self._chunks: dict[str, Chunk] = {}
        self._bm25 = BM25Index()
        self._bm25_dirty = True
        self.reindexed_on_startup = False
        self._open_collection()

    # ------------------------------------------------------------------ persistence
    def _load_registry(self) -> dict:
        if self._registry_path.exists():
            return json.loads(self._registry_path.read_text(encoding="utf-8"))
        return {"embedder": None, "dim": None, "chunk_size": None, "chunk_overlap": None, "documents": {}}

    def _save_registry(self) -> None:
        if self._defer_save:
            return
        tmp = self._registry_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._registry, indent=2), encoding="utf-8")
        for attempt in range(10):  # Windows: antivirus/indexer can briefly lock the target file
            try:
                tmp.replace(self._registry_path)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.05 * (attempt + 1))

    def add_texts(self, docs: list[dict]) -> int:
        """Bulk-index raw-text documents ({doc_id, title, text}); the registry is written once at the end."""
        added = 0
        with self._lock:
            self._defer_save = True
            try:
                for d in docs:
                    if d["doc_id"] not in self._registry["documents"]:
                        self.add_text(d["doc_id"], d.get("title") or d["doc_id"], d["text"])
                        added += 1
            finally:
                self._defer_save = False
                self._save_registry()
        return added

    def _open_collection(self) -> None:
        settings_changed = (
            self._registry["embedder"] != self.embedder.name
            or self._registry["chunk_size"] != self.chunk_size
            or self._registry["chunk_overlap"] != self.chunk_overlap
        )
        exists = self._client.collection_exists(COLLECTION)
        if exists and not settings_changed:
            self._load_chunks()
            return
        if exists:
            # qdrant-client local mode reloads persisted points after delete_collection + create_collection,
            # so the storage folder is removed and re-opened instead.
            self._client.close()
            shutil.rmtree(self.root / "qdrant", ignore_errors=True)
            self._client = QdrantClient(path=str(self.root / "qdrant"))
        self._client.create_collection(
            COLLECTION, vectors_config=VectorParams(size=self.embedder.dim, distance=Distance.COSINE)
        )
        old_docs = self._registry["documents"]
        self._registry.update(
            embedder=self.embedder.name, dim=self.embedder.dim, chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap, documents={},
        )
        self._save_registry()
        # Index settings changed: rebuild from the stored original files.
        for info in old_docs.values():
            path = self.root / "files" / (info.get("stored_name") or "")
            if info.get("stored_name") and path.exists():
                self.add_bytes(info["filename"], path.read_bytes(), _stored_name=info["stored_name"])
                self.reindexed_on_startup = True

    def _load_chunks(self) -> None:
        offset = None
        while True:
            points, offset = self._client.scroll(COLLECTION, limit=512, offset=offset, with_payload=True)
            for p in points:
                c = Chunk(**p.payload)
                self._chunks[c.chunk_id] = c
            if offset is None:
                break
        self._bm25_dirty = True

    def close(self) -> None:
        with self._lock:
            self._client.close()

    # ------------------------------------------------------------------ documents
    def list_documents(self) -> list[DocumentInfo]:
        docs = [DocumentInfo(**{k: v for k, v in d.items() if k != "stored_name"})
                for d in self._registry["documents"].values()]
        return sorted(docs, key=lambda d: d.created_at, reverse=True)

    def get_document(self, doc_id: str) -> DocumentInfo | None:
        d = self._registry["documents"].get(doc_id)
        return DocumentInfo(**{k: v for k, v in d.items() if k != "stored_name"}) if d else None

    @property
    def num_documents(self) -> int:
        return len(self._registry["documents"])

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    def add_bytes(self, filename: str, data: bytes, *, _stored_name: str | None = None) -> UploadResult:
        """Validate, extract, chunk, embed and index a file. Duplicate content is detected by SHA-256."""
        filename = safe_filename(filename)
        try:
            file_type = file_type_for(filename)
        except ExtractionError as exc:
            return UploadResult(filename=filename, status="failed", message=str(exc))
        sha = hashlib.sha256(data).hexdigest()
        with self._lock:
            for d in self._registry["documents"].values():
                if d["sha256"] == sha:
                    return UploadResult(
                        filename=filename, status="duplicate",
                        message=f"Identical content is already indexed as '{d['filename']}'.",
                        document=self.get_document(d["doc_id"]),
                    )
            try:
                pages = extract_pages(data, file_type)
            except ExtractionError as exc:
                return UploadResult(filename=filename, status="failed", message=str(exc))
            doc_id = sha[:12]
            return self._index(doc_id, filename, file_type, data, sha, pages, _stored_name)

    def add_text(self, doc_id: str, title: str, text: str) -> DocumentInfo:
        """Index raw text (used for benchmark corpora). No original file is stored."""
        data = text.encode("utf-8")
        with self._lock:
            if doc_id in self._registry["documents"]:
                return self.get_document(doc_id)  # type: ignore[return-value]
            filename = safe_filename(title) if title else doc_id
            res = self._index(doc_id, filename, "txt", data, hashlib.sha256(data).hexdigest(),
                              [PageText(None, text)], None, store_file=False)
            return res.document  # type: ignore[return-value]

    def _index(self, doc_id, filename, file_type, data, sha, pages, stored_name, store_file=True) -> UploadResult:
        chunks = chunk_pages(pages, doc_id, filename, self.chunk_size, self.chunk_overlap)
        if not chunks:
            return UploadResult(filename=filename, status="failed", message="No indexable text after chunking.")
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        points = [
            PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)), vector=v.tolist(), payload=c.model_dump())
            for c, v in zip(chunks, vectors)
        ]
        for i in range(0, len(points), 256):
            self._client.upsert(COLLECTION, points=points[i : i + 256])
        if store_file:
            stored_name = stored_name or f"{doc_id}_{filename}"
            (self.root / "files" / stored_name).write_bytes(data)
        pdf_pages = [p.page for p in pages if p.page is not None]
        info = DocumentInfo(
            doc_id=doc_id, filename=filename, file_type=file_type, size_bytes=len(data), sha256=sha,
            num_chunks=len(chunks), num_pages=max(pdf_pages) if pdf_pages else None,
            num_characters=sum(len(p.text) for p in pages),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self._registry["documents"][doc_id] = {**info.model_dump(), "stored_name": stored_name if store_file else None}
        self._save_registry()
        for c in chunks:
            self._chunks[c.chunk_id] = c
        self._bm25_dirty = True
        return UploadResult(filename=filename, status="indexed",
                            message=f"Indexed {len(chunks)} chunks.", document=info)

    def delete_document(self, doc_id: str) -> bool:
        with self._lock:
            info = self._registry["documents"].pop(doc_id, None)
            if info is None:
                return False
            self._client.delete(
                COLLECTION,
                points_selector=FilterSelector(
                    filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
                ),
            )
            if info.get("stored_name"):
                (self.root / "files" / info["stored_name"]).unlink(missing_ok=True)
            self._save_registry()
            self._chunks = {k: c for k, c in self._chunks.items() if c.doc_id != doc_id}
            self._bm25_dirty = True
            return True

    # ------------------------------------------------------------------ search
    def get_chunk(self, chunk_id: str) -> Chunk:
        return self._chunks[chunk_id]

    def dense_search(self, query: str, k: int) -> list[tuple[str, float]]:
        if not self._chunks:
            return []
        vec = self.embedder.embed_query(query)
        with self._lock:
            res = self._client.query_points(COLLECTION, query=vec.tolist(), limit=k, with_payload=["chunk_id"])
        return [(p.payload["chunk_id"], float(p.score)) for p in res.points]

    def bm25_search(self, query: str, k: int) -> list[tuple[str, float]]:
        with self._lock:
            if self._bm25_dirty:
                self._bm25.build({cid: c.text for cid, c in self._chunks.items()})
                self._bm25_dirty = False
            return self._bm25.search(query, k)
