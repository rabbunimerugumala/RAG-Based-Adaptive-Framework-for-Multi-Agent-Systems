"use client";

import { FileText, Trash2, Upload } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { useHealth } from "@/components/providers";
import { Badge, Button, Card, CardHeader, EmptyState, Notice, PageHeader, Spinner } from "@/components/ui";
import { api, fmt, uploadFiles } from "@/lib/api";
import type { DocumentInfo, UploadResult } from "@/lib/types";

const ACCEPT = ".pdf,.docx,.txt,.md,.markdown";

export default function DocumentsPage() {
  const { refresh: refreshHealth } = useHealth();
  const [docs, setDocs] = useState<DocumentInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<number | null>(null);
  const [results, setResults] = useState<UploadResult[]>([]);
  const [dragging, setDragging] = useState(false);
  const [confirm, setConfirm] = useState<DocumentInfo | null>(null);
  const [deleting, setDeleting] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  const load = useCallback(() => {
    api.documents().then((d) => { setDocs(d); setError(null); }).catch((e: Error) => setError(e.message));
  }, []);
  useEffect(load, [load]);

  async function handleFiles(list: FileList | null) {
    if (!list?.length) return;
    setResults([]);
    setProgress(0);
    try {
      setResults(await uploadFiles(Array.from(list), setProgress));
      load();
      refreshHealth();
    } catch (e) {
      setResults([{ filename: "Upload", status: "failed", message: (e as Error).message, document: null }]);
    } finally {
      setProgress(null);
      if (input.current) input.current.value = "";
    }
  }

  async function doDelete() {
    if (!confirm) return;
    setDeleting(true);
    try {
      await api.deleteDocument(confirm.doc_id);
      setConfirm(null);
      load();
      refreshHealth();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setDeleting(false);
    }
  }

  const uploading = progress != null;

  return (
    <>
      <PageHeader title="Document Library" description="Your private knowledge base. Files are processed locally: text extraction, chunking, embedding and indexing in Qdrant (local mode) and BM25." />

      <Card
        className={`border-dashed p-6 text-center transition-colors ${dragging ? "border-accent bg-accent-soft" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); if (!uploading) handleFiles(e.dataTransfer.files); }}
      >
        <Upload className="mx-auto size-6 text-subtle" aria-hidden />
        <p className="mt-2 text-sm font-medium text-fg">Drop files here or choose files to upload</p>
        <p className="mt-1 text-xs text-muted">PDF (with a text layer), DOCX, TXT, Markdown · scanned PDFs are not supported (no OCR) · max size set by the backend (default 20 MB)</p>
        <input ref={input} type="file" multiple accept={ACCEPT} className="sr-only" id="file-input" onChange={(e) => handleFiles(e.target.files)} disabled={uploading} />
        <Button className="mt-4" variant="primary" onClick={() => input.current?.click()} loading={uploading}>
          {uploading ? (progress! < 1 ? `Uploading ${Math.round(progress! * 100)}%` : "Indexing…") : "Choose files"}
        </Button>
        {uploading && (
          <div className="mx-auto mt-3 h-1.5 max-w-xs overflow-hidden rounded-full bg-surface-2" role="progressbar" aria-valuenow={Math.round(progress! * 100)} aria-valuemin={0} aria-valuemax={100} aria-label="Upload progress">
            <div className="h-full bg-accent transition-all" style={{ width: `${progress! * 100}%` }} />
          </div>
        )}
      </Card>

      {results.length > 0 && (
        <ul className="mt-4 space-y-2">
          {results.map((r, i) => (
            <li key={i}>
              <Notice tone={r.status === "indexed" ? "success" : r.status === "duplicate" ? "warning" : "danger"}
                title={`${r.filename}: ${r.status}`}>{r.message}</Notice>
            </li>
          ))}
        </ul>
      )}

      {error && <Notice tone="danger" title="Could not load documents" className="mt-4">{error}</Notice>}

      <Card className="mt-6">
        <CardHeader title="Indexed documents" description={docs ? `${docs.length} document(s)` : undefined} />
        {docs == null ? <div className="px-5 py-6">{!error && <Spinner label="Loading…" />}</div> : docs.length === 0 ? (
          <EmptyState icon={<FileText className="size-8" />} title="No documents yet">
            Upload a file above. A few sample documents are provided in <code className="font-mono text-xs">data/sample_documents/</code>.
          </EmptyState>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-border text-xs text-subtle">
                <tr>
                  <th className="px-5 py-2 font-medium">File</th><th className="px-3 py-2 font-medium">Type</th>
                  <th className="px-3 py-2 font-medium">Size</th><th className="px-3 py-2 font-medium">Pages</th>
                  <th className="px-3 py-2 font-medium">Chunks</th><th className="px-3 py-2 font-medium">Status</th>
                  <th className="px-3 py-2 font-medium">Added</th><th className="px-5 py-2"><span className="sr-only">Actions</span></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {docs.map((d) => (
                  <tr key={d.doc_id}>
                    <td className="max-w-[260px] truncate px-5 py-3 font-medium text-fg" title={d.filename}>{d.filename}</td>
                    <td className="px-3 py-3 uppercase text-muted">{d.file_type}</td>
                    <td className="px-3 py-3 tabular-nums text-muted">{fmt.bytes(d.size_bytes)}</td>
                    <td className="px-3 py-3 tabular-nums text-muted">{d.num_pages ?? "–"}</td>
                    <td className="px-3 py-3 tabular-nums text-muted">{d.num_chunks}</td>
                    <td className="px-3 py-3"><Badge tone="success">{d.status}</Badge></td>
                    <td className="whitespace-nowrap px-3 py-3 text-muted">{fmt.date(d.created_at)}</td>
                    <td className="px-5 py-3 text-right">
                      <Button size="sm" variant="ghost" onClick={() => setConfirm(d)} aria-label={`Delete ${d.filename}`}>
                        <Trash2 className="size-4" />
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {confirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4" role="dialog" aria-modal="true" aria-labelledby="del-title">
          <div className="absolute inset-0 bg-black/40" onClick={() => !deleting && setConfirm(null)} />
          <Card className="relative w-full max-w-md p-5 shadow-xl">
            <h2 id="del-title" className="font-semibold text-fg">Delete document?</h2>
            <p className="mt-2 text-sm text-muted">
              <span className="font-medium text-fg">{confirm.filename}</span> and its {confirm.num_chunks} indexed chunk(s) will be removed from the knowledge base. This cannot be undone.
            </p>
            <div className="mt-5 flex justify-end gap-2">
              <Button onClick={() => setConfirm(null)} disabled={deleting} autoFocus>Cancel</Button>
              <Button variant="danger" onClick={doDelete} loading={deleting}>Delete</Button>
            </div>
          </Card>
        </div>
      )}
    </>
  );
}
