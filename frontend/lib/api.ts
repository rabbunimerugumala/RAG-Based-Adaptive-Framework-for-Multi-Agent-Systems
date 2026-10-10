// Thin typed client for the FastAPI backend. All data shown in the UI comes from these calls.
import type {
  CompareResponse,
  DatasetInfo,
  DocumentInfo,
  EvalJob,
  Health,
  Method,
  QueryResponse,
  RunSummary,
  UploadResult,
} from "./types";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message: string, public status?: number) {
    super(message);
  }
}

async function request<T>(path: string, init: RequestInit = {}, timeoutMs = 30_000): Promise<T> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(new DOMException("Request timed out", "TimeoutError")), timeoutMs);
  // forward an external abort (e.g. the user pressing Cancel)
  init.signal?.addEventListener("abort", () => ctrl.abort(init.signal?.reason));
  try {
    const res = await fetch(`${API_URL}${path}`, { ...init, signal: ctrl.signal });
    if (!res.ok) {
      let detail = `${res.status} ${res.statusText}`;
      try {
        const body = await res.json();
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
      } catch {
        /* non-JSON error body */
      }
      throw new ApiError(detail, res.status);
    }
    return (await res.json()) as T;
  } catch (err) {
    if (err instanceof ApiError) throw err;
    if (err instanceof DOMException && (err.name === "AbortError" || err.name === "TimeoutError")) {
      throw new ApiError(err.name === "TimeoutError" ? "The request timed out." : "Request cancelled.");
    }
    if (ctrl.signal.aborted) {
      const reason = ctrl.signal.reason;
      throw new ApiError(reason instanceof DOMException && reason.name === "TimeoutError" ? "The request timed out." : "Request cancelled.");
    }
    throw new ApiError(`Cannot reach the backend at ${API_URL}. Is it running?`);
  } finally {
    clearTimeout(timer);
  }
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

// LLM runs (especially adaptive retries on CPU) can be slow, so queries get a long timeout.
const QUERY_TIMEOUT = 10 * 60_000;

export const api = {
  health: () => request<Health>("/api/health", {}, 8_000),
  documents: () => request<DocumentInfo[]>("/api/documents"),
  deleteDocument: (id: string) => request<{ deleted: string }>(`/api/documents/${encodeURIComponent(id)}`, { method: "DELETE" }),
  query: (question: string, method: Method, signal?: AbortSignal) =>
    request<QueryResponse>("/api/query", { ...json({ question, method }), signal }, QUERY_TIMEOUT),
  compare: (question: string, signal?: AbortSignal) =>
    request<CompareResponse>("/api/compare", { ...json({ question }), signal }, QUERY_TIMEOUT),
  runs: (limit = 20) => request<RunSummary[]>(`/api/runs?limit=${limit}`),
  run: (id: string) => request<QueryResponse>(`/api/runs/${encodeURIComponent(id)}`),
  datasets: () => request<DatasetInfo[]>("/api/datasets", {}, 60_000),
  evaluations: () => request<EvalJob[]>("/api/evaluations"),
  evaluation: (id: string) => request<EvalJob>(`/api/evaluations/${encodeURIComponent(id)}`),
  startEvaluation: (dataset: string, methods: Method[], limit: number | null) =>
    request<EvalJob>("/api/evaluations", json({ dataset, methods, limit }), 60_000),
  cancelEvaluation: (id: string) => request<unknown>(`/api/evaluations/${encodeURIComponent(id)}/cancel`, { method: "POST" }),
  exportUrl: (id: string, format: "csv" | "json") => `${API_URL}/api/evaluations/${encodeURIComponent(id)}/export?format=${format}`,
};

/** Upload with real progress events (fetch cannot report upload progress). */
export function uploadFiles(files: File[], onProgress: (fraction: number) => void): Promise<UploadResult[]> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f, f.name));
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_URL}/api/documents`);
    xhr.timeout = 10 * 60_000;
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      try {
        const body = JSON.parse(xhr.responseText);
        if (xhr.status >= 200 && xhr.status < 300) resolve(body as UploadResult[]);
        else reject(new ApiError(typeof body.detail === "string" ? body.detail : `Upload failed (${xhr.status})`, xhr.status));
      } catch {
        reject(new ApiError(`Upload failed (${xhr.status})`, xhr.status));
      }
    };
    xhr.onerror = () => reject(new ApiError(`Cannot reach the backend at ${API_URL}. Is it running?`));
    xhr.ontimeout = () => reject(new ApiError("Upload timed out."));
    xhr.send(form);
  });
}

export const fmt = {
  pct: (v: number | null | undefined, digits = 0) => (v == null ? "n/a" : `${(v * 100).toFixed(digits)}%`),
  num: (v: number | null | undefined, digits = 2) => (v == null ? "n/a" : v.toFixed(digits)),
  ms: (v: number | null | undefined) =>
    v == null ? "n/a" : v >= 1000 ? `${(v / 1000).toFixed(v >= 10_000 ? 0 : 1)} s` : `${Math.round(v)} ms`,
  bytes: (b: number) => (b >= 1048576 ? `${(b / 1048576).toFixed(1)} MB` : b >= 1024 ? `${(b / 1024).toFixed(0)} KB` : `${b} B`),
  date: (iso: string) => new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }),
};
