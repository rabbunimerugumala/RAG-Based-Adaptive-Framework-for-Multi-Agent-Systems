// TypeScript mirrors of the backend Pydantic schemas (src/adaptive_rag/schemas.py).

export type Method = "standard" | "adaptive";

export interface DocumentInfo {
  doc_id: string;
  filename: string;
  file_type: string;
  size_bytes: number;
  sha256: string;
  num_chunks: number;
  num_pages: number | null;
  num_characters: number;
  status: "indexed";
  created_at: string;
}

export interface UploadResult {
  filename: string;
  status: "indexed" | "duplicate" | "failed";
  message: string;
  document: DocumentInfo | null;
}

export interface Evidence {
  ref: number;
  chunk_id: string;
  doc_id: string;
  filename: string;
  page: number | null;
  chunk_index: number;
  text: string;
  rrf_score: number;
  dense_score: number | null;
  bm25_score: number | null;
  rerank_score: number | null;
  keyword_overlap: number;
  matched_queries: string[];
  retrieval_rank: number | null;
  /** Evidence Agent judge score: log P(Yes) - log P(No). Relevance, not proof of truth. */
  judge_score: number | null;
  candidate_answer: string | null;
  filter_reason: string | null;
}

export interface EvidenceFilterReport {
  mode: string;
  pool_size: number;
  duplicates_removed: number;
  threshold: number | null;
  mean: number | null;
  std: number | null;
  n_sigma: number;
  retained: number;
  filtered: number;
  fallback: string | null;
  judge_calls: number;
}

export type ClaimLabel = "supported" | "partially_supported" | "unsupported" | "contradicted";

export interface ClaimCheck {
  claim: string;
  label: ClaimLabel;
  lexical_support: number;
  nli_entailment: number | null;
  nli_contradiction: number | null;
  best_evidence_ref: number | null;
  cited_refs: number[];
  reason: string;
}

export interface VerificationResult {
  status: ClaimLabel | "not_applicable";
  support_score: number | null;
  citation_consistency: number | null;
  claims: ClaimCheck[];
  unsupported_claims: string[];
  contradicted_claims: string[];
  method: string;
  summary: string;
  llm_calls?: number;
}

export interface ConfidenceAssessment {
  score: number | null;
  components: Record<string, number | null>;
  weights: Record<string, number>;
  missing_components: string[];
  decision: "accept" | "retry" | "abstain";
  reason: string;
  retry_strategy: string | null;
  note: string;
}

export interface TraceStep {
  step: string;
  title: string;
  attempt: number;
  status: "completed" | "skipped" | "failed";
  duration_ms: number;
  input: Record<string, unknown>;
  output: Record<string, unknown>;
  decision: string | null;
  note: string | null;
}

export interface Citation {
  ref: number;
  chunk_id: string;
  doc_id: string;
  filename: string;
  page: number | null;
  chunk_index: number;
  valid: boolean;
}

export interface RunStats {
  latency_ms: number;
  llm_calls: number;
  nli_calls: number;
  retrieval_attempts: number;
  retries: number;
  analysis_latency_ms: number;
}

export type RunStatus = "answered" | "abstained" | "no_documents" | "no_retrieval_needed" | "error";

export interface QueryResponse {
  run_id: string;
  method: Method;
  question: string;
  answer: string;
  status: RunStatus;
  citations: Citation[];
  evidence: Evidence[];
  verification: VerificationResult | null;
  verification_used_in_decision: boolean;
  confidence: ConfidenceAssessment | null;
  route?: string | null;
  support_level: "verified" | "partial" | "unverified" | "none";
  unresolved_claims: string[];
  filtered_evidence: Evidence[];
  evidence_filter: EvidenceFilterReport | null;
  trace: TraceStep[];
  stats: RunStats;
  mode: string;
  models: Record<string, string>;
  created_at: string;
}

export interface CompareResponse {
  question: string;
  standard: QueryResponse;
  adaptive: QueryResponse;
  note: string;
}

export interface RunSummary {
  run_id: string;
  method: Method;
  question: string;
  status: RunStatus;
  confidence: number | null;
  verification: string | null;
  latency_ms: number;
  created_at: string;
  mode: string;
}

export interface ModelStatus {
  name: string;
  ready: boolean;
  detail: string;
}

export interface Health {
  status: string;
  version: string;
  mode: "dev" | "real";
  dev_mode_warning: string | null;
  models: Record<"embedder" | "generator" | "nli" | "reranker", ModelStatus>;
  knowledge_base: { documents: number; chunks: number; embedder: string; reindexed_on_startup: boolean };
  config: Record<string, unknown> & {
    top_k: number;
    max_retries: number;
    accept_threshold: number;
    abstain_threshold: number;
  };
}

export interface DatasetInfo {
  name: string;
  title: string;
  description: string;
  source: string;
  license: string;
  synthetic: boolean;
  num_questions: number;
  num_documents: number;
  num_answerable: number;
  num_unanswerable: number;
  valid: boolean;
  issues: string[];
}

export interface Variant {
  name: string;
  label: string;
  pipeline: Method;
  settings: Record<string, unknown>;
}

export interface EvalRow {
  question_id: string;
  method: string;
  question: string;
  answerable: boolean;
  type: string;
  status: string;
  abstained: boolean;
  correct: boolean;
  em: number;
  f1: number;
  cited_gold: boolean | null;
  evidence_has_answer: boolean | null;
  verification: string | null;
  support_score: number | null;
  support_level?: string;
  evidence_precision?: number | null;
  grounded?: boolean | null;
  passages_used?: number;
  passages_filtered?: number;
  confidence: number | null;
  latency_ms: number;
  llm_calls: number;
  nli_calls: number;
  retrieval_attempts: number;
  answer: string;
  gold_answers: string[];
  run_id: string | null;
  error: string | null;
}

export type MetricSummary = Record<string, number | null>;

export interface EvalResult {
  run_id: string;
  dataset: DatasetInfo & { num_questions_available: number };
  methods: string[];
  method_labels?: Record<string, string>;
  consistency?: Record<string, { n: number; same_status: number | null; same_answer: number | null }>;
  software?: { adaptive_rag: string; git_commit: string | null };
  sample: { limit: number | null; seed: number; num_questions: number; question_ids: string[] };
  mode: string;
  dev_mode_warning: string | null;
  models: Record<string, string>;
  config: Record<string, unknown>;
  started_at: string;
  finished_at: string;
  runtime_s: number;
  summary: Record<string, MetricSummary>;
}

export interface EvalJob {
  job_id: string;
  dataset: string;
  methods: string[];
  limit: number | null;
  status: "running" | "completed" | "failed" | "cancelled";
  done: number;
  total: number | null;
  rows?: EvalRow[];
  error: string | null;
  started_at: string;
  mode?: string;
  result?: EvalResult | null;
}
