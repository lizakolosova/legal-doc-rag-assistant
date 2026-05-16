import axios from 'axios'

const BASE_URL = (import.meta.env['VITE_API_URL'] as string | undefined) ?? ''

const api = axios.create({ baseURL: BASE_URL })

// ── Types (mirror backend schemas exactly) ─────────────────────────────────

export type DocumentStatus = 'pending' | 'processing' | 'ready' | 'failed'

export interface DocumentResponse {
  document_id: string
  filename: string
  status: DocumentStatus
  num_chunks: number | null
  upload_time: string
  file_size_bytes: number
  error_message: string | null
}

export interface IngestResponse {
  document_id: string
  status: DocumentStatus
}

export interface Citation {
  index: number
  source_file: string
  page_number: number
  chunk_text: string
  section_header: string | null
}

export interface RetrievedChunk {
  chunk_id: string
  document_id: string
  text: string
  score: number
  source_file: string
  page_number: number
  section_header: string | null
}

export interface QueryRequest {
  question: string
  document_ids?: string[] | null
  top_k?: number
  use_reranker?: boolean
}

export interface QueryResponse {
  answer: string
  citations: Citation[]
  chunks: RetrievedChunk[]
  retrieval_method: string
  latency_ms: number
}

export interface EvalResult {
  question_id: string
  question: string
  expected_answer: string
  generated_answer: string
  retrieved_chunks: RetrievedChunk[]
  context_precision: number
  context_recall: number
  faithfulness: number | null
  answer_relevancy: number | null
  latency_ms: number
}

export interface EvalRunSummaryResponse {
  run_id: string
  created_at: string
  summary: Record<string, number>
  config: Record<string, unknown>
}

export interface EvalRunListResponse {
  runs: EvalRunSummaryResponse[]
}

export interface EvalRunDetailResponse {
  run_id: string
  created_at: string
  summary: Record<string, number>
  config: Record<string, unknown>
  results: EvalResult[]
}

// ── Endpoints ──────────────────────────────────────────────────────────────

export async function uploadDocument(
  file: File,
  onProgress: (pct: number) => void,
): Promise<IngestResponse> {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post<IngestResponse>('/api/documents', form, {
    onUploadProgress: (e) => {
      if (e.total) onProgress(Math.round((e.loaded / e.total) * 100))
    },
  })
  return data
}

export async function listDocuments(): Promise<DocumentResponse[]> {
  const { data } = await api.get<DocumentResponse[]>('/api/documents')
  return data
}

export async function deleteDocument(documentId: string): Promise<void> {
  await api.delete(`/api/documents/${documentId}`)
}

export async function queryDocuments(req: QueryRequest): Promise<QueryResponse> {
  const { data } = await api.post<QueryResponse>('/api/query', req)
  return data
}

export async function streamQuery(
  req: QueryRequest,
  onDelta: (delta: string) => void,
): Promise<Citation[]> {
  const url = `${BASE_URL}/api/query/stream`
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })

  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || `HTTP ${response.status}`)
  }

  const reader = response.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let citations: Citation[] = []

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''
    for (const line of lines) {
      if (!line.trim()) continue
      const parsed = JSON.parse(line) as
        | { delta: string }
        | { done: true; citations: Citation[] }
      if ('delta' in parsed) {
        onDelta(parsed.delta)
      } else if ('done' in parsed) {
        citations = parsed.citations
      }
    }
  }

  return citations
}

export async function runEvaluation(): Promise<EvalRunSummaryResponse> {
  const { data } = await api.post<EvalRunSummaryResponse>('/api/eval/run')
  return data
}

export async function listEvalRuns(): Promise<EvalRunListResponse> {
  const { data } = await api.get<EvalRunListResponse>('/api/eval/results')
  return data
}

export async function getEvalRun(runId: string): Promise<EvalRunDetailResponse> {
  const { data } = await api.get<EvalRunDetailResponse>(`/api/eval/results/${runId}`)
  return data
}
