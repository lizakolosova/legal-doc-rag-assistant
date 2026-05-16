import { useCallback, useEffect, useRef, useState } from 'react'
import { deleteDocument } from '../hooks/useApi'
import type { DocumentResponse, DocumentStatus } from '../hooks/useApi'

interface Props {
  documents: DocumentResponse[]
  onRefresh: () => void
  selectedIds: string[]
  onToggleSelect: (id: string) => void
}

const STATUS_BADGE: Record<DocumentStatus, string> = {
  ready: 'bg-green-100 text-green-800',
  failed: 'bg-red-100 text-red-800',
  processing: 'bg-amber-100 text-amber-800',
  pending: 'bg-slate-100 text-slate-600',
}

function needsPolling(docs: DocumentResponse[]): boolean {
  return docs.some((d) => d.status === 'pending' || d.status === 'processing')
}

export default function DocumentList({ documents, onRefresh, selectedIds, onToggleSelect }: Props) {
  const [deleting, setDeleting] = useState<string | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    if (needsPolling(documents)) {
      if (!pollRef.current) {
        pollRef.current = setInterval(onRefresh, 3000)
      }
    } else {
      if (pollRef.current) {
        clearInterval(pollRef.current)
        pollRef.current = null
      }
    }
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current)
        pollRef.current = null
      }
    }
  }, [documents, onRefresh])

  const handleDelete = useCallback(async (doc: DocumentResponse) => {
    if (!confirm(`Delete "${doc.filename}"?`)) return
    setDeleting(doc.document_id)
    try {
      await deleteDocument(doc.document_id)
      onRefresh()
    } catch {
      // silently refresh so list stays consistent
      onRefresh()
    } finally {
      setDeleting(null)
    }
  }, [onRefresh])

  if (documents.length === 0) {
    return (
      <p className="px-4 py-3 text-xs text-slate-400 italic">No documents uploaded yet.</p>
    )
  }

  return (
    <ul className="divide-y divide-slate-100">
      {documents.map((doc) => {
        const selected = selectedIds.includes(doc.document_id)
        return (
          <li
            key={doc.document_id}
            className={`group flex items-start gap-2 px-3 py-2.5 text-sm transition-colors
              ${selected ? 'bg-blue-50' : 'hover:bg-slate-50'}`}
          >
            <input
              type="checkbox"
              className="mt-0.5 h-3.5 w-3.5 cursor-pointer accent-blue-600"
              checked={selected}
              onChange={() => onToggleSelect(doc.document_id)}
              title="Include in query scope"
            />
            <div className="min-w-0 flex-1">
              <p className="truncate font-medium text-slate-800 text-xs" title={doc.filename}>
                {doc.filename}
              </p>
              <div className="mt-1 flex items-center gap-2">
                <span className={`inline-block rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${STATUS_BADGE[doc.status]}`}>
                  {doc.status}
                </span>
                {doc.num_chunks != null && (
                  <span className="text-[10px] text-slate-400">{doc.num_chunks} chunks</span>
                )}
              </div>
              {doc.error_message && (
                <p className="mt-1 text-[10px] text-red-600 leading-tight">{doc.error_message}</p>
              )}
            </div>
            <button
              onClick={() => void handleDelete(doc)}
              disabled={deleting === doc.document_id}
              className="mt-0.5 shrink-0 rounded p-1 text-slate-300 opacity-0 transition-opacity
                hover:bg-red-50 hover:text-red-500 group-hover:opacity-100
                disabled:cursor-not-allowed disabled:opacity-50"
              title="Delete document"
            >
              <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
            </button>
          </li>
        )
      })}
    </ul>
  )
}

