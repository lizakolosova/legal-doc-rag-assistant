import { useCallback, useEffect, useState } from 'react'
import { listDocuments } from './hooks/useApi'
import type { DocumentResponse } from './hooks/useApi'
import ChatInterface from './components/ChatInterface'
import DocumentList from './components/DocumentList'
import DocumentUpload from './components/DocumentUpload'
import EvalDashboard from './components/EvalDashboard'

type Tab = 'chat' | 'eval'

export default function App() {
  const [documents, setDocuments] = useState<DocumentResponse[]>([])
  const [tab, setTab] = useState<Tab>('chat')
  const [sidebarSelectedIds, setSidebarSelectedIds] = useState<string[]>([])

  const refreshDocuments = useCallback(async () => {
    try {
      const docs = await listDocuments()
      setDocuments(docs)
    } catch {
      // backend may not be up yet; silently retry on next cycle
    }
  }, [])

  useEffect(() => { void refreshDocuments() }, [refreshDocuments])

  const toggleSidebarSelect = (id: string) => {
    setSidebarSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    )
  }

  return (
    <div className="flex h-screen overflow-hidden bg-slate-100 font-sans antialiased">
      {/* ── Sidebar ───────────────────────────────────────────────────── */}
      <aside className="flex w-[280px] shrink-0 flex-col border-r border-slate-200 bg-white">
        {/* logo / header */}
        <div className="flex items-center gap-2.5 border-b border-slate-200 px-4 py-4">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-blue-600">
            <svg className="h-4 w-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <span className="text-sm font-semibold text-slate-800 tracking-tight">Legal Doc Assistant</span>
        </div>

        {/* upload zone */}
        <div className="border-b border-slate-100 pt-3">
          <DocumentUpload onUploadSuccess={refreshDocuments} />
        </div>

        {/* document list */}
        <div className="flex-1 overflow-y-auto">
          <p className="px-4 py-2 text-[10px] font-semibold uppercase tracking-widest text-slate-400">
            Documents
          </p>
          <DocumentList
            documents={documents}
            onRefresh={refreshDocuments}
            selectedIds={sidebarSelectedIds}
            onToggleSelect={toggleSidebarSelect}
          />
        </div>

        {/* footer */}
        <div className="border-t border-slate-100 px-4 py-3">
          <p className="text-[10px] text-slate-400">{documents.length} document{documents.length !== 1 ? 's' : ''} · RAG pipeline</p>
        </div>
      </aside>

      {/* ── Main area ─────────────────────────────────────────────────── */}
      <main className="flex flex-1 flex-col overflow-hidden">
        {/* tab bar */}
        <div className="flex items-center gap-1 border-b border-slate-200 bg-white px-4">
          {(['chat', 'eval'] as Tab[]).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`
                px-4 py-3 text-sm font-medium transition-colors border-b-2 -mb-px
                ${tab === t
                  ? 'border-blue-600 text-blue-700'
                  : 'border-transparent text-slate-500 hover:text-slate-700'
                }
              `}
            >
              {t === 'chat' ? 'Chat' : 'Evaluation'}
            </button>
          ))}
        </div>

        {/* tab content */}
        <div className="flex-1 overflow-hidden">
          {tab === 'chat' ? (
            <ChatInterface documents={documents} />
          ) : (
            <EvalDashboard />
          )}
        </div>
      </main>
    </div>
  )
}
