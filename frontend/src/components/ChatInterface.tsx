import { useCallback, useEffect, useRef, useState } from 'react'
import { streamQuery } from '../hooks/useApi'
import type { Citation, DocumentResponse } from '../hooks/useApi'
import AnswerCard from './AnswerCard'

interface Message {
  id: string
  role: 'user' | 'assistant'
  text: string
  citations: Citation[]
  streaming: boolean
}

interface Props {
  documents: DocumentResponse[]
}

export default function ChatInterface({ documents }: Props) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [selectedIds, setSelectedIds] = useState<string[]>([])
  const [streaming, setStreaming] = useState(false)
  const [scopeAll, setScopeAll] = useState(true)
  const bottomRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const toggleDoc = (id: string) => {
    setSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    )
  }

  const handleSubmit = useCallback(async () => {
    const question = input.trim()
    if (!question || streaming) return

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      text: question,
      citations: [],
      streaming: false,
    }
    const assistantId = crypto.randomUUID()
    const assistantMsg: Message = {
      id: assistantId,
      role: 'assistant',
      text: '',
      citations: [],
      streaming: true,
    }

    setMessages((prev) => [...prev, userMsg, assistantMsg])
    setInput('')
    setStreaming(true)

    try {
      const docIds = scopeAll ? null : selectedIds.length > 0 ? selectedIds : null
      const citations = await streamQuery(
        { question, document_ids: docIds, top_k: 5, use_reranker: true },
        (delta) => {
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, text: m.text + delta } : m))
          )
        },
      )
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId ? { ...m, citations, streaming: false } : m
        )
      )
    } catch (err: unknown) {
      const msg =
        (err instanceof Error ? err.message : null) ?? 'Something went wrong. Please try again.'
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? { ...m, text: msg, streaming: false }
            : m
        )
      )
    } finally {
      setStreaming(false)
    }
  }, [input, streaming, scopeAll, selectedIds])

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      void handleSubmit()
    }
  }

  const readyDocs = documents.filter((d) => d.status === 'ready')

  return (
    <div className="flex h-full flex-col">
      {/* scope selector */}
      <div className="flex items-center gap-3 border-b border-slate-200 bg-slate-50 px-4 py-2.5">
        <span className="text-xs font-medium text-slate-500">Scope:</span>
        <label className="flex cursor-pointer items-center gap-1.5 text-xs text-slate-700">
          <input
            type="radio"
            name="scope"
            checked={scopeAll}
            onChange={() => setScopeAll(true)}
            className="accent-blue-600"
          />
          All documents
        </label>
        <label className="flex cursor-pointer items-center gap-1.5 text-xs text-slate-700">
          <input
            type="radio"
            name="scope"
            checked={!scopeAll}
            onChange={() => setScopeAll(false)}
            className="accent-blue-600"
          />
          Selected
        </label>
        {!scopeAll && readyDocs.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {readyDocs.map((doc) => (
              <label key={doc.document_id}
                className={`flex cursor-pointer items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium border transition-colors
                  ${selectedIds.includes(doc.document_id)
                    ? 'bg-blue-100 border-blue-300 text-blue-800'
                    : 'bg-white border-slate-200 text-slate-600 hover:border-slate-300'}`}
              >
                <input
                  type="checkbox"
                  className="sr-only"
                  checked={selectedIds.includes(doc.document_id)}
                  onChange={() => toggleDoc(doc.document_id)}
                />
                {doc.filename}
              </label>
            ))}
          </div>
        )}
        {!scopeAll && readyDocs.length === 0 && (
          <span className="text-xs text-slate-400 italic">No ready documents</span>
        )}
      </div>

      {/* message history */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {messages.length === 0 && (
          <div className="flex h-full flex-col items-center justify-center text-center text-slate-400">
            <svg className="mb-3 h-10 w-10 opacity-30" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1}
                d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
            </svg>
            <p className="text-sm font-medium">Ask a question about your documents</p>
            <p className="mt-1 text-xs">Upload PDFs or DOCX files in the sidebar, then ask anything.</p>
          </div>
        )}
        {messages.map((msg) => (
          <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            {msg.role === 'user' ? (
              <div className="max-w-lg rounded-2xl rounded-br-sm bg-blue-600 px-4 py-2.5 text-sm text-white shadow-sm">
                {msg.text}
              </div>
            ) : (
              <div className="w-full max-w-3xl">
                {msg.streaming && !msg.text ? (
                  <div className="flex items-center gap-2 text-slate-400 text-sm">
                    <span className="inline-flex gap-1">
                      <span className="animate-bounce delay-0 h-1.5 w-1.5 rounded-full bg-slate-400" style={{ animationDelay: '0ms' }} />
                      <span className="animate-bounce h-1.5 w-1.5 rounded-full bg-slate-400" style={{ animationDelay: '150ms' }} />
                      <span className="animate-bounce h-1.5 w-1.5 rounded-full bg-slate-400" style={{ animationDelay: '300ms' }} />
                    </span>
                    Generating…
                  </div>
                ) : (
                  <AnswerCard answer={msg.text} citations={msg.citations} />
                )}
              </div>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* input */}
      <div className="border-t border-slate-200 bg-white p-3">
        <div className="flex items-end gap-2 rounded-xl border border-slate-300 bg-white px-3 py-2
          focus-within:border-blue-400 focus-within:ring-1 focus-within:ring-blue-300 transition-shadow">
          <textarea
            ref={textareaRef}
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Ask a question… (Enter to send, Shift+Enter for newline)"
            disabled={streaming}
            className="flex-1 resize-none bg-transparent text-sm text-slate-800 placeholder-slate-400
              outline-none disabled:opacity-50 max-h-40 overflow-y-auto"
            style={{ lineHeight: '1.5' }}
          />
          <button
            onClick={() => void handleSubmit()}
            disabled={streaming || !input.trim()}
            className="mb-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg
              bg-blue-600 text-white transition-colors hover:bg-blue-700
              disabled:bg-slate-200 disabled:text-slate-400 disabled:cursor-not-allowed"
          >
            <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 12h14M12 5l7 7-7 7" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  )
}
