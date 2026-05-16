import { useState } from 'react'
import type { Citation } from '../hooks/useApi'

interface Props {
  answer: string
  citations: Citation[]
}

function renderWithFootnotes(
  text: string,
  onClickRef: (idx: number) => void,
): React.ReactNode[] {
  const parts = text.split(/(\[\d+\])/g)
  return parts.map((part, i) => {
    const match = /^\[(\d+)\]$/.exec(part)
    if (match) {
      const n = parseInt(match[1], 10)
      return (
        <button
          key={i}
          onClick={() => onClickRef(n)}
          className="mx-0.5 inline-flex h-4 w-4 items-center justify-center rounded-full bg-blue-100
            text-[10px] font-bold text-blue-700 hover:bg-blue-200 align-super leading-none"
        >
          {n}
        </button>
      )
    }
    return <span key={i}>{part}</span>
  })
}

export default function AnswerCard({ answer, citations }: Props) {
  const [expanded, setExpanded] = useState<Set<number>>(new Set())

  const toggleCitation = (idx: number) => {
    setExpanded((prev) => {
      const next = new Set(prev)
      if (next.has(idx)) next.delete(idx)
      else next.add(idx)
      return next
    })
  }

  const citationMap = new Map(citations.map((c) => [c.index, c]))

  // collect which citation numbers actually appear in the answer text
  const mentioned = [...new Set(
    [...answer.matchAll(/\[(\d+)\]/g)].map((m) => parseInt(m[1], 10))
  )].filter((n) => citationMap.has(n)).sort((a, b) => a - b)

  return (
    <div className="rounded-xl border border-slate-200 bg-white px-5 py-4 shadow-sm">
      <p className="text-sm leading-relaxed text-slate-800 whitespace-pre-wrap">
        {renderWithFootnotes(answer, toggleCitation)}
      </p>

      {mentioned.length > 0 && (
        <div className="mt-4 space-y-1 border-t border-slate-100 pt-3">
          {mentioned.map((n) => {
            const c = citationMap.get(n)!
            const isOpen = expanded.has(n)
            return (
              <div key={n} className="text-xs">
                <button
                  onClick={() => toggleCitation(n)}
                  className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-left
                    text-slate-600 hover:bg-slate-50"
                >
                  <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full
                    bg-blue-100 text-[10px] font-bold text-blue-700">
                    {n}
                  </span>
                  <span className="truncate font-medium">{c.source_file}</span>
                  <span className="text-slate-400">p. {c.page_number}</span>
                  {c.section_header && (
                    <span className="truncate text-slate-400 italic">— {c.section_header}</span>
                  )}
                  <svg
                    className={`ml-auto h-3.5 w-3.5 shrink-0 transition-transform ${isOpen ? 'rotate-180' : ''}`}
                    fill="none" viewBox="0 0 24 24" stroke="currentColor"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>
                {isOpen && (
                  <div className="mx-2 mb-1 rounded-md bg-slate-50 px-3 py-2 font-mono text-[11px]
                    leading-relaxed text-slate-600 whitespace-pre-wrap border border-slate-100">
                    {c.chunk_text}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
