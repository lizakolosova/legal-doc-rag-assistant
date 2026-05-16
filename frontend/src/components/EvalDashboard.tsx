import { useCallback, useEffect, useState } from 'react'
import { getEvalRun, listEvalRuns, runEvaluation } from '../hooks/useApi'
import type { EvalResult, EvalRunDetailResponse, EvalRunSummaryResponse } from '../hooks/useApi'

const METRIC_KEYS = [
  'context_precision',
  'context_recall',
] as const

type MetricKey = (typeof METRIC_KEYS)[number]

const METRIC_LABELS: Record<MetricKey, string> = {
  context_precision: 'Context Precision',
  context_recall: 'Context Recall',
}

function metricColor(val: number | null | undefined): string {
  if (val == null) return 'text-slate-400'
  if (val >= 0.7) return 'text-green-700'
  if (val >= 0.4) return 'text-amber-700'
  return 'text-red-700'
}

function metricBg(val: number | null | undefined): string {
  if (val == null) return 'bg-slate-100'
  if (val >= 0.7) return 'bg-green-50'
  if (val >= 0.4) return 'bg-amber-50'
  return 'bg-red-50'
}

function pct(val: number | null | undefined): string {
  if (val == null) return '—'
  return `${Math.round(val * 100)}%`
}

function SummaryTable({ summary }: { summary: Record<string, number> }) {
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="border-b border-slate-200">
          {METRIC_KEYS.map((k) => (
            <th key={k} className="pb-2 pr-4 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
              {METRIC_LABELS[k]}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        <tr>
          {METRIC_KEYS.map((k) => {
            const val = summary[k] as number | undefined
            return (
              <td key={k} className={`pr-4 pt-3 font-bold ${metricColor(val)}`}>
                {pct(val)}
              </td>
            )
          })}
        </tr>
      </tbody>
    </table>
  )
}

function ResultRow({ result }: { result: EvalResult }) {
  const [open, setOpen] = useState(false)
  return (
    <>
      <tr
        className="cursor-pointer border-b border-slate-100 hover:bg-slate-50"
        onClick={() => setOpen((p) => !p)}
      >
        <td className="py-2 pr-4 text-xs text-slate-700 max-w-xs truncate">{result.question}</td>
        {METRIC_KEYS.map((k) => {
          const val = result[k] as number | null
          return (
            <td key={k} className={`py-2 pr-4 text-xs font-semibold ${metricColor(val)} ${metricBg(val)} rounded px-1`}>
              {pct(val)}
            </td>
          )
        })}
        <td className="py-2 text-xs text-slate-400">{Math.round(result.latency_ms)}ms</td>
        <td className="py-2 pl-2 text-slate-400">
          <svg className={`h-3.5 w-3.5 transition-transform ${open ? 'rotate-180' : ''}`}
            fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </td>
      </tr>
      {open && (
        <tr className="bg-slate-50">
          <td colSpan={5} className="px-3 pb-3 pt-1">
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div>
                <p className="font-semibold text-slate-600 mb-1">Expected</p>
                <p className="text-slate-700 leading-relaxed">{result.expected_answer}</p>
              </div>
              <div>
                <p className="font-semibold text-slate-600 mb-1">Generated</p>
                <p className="text-slate-700 leading-relaxed">{result.generated_answer}</p>
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  )
}

function RunRow({ run, onSelect }: { run: EvalRunSummaryResponse; onSelect: () => void }) {
  return (
    <tr className="cursor-pointer border-b border-slate-100 hover:bg-slate-50" onClick={onSelect}>
      <td className="py-2 pr-4 text-xs text-slate-600">
        {new Date(run.created_at).toLocaleString()}
      </td>
      {METRIC_KEYS.map((k) => {
        const val = run.summary[k]
        return (
          <td key={k} className={`py-2 pr-4 text-xs font-semibold ${metricColor(val)}`}>
            {pct(val)}
          </td>
        )
      })}
      <td className="py-2 text-xs text-blue-600 hover:underline">Details →</td>
    </tr>
  )
}

export default function EvalDashboard() {
  const [runs, setRuns] = useState<EvalRunSummaryResponse[]>([])
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedRun, setSelectedRun] = useState<EvalRunDetailResponse | null>(null)
  const [loadingDetail, setLoadingDetail] = useState(false)

  const fetchRuns = useCallback(async () => {
    try {
      const data = await listEvalRuns()
      setRuns(data.runs)
    } catch {
      // non-blocking
    }
  }, [])

  useEffect(() => { void fetchRuns() }, [fetchRuns])

  const handleRun = async () => {
    setRunning(true)
    setError(null)
    try {
      await runEvaluation()
      await fetchRuns()
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        'Evaluation failed.'
      setError(msg)
    } finally {
      setRunning(false)
    }
  }

  const handleSelectRun = async (runId: string) => {
    if (selectedRun?.run_id === runId) {
      setSelectedRun(null)
      return
    }
    setLoadingDetail(true)
    try {
      const detail = await getEvalRun(runId)
      setSelectedRun(detail)
    } finally {
      setLoadingDetail(false)
    }
  }

  return (
    <div className="h-full overflow-y-auto px-6 py-5">
      <div className="flex items-center justify-between mb-5">
        <div>
          <h2 className="text-lg font-semibold text-slate-800">Evaluation Dashboard</h2>
          <p className="text-xs text-slate-500 mt-0.5">Run RAGAS evaluation against the golden dataset</p>
        </div>
        <button
          onClick={() => void handleRun()}
          disabled={running}
          className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium
            text-white shadow-sm transition-colors hover:bg-blue-700
            disabled:bg-blue-400 disabled:cursor-not-allowed"
        >
          {running ? (
            <>
              <svg className="h-4 w-4 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Running…
            </>
          ) : (
            'Run Evaluation'
          )}
        </button>
      </div>

      {error && (
        <div className="mb-4 rounded-lg bg-red-50 border border-red-200 px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {runs.length === 0 ? (
        <div className="rounded-xl border border-dashed border-slate-300 py-12 text-center text-slate-400">
          <p className="text-sm">No evaluation runs yet. Click "Run Evaluation" to start.</p>
        </div>
      ) : (
        <div className="rounded-xl border border-slate-200 bg-white overflow-hidden shadow-sm">
          <table className="w-full">
            <thead className="bg-slate-50 border-b border-slate-200">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">Timestamp</th>
                {METRIC_KEYS.map((k) => (
                  <th key={k} className="py-3 pr-4 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
                    {METRIC_LABELS[k]}
                  </th>
                ))}
                <th />
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <RunRow key={run.run_id} run={run} onSelect={() => void handleSelectRun(run.run_id)} />
              ))}
            </tbody>
          </table>
        </div>
      )}

      {loadingDetail && (
        <div className="mt-6 text-center text-sm text-slate-400">Loading details…</div>
      )}

      {selectedRun && !loadingDetail && (
        <div className="mt-6">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-700">
              Run details — {new Date(selectedRun.created_at).toLocaleString()}
            </h3>
            <button
              onClick={() => setSelectedRun(null)}
              className="text-xs text-slate-400 hover:text-slate-600"
            >
              Close
            </button>
          </div>

          <div className="mb-5 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
            <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-slate-500">Summary</p>
            <SummaryTable summary={selectedRun.summary} />
          </div>

          {selectedRun.results.length > 0 && (
            <div className="rounded-xl border border-slate-200 bg-white overflow-hidden shadow-sm">
              <table className="w-full">
                <thead className="bg-slate-50 border-b border-slate-200">
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">Question</th>
                    {METRIC_KEYS.map((k) => (
                      <th key={k} className="py-3 pr-4 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
                        {METRIC_LABELS[k]}
                      </th>
                    ))}
                    <th className="py-3 pr-2 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">Latency</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {selectedRun.results.map((r) => (
                    <ResultRow key={r.question_id} result={r} />
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
