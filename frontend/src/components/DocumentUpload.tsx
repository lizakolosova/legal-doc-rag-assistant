import { useCallback, useRef, useState } from 'react'
import { uploadDocument } from '../hooks/useApi'

interface Props {
  onUploadSuccess: () => void
}

const ACCEPTED = ['.pdf', '.docx']
const ACCEPTED_MIME = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']

function isAccepted(file: File): boolean {
  const ext = '.' + (file.name.split('.').pop() ?? '').toLowerCase()
  return ACCEPTED.includes(ext) || ACCEPTED_MIME.includes(file.type)
}

export default function DocumentUpload({ onUploadSuccess }: Props) {
  const [dragging, setDragging] = useState(false)
  const [progress, setProgress] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFile = useCallback(async (file: File) => {
    if (!isAccepted(file)) {
      setError('Only .pdf and .docx files are accepted.')
      return
    }
    setError(null)
    setProgress(0)
    try {
      await uploadDocument(file, setProgress)
      setProgress(null)
      onUploadSuccess()
    } catch (err: unknown) {
      setProgress(null)
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        'Upload failed. Please try again.'
      setError(msg)
    }
  }, [onUploadSuccess])

  const onDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) void handleFile(file)
  }, [handleFile])

  const onInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) void handleFile(file)
    e.target.value = ''
  }, [handleFile])

  return (
    <div className="px-3 pb-3">
      <div
        role="button"
        tabIndex={0}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`
          flex flex-col items-center justify-center gap-1 rounded-lg border-2 border-dashed
          px-4 py-5 text-center cursor-pointer transition-colors select-none
          ${dragging
            ? 'border-blue-500 bg-blue-50 text-blue-700'
            : 'border-slate-300 bg-slate-50 text-slate-500 hover:border-slate-400 hover:bg-slate-100'
          }
        `}
      >
        <svg className="h-6 w-6 opacity-60" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
            d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
        </svg>
        <span className="text-sm font-medium">Upload document</span>
        <span className="text-xs opacity-70">PDF or DOCX · drag & drop or click</span>
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.docx"
          className="sr-only"
          onChange={onInputChange}
        />
      </div>

      {progress !== null && (
        <div className="mt-2">
          <div className="h-1.5 w-full rounded-full bg-slate-200">
            <div
              className="h-1.5 rounded-full bg-blue-500 transition-all duration-150"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="mt-1 text-right text-xs text-slate-500">{progress}%</p>
        </div>
      )}

      {error && (
        <p className="mt-2 rounded-md bg-red-50 px-3 py-2 text-xs text-red-700">{error}</p>
      )}
    </div>
  )
}
