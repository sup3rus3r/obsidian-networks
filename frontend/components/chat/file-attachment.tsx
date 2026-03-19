'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { X, Paperclip, FileText } from 'lucide-react'
import { Progress } from '@/components/ui/progress'
import { cn } from '@/lib/utils'

const ACCEPTED = {
  'text/csv'       : ['.csv'],
  'application/json': ['.json'],
  'image/*'        : ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff'],
  'video/*'        : ['.mp4', '.mov', '.avi', '.mkv', '.webm'],
}
const MAX_BYTES = 500 * 1024 * 1024 // 500 MB

// ── Attachment chip ───────────────────────────────────────────────────────────

interface AttachmentChipProps {
  file        : File
  uploading   : boolean
  progress    : number
  onRemove    : () => void
}

export function AttachmentChip({ file, uploading, progress, onRemove }: AttachmentChipProps) {
  const sizeLabel =
    file.size < 1024 * 1024
      ? `${(file.size / 1024).toFixed(1)} KB`
      : `${(file.size / (1024 * 1024)).toFixed(1)} MB`

  return (
    <div className="flex flex-col gap-1 rounded-lg border border-zinc-700 bg-zinc-800/80 px-3 py-2">
      <div className="flex items-center gap-2">
        <FileText className="h-3.5 w-3.5 shrink-0 text-[#39FF14]" />
        <span className="flex-1 truncate text-xs text-zinc-200">{file.name}</span>
        <span className="shrink-0 text-[10px] text-zinc-500">{sizeLabel}</span>
        {!uploading && (
          <button
            onClick={onRemove}
            className="cursor-pointer shrink-0 rounded p-0.5 text-zinc-500 hover:bg-zinc-700 hover:text-[#39FF14]"
          >
            <X className="h-3 w-3" />
          </button>
        )}
      </div>
      {uploading && (
        <Progress value={progress} className="h-1 bg-zinc-700 [&>div]:bg-violet-500" />
      )}
    </div>
  )
}

// ── Paperclip button ──────────────────────────────────────────────────────────

interface PaperclipButtonProps {
  onFile   : (f: File) => void
  disabled?: boolean
}

export function PaperclipButton({ onFile, disabled }: PaperclipButtonProps) {
  const onDrop = useCallback(
    (accepted: File[]) => { if (accepted[0]) onFile(accepted[0]) },
    [onFile]
  )

  const { getInputProps, open } = useDropzone({
    onDrop,
    accept  : ACCEPTED,
    maxSize : MAX_BYTES,
    multiple: false,
    noClick : true,
    noKeyboard: true,
  })

  return (
    <>
      <input {...getInputProps()} />
      <button
        type="button"
        onClick={open}
        disabled={disabled}
        title="Attach a dataset, image, or video"
        className={cn(
          'cursor-pointer flex h-[42px] w-9 shrink-0 items-center justify-center rounded-xl',
          'border border-zinc-700 bg-zinc-900 text-zinc-400',
          'hover:border-[#39FF14]/60 hover:text-[#39FF14]',
          'disabled:cursor-not-allowed disabled:opacity-40',
          'transition-colors'
        )}
      >
        <Paperclip className="h-4 w-4" />
      </button>
    </>
  )
}

// ── Page-level dropzone overlay ───────────────────────────────────────────────

interface DropOverlayProps {
  isDragging: boolean
}

export function DropOverlay({ isDragging }: DropOverlayProps) {
  if (!isDragging) return null
  return (
    <div className="pointer-events-none fixed inset-0 z-50 flex items-center justify-center bg-zinc-950/75 backdrop-blur-sm">
      <div className="flex flex-col items-center gap-3 rounded-2xl border-2 border-dashed border-[#39FF14]/60 bg-zinc-900/80 px-12 py-10">
        <Paperclip className="h-8 w-8 text-[#39FF14]" />
        <p className="text-sm font-medium text-zinc-100">Drop your file here</p>
        <p className="text-xs text-zinc-500">Max 500 MB · CSV, JSON, image, or video</p>
      </div>
    </div>
  )
}

// ── useDragDropZone — document-level global drop ──────────────────────────────
// Uses document listeners so the overlay fires anywhere in the browser window,
// not just over a specific element.

const ACCEPTED_EXTENSIONS = ['.csv', '.json', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff', '.mp4', '.mov', '.avi', '.mkv', '.webm']
const ACCEPTED_MIME       = ['text/csv', 'application/json']

function isAccepted(file: File): boolean {
  if (ACCEPTED_MIME.includes(file.type)) return true
  if (file.type.startsWith('image/') || file.type.startsWith('video/')) return true
  const ext = '.' + (file.name.split('.').pop() ?? '').toLowerCase()
  return ACCEPTED_EXTENSIONS.includes(ext)
}

export function useDragDropZone(onFile: (f: File) => void) {
  const [isDragActive, setIsDragActive] = useState(false)
  // Counter tracks nested dragenter/dragleave pairs across child elements
  const counter = useRef(0)

  const handleFile = useCallback((file: File) => {
    if (!isAccepted(file)) return
    if (file.size > MAX_BYTES) return
    onFile(file)
  }, [onFile])

  useEffect(() => {
    const onDragEnter = (e: DragEvent) => {
      if (e.dataTransfer?.types?.includes('Files')) {
        counter.current++
        setIsDragActive(true)
      }
    }
    const onDragLeave = (e: DragEvent) => {
      if (e.dataTransfer?.types?.includes('Files')) {
        counter.current = Math.max(0, counter.current - 1)
        if (counter.current === 0) setIsDragActive(false)
      }
    }
    const onDragOver = (e: DragEvent) => {
      if (e.dataTransfer?.types?.includes('Files')) e.preventDefault()
    }
    const onDrop = (e: DragEvent) => {
      e.preventDefault()
      counter.current = 0
      setIsDragActive(false)
      const file = e.dataTransfer?.files?.[0]
      if (file) handleFile(file)
    }

    document.addEventListener('dragenter', onDragEnter)
    document.addEventListener('dragleave', onDragLeave)
    document.addEventListener('dragover',  onDragOver)
    document.addEventListener('drop',      onDrop)
    return () => {
      document.removeEventListener('dragenter', onDragEnter)
      document.removeEventListener('dragleave', onDragLeave)
      document.removeEventListener('dragover',  onDragOver)
      document.removeEventListener('drop',      onDrop)
    }
  }, [handleFile])

  return { isDragActive }
}
