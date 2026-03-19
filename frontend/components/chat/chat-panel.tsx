'use client'

import { useChat } from '@ai-sdk/react'
import { DefaultChatTransport, isTextUIPart, type UIMessage } from 'ai'
import { useEffect, useRef, useCallback, useState, useMemo, forwardRef, useImperativeHandle } from 'react'
import { Copy, Check, Paperclip, Trash2, ChevronDown, Brain } from 'lucide-react'
import { Skeleton } from '@/components/ui/skeleton'
import { Avatar, AvatarFallback } from '@/components/ui/avatar'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { MessageContent } from './message-content'
import { ToolStatusChip, ToolSourcesList, type AnyToolPart } from './tool-status'
import { AttachmentChip, DropOverlay, useDragDropZone } from './file-attachment'
import { DatasetCard, type DatasetPreview } from './dataset-card'
import { uploadDataset, getDatasetPreview } from '@/app/api/platform'
import { cn } from '@/lib/utils'
import {
  PromptInput,
  PromptInputTextarea,
  PromptInputHeader,
  PromptInputFooter,
  PromptInputTools,
  PromptInputButton,
  PromptInputSubmit,
  type PromptInputMessage,
} from '@/components/ai-elements/prompt-input'

function getMessageText(msg: UIMessage): string {
  return msg.parts
    .filter(isTextUIPart)
    .map(p => p.text)
    .join('')
}

/** Strip the injected schema block from user messages for display. */
function getDisplayText(msg: UIMessage): string {
  const full = getMessageText(msg)
  const marker = full.indexOf('\n\n[Dataset:')
  return marker !== -1 ? full.slice(0, marker).trim() : full
}

function formatSchema(filename: string, preview: DatasetPreview): string {
  const cols = preview.columns
    .map(c => `${c} (${preview.dtypes[c] ?? 'unknown'})`)
    .join(', ')
  return (
    `\n\n[Dataset: ${filename}]\n` +
    `Rows: ${preview.row_count} · Columns: ${preview.columns.length}\n` +
    `Schema: ${cols}`
  )
}

interface LocalCard {
  id         : string
  afterMsgId : string | 'top'
  filename   : string
  preview    : DatasetPreview
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)

  const copy = async () => {
    await navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          onClick={copy}
          className="cursor-pointer rounded p-1 text-zinc-600 transition-colors hover:text-[#39FF14] hover:bg-zinc-700"
        >
          {copied ? <Check className="h-3.5 w-3.5 text-[#39FF14]" /> : <Copy className="h-3.5 w-3.5" />}
        </button>
      </TooltipTrigger>
      <TooltipContent side="top">
        <p>{copied ? 'Copied!' : 'Copy message'}</p>
      </TooltipContent>
    </Tooltip>
  )
}

function ThinkingBlock({ text, isStreaming }: { text: string; isStreaming: boolean }) {
  const [open, setOpen] = useState(isStreaming)

  // Auto-open while streaming, auto-close when done
  useEffect(() => {
    if (isStreaming) setOpen(true)
    else setOpen(false)
  }, [isStreaming])

  return (
    <div className="mb-2 rounded-lg border border-zinc-700/50 bg-zinc-800/40 text-xs">
      <button
        onClick={() => setOpen(o => !o)}
        className="flex w-full items-center gap-2 px-3 py-2 text-zinc-400 hover:text-zinc-200 transition-colors"
      >
        <Brain className={cn('h-3.5 w-3.5 shrink-0', isStreaming && 'animate-pulse text-[#39FF14]')} />
        <span className="font-medium">
          {isStreaming ? 'Thinking…' : 'Thought process'}
        </span>
        <ChevronDown className={cn('ml-auto h-3.5 w-3.5 transition-transform', open && 'rotate-180')} />
      </button>
      {open && (
        <div className="border-t border-zinc-700/50 px-3 py-2 text-zinc-500 leading-relaxed whitespace-pre-wrap font-mono text-[11px] max-h-48 overflow-y-auto">
          {text}
        </div>
      )}
    </div>
  )
}

interface MessageBubbleProps {
  msg        : UIMessage
  isLast     : boolean
  isStreaming: boolean
}

function MessageBubble({ msg, isLast, isStreaming }: MessageBubbleProps) {
  const isUser = msg.role === 'user'

  if (isUser) {
    const dispText = getDisplayText(msg)
    // Silent messages (e.g. injected compile errors) are not shown in the chat UI
    if (dispText.startsWith('[__silent__]')) return null
    return (
      <div className="group flex items-start gap-3 px-4 py-3 flex-row-reverse">
        <Avatar className="h-7 w-7 shrink-0 mt-0.5">
          <AvatarFallback className="text-xs font-medium bg-zinc-700 text-zinc-200">U</AvatarFallback>
        </Avatar>
        <div className="flex flex-col gap-1 max-w-[85%] items-end">
          <div className="rounded-2xl rounded-tr-sm bg-zinc-700 px-4 py-2.5 text-sm text-zinc-100">
            {dispText}
          </div>
          <div className="flex justify-end">
            <CopyButton text={dispText} />
          </div>
        </div>
      </div>
    )
  }

  const textParts      = msg.parts.filter(isTextUIPart)
  const toolParts      = msg.parts.filter(p => p.type.startsWith('tool-')) as AnyToolPart[]
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const reasoningText  = msg.parts.filter(p => p.type === 'reasoning').map(p => (p as any).text ?? (p as any).reasoning ?? '').join('\n')
  const fullText       = textParts.map(p => p.text).join('').trimStart()

  return (
    <div className="group flex items-start gap-3 px-4 py-3">
      <Avatar className="h-7 w-7 shrink-0 mt-0.5">
        <AvatarFallback className="text-xs font-medium bg-[#39FF14]/20 text-[#39FF14]">AI</AvatarFallback>
      </Avatar>

      <div className="flex flex-col gap-1.5 max-w-[85%]">
        {/* Thinking block — shown for reasoning models */}
        {reasoningText && (
          <ThinkingBlock text={reasoningText} isStreaming={isStreaming && isLast && !fullText} />
        )}

        {/* Tool status chips */}
        {toolParts.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {toolParts.map(part => (
              <ToolStatusChip key={part.toolCallId} part={part} />
            ))}
          </div>
        )}

        {/* Text response */}
        {fullText && (
          <MessageContent
            content={fullText}
            isStreaming={isStreaming && isLast}
            className="text-zinc-200"
          />
        )}

        {/* Sources collapsible */}
        <ToolSourcesList parts={toolParts} />

        <CopyButton text={fullText} />
      </div>
    </div>
  )
}

function ThinkingSkeleton() {
  return (
    <div className="flex items-start gap-3 px-4 py-3">
      <Avatar className="h-7 w-7 shrink-0 mt-0.5">
        <AvatarFallback className="bg-[#39FF14]/20 text-[#39FF14] text-xs font-medium">
          AI
        </AvatarFallback>
      </Avatar>
      <div className="flex flex-col gap-2 pt-1">
        <Skeleton className="h-3.5 w-48 bg-zinc-700" />
        <Skeleton className="h-3.5 w-64 bg-zinc-700" />
        <Skeleton className="h-3.5 w-36 bg-zinc-700" />
      </div>
    </div>
  )
}

export interface ChatPanelHandle {
  sendError: (error: string) => void
}

interface ChatPanelProps {
  sessionId: string | null
}

export const ChatPanel = forwardRef<ChatPanelHandle, ChatPanelProps>(function ChatPanel({ sessionId }, ref) {
  const scrollRef  = useRef<HTMLDivElement>(null)
  const bottomRef  = useRef<HTMLDivElement>(null)
  const nearBottom = useRef(true)

  // File upload state
  const [pendingFiles,    setPendingFiles]    = useState<File[]>([])
  const [uploadProgress,  setUploadProgress]  = useState(0)
  const [isUploading,     setIsUploading]     = useState(false)
  const [uploadError,     setUploadError]     = useState<string | null>(null)
  const [supportsVision,  setSupportsVision]  = useState<boolean | null>(null)

  useEffect(() => {
    fetch('/api/provider').then(r => r.json()).then(d => setSupportsVision(d.supportsVision)).catch(() => setSupportsVision(null))
  }, [])

  // Local dataset cards inserted between messages — restored from sessionStorage on mount
  const [localCards, setLocalCards] = useState<LocalCard[]>(() => {
    if (typeof window === 'undefined' || !sessionId) return []
    try {
      const raw = sessionStorage.getItem(`ob_cards_${sessionId}`)
      return raw ? (JSON.parse(raw) as LocalCard[]) : []
    } catch {
      return []
    }
  })

  // AI SDK v6 — transport replaces the old `api` string shorthand
  const transport = useMemo(
    () => new DefaultChatTransport({ api: '/api/chat', body: { sessionId } }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [sessionId]
  )

  // Restore messages from sessionStorage so a browser refresh doesn't wipe the chat
  const storageKey = sessionId ? `ob_chat_${sessionId}` : null
  const cardsKey   = sessionId ? `ob_cards_${sessionId}` : null

  const { messages, sendMessage, stop, status, setMessages } = useChat({ transport })

  // Rehydrate messages from sessionStorage on mount (restores chat after browser refresh)
  const hydratedRef = useRef(false)
  useEffect(() => {
    if (hydratedRef.current || !storageKey) return
    hydratedRef.current = true
    try {
      const raw = sessionStorage.getItem(storageKey)
      if (raw) setMessages(JSON.parse(raw) as UIMessage[])
    } catch { /* corrupted storage — start fresh */ }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [storageKey])

  // Persist messages whenever they change
  useEffect(() => {
    if (!storageKey || !hydratedRef.current) return
    try {
      sessionStorage.setItem(storageKey, JSON.stringify(messages))
    } catch { /* quota exceeded — best effort */ }
  }, [messages, storageKey])

  // Persist local dataset cards
  useEffect(() => {
    if (!cardsKey) return
    try {
      sessionStorage.setItem(cardsKey, JSON.stringify(localCards))
    } catch { /* quota exceeded — best effort */ }
  }, [localCards, cardsKey])

  useImperativeHandle(ref, () => ({
    sendError: (error: string) => {
      sendMessage({
        text: `[__silent__]The training script failed to compile with the following error:\n\n${error}\n\nPlease fix the script and call create_notebook again with the corrected version.`,
      })
    },
  }), [sendMessage])

  const isLoading = status === 'streaming' || status === 'submitted'

  const handleClearChat = useCallback(() => {
    stop()
    setMessages([])
    setLocalCards([])
    if (storageKey) sessionStorage.removeItem(storageKey)
    if (cardsKey)   sessionStorage.removeItem(cardsKey)
  }, [stop, setMessages, storageKey, cardsKey])

  const onScroll = useCallback(() => {
    const el = scrollRef.current
    if (!el) return
    nearBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80
  }, [])

  useEffect(() => {
    if (nearBottom.current) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages, localCards])

  const handleFileSelect = useCallback((file: File) => {
    const isImage = file.type.startsWith('image/')
    setPendingFiles(prev => isImage ? [...prev, file] : [file])
    setUploadProgress(0)
    setUploadError(null)
  }, [])

  const handleSubmit = useCallback(async ({ text }: PromptInputMessage) => {
    const trimmed = text.trim()
    if (!trimmed && pendingFiles.length === 0) return
    if (isLoading || isUploading) return

    let schemaBlock = ''
    const filesToUpload = pendingFiles
    const isVisionFile = filesToUpload.length > 0 &&
      (filesToUpload[0].type.startsWith('image/') || filesToUpload[0].type.startsWith('video/'))

    if (filesToUpload.length > 0 && sessionId) {
      setIsUploading(true)
      setUploadError(null)

      for (const file of filesToUpload) {
        const uploaded = await uploadDataset(sessionId, file, setUploadProgress)
        if (!uploaded) {
          setUploadError('Upload failed. Please try again.')
          setIsUploading(false)
          return
        }
      }

      if (isVisionFile) {
        const mediaType = filesToUpload[0].type.startsWith('video/') ? 'video' : 'image'
        const names = filesToUpload.map(f => f.name).join(', ')
        schemaBlock = `\n\n[Attached ${filesToUpload.length} ${mediaType} file(s): ${names}. Focus on vision-based architectures and computer vision approaches.]`
      } else {
        const preview = await getDatasetPreview(sessionId) as DatasetPreview | null
        if (preview) {
          const cardId = `card-${Date.now()}`
          const afterId = messages.length > 0 ? messages[messages.length - 1].id : 'top'
          setLocalCards(prev => [
            ...prev,
            { id: cardId, afterMsgId: afterId, filename: filesToUpload[0].name, preview },
          ])
          schemaBlock = formatSchema('dataset.csv', preview)
        }
      }

      setIsUploading(false)
      setPendingFiles([])

      window.dispatchEvent(new Event('dataset-uploaded'))
    }

    const defaultMessage = isVisionFile
      ? `I've uploaded ${filesToUpload.length} ${filesToUpload[0].type.startsWith('video/') ? 'video' : 'image'} file(s). Please suggest the best vision architecture for this task.`
      : `I've uploaded a dataset. Please analyse it and suggest the best ML architecture.`

    const messageText = trimmed || defaultMessage
    sendMessage({ text: messageText + schemaBlock })
  }, [pendingFiles, sessionId, isLoading, isUploading, messages, sendMessage])

  const { isDragActive } = useDragDropZone(handleFileSelect)

  const showSkeleton = isLoading && messages[messages.length - 1]?.role === 'user'

  // Collect cards keyed by the message ID they follow
  const cardsByAfter = useMemo(() => {
    const map: Record<string, LocalCard[]> = {}
    for (const card of localCards) {
      if (!map[card.afterMsgId]) map[card.afterMsgId] = []
      map[card.afterMsgId].push(card)
    }
    return map
  }, [localCards])

  return (
    <div className="flex h-full flex-col bg-zinc-950">
      <DropOverlay isDragging={isDragActive} />

      {/* Empty state */}
      {messages.length === 0 && localCards.length === 0 && (
        <div className="flex flex-1 flex-col items-center justify-center gap-4 px-8 text-center">
          <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-6 max-w-sm">
            <p className="text-sm text-zinc-400 leading-relaxed">
              Describe your ML problem — or attach a dataset to ground the solution in your data.
              CSV, JSON, images, and video are all supported. I&apos;ll research the right architecture,
              build a complete training notebook, and compile a <code className="text-[#39FF14]/80">.keras</code> model.
            </p>
          </div>
        </div>
      )}

      {/* Message list */}
      {(messages.length > 0 || localCards.length > 0) && (
        <div ref={scrollRef} className="flex-1 overflow-y-auto" onScroll={onScroll}>
          <div className="flex flex-col py-2">

            {/* Cards that precede all messages */}
            {(cardsByAfter['top'] ?? []).map(card => (
              <div key={card.id} className="px-4 py-2">
                <DatasetCard filename={card.filename} preview={card.preview} />
              </div>
            ))}

            {messages.map((msg, i) => (
              <div key={msg.id}>
                <MessageBubble
                  msg={msg}
                  isLast={i === messages.length - 1}
                  isStreaming={isLoading}
                />
                {/* Dataset cards injected after this message */}
                {(cardsByAfter[msg.id] ?? []).map(card => (
                  <div key={card.id} className="px-4 py-2">
                    <DatasetCard filename={card.filename} preview={card.preview} />
                  </div>
                ))}
              </div>
            ))}

            {showSkeleton && <ThinkingSkeleton />}
            <div ref={bottomRef} />
          </div>
        </div>
      )}

      {/* Input area */}
      <div className="border-t border-zinc-800 bg-zinc-950 px-4 py-3">

        {uploadError && (
          <p className="mb-2 text-xs text-red-400">{uploadError}</p>
        )}

        {/* Vision capability warning */}
        {pendingFiles.length > 0 &&
          (pendingFiles[0].type.startsWith('image/') || pendingFiles[0].type.startsWith('video/')) &&
          supportsVision === false && (
          <div className="mb-2 flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/8 px-3 py-2">
            <span className="mt-0.5 text-amber-400">⚠</span>
            <p className="text-xs text-amber-300/80">
              Your current model may not support vision inputs. For best results, describe what&apos;s in the file and what you want to build instead.
            </p>
          </div>
        )}

        <PromptInput
          onSubmit={handleSubmit}
          className="rounded-xl border border-zinc-700 bg-zinc-900"
        >
          {/* Attachment chips shown above textarea when files are selected */}
          {pendingFiles.length > 0 && (
            <PromptInputHeader className="flex flex-wrap gap-2 px-3 pt-2">
              {pendingFiles.map((file, i) => (
                <AttachmentChip
                  key={`${file.name}-${i}`}
                  file={file}
                  uploading={isUploading}
                  progress={uploadProgress}
                  onRemove={() => {
                    setPendingFiles(prev => prev.filter((_, idx) => idx !== i))
                    setUploadError(null)
                  }}
                />
              ))}
            </PromptInputHeader>
          )}

          <PromptInputTextarea
            placeholder="Upload a dataset or describe your ML task…"
            className="bg-transparent px-3 py-2.5 text-sm text-zinc-100 placeholder:text-zinc-500 focus:outline-none"
          />

          <PromptInputFooter className="px-2 pb-2">
            <PromptInputTools>
              {/* Paperclip button — opens file picker */}
              <Tooltip>
                <TooltipTrigger asChild>
                  <PromptInputButton
                    tooltip="Attach CSV or JSON dataset"
                    disabled={isLoading || isUploading || (pendingFiles.length > 0 && !pendingFiles[0].type.startsWith('image/'))}
                    className="cursor-pointer text-zinc-400 hover:text-[#39FF14]"
                    onClick={() => {
                      // Trigger a hidden file input via a temporary input element
                      const input = document.createElement('input')
                      input.type = 'file'
                      input.accept = '.csv,.json,text/csv,application/json,image/*,video/*'
                      input.multiple = true
                      input.onchange = () => {
                        const files = Array.from(input.files ?? [])
                        files.forEach(f => handleFileSelect(f))
                      }
                      input.click()
                    }}
                  >
                    <Paperclip className="h-4 w-4" />
                  </PromptInputButton>
                </TooltipTrigger>
                <TooltipContent side="top">
                  <p>Attach a file — CSV, JSON, image, or video</p>
                </TooltipContent>
              </Tooltip>
              {/* Clear chat button — only shown when there are messages */}
              {messages.length > 0 && (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <PromptInputButton
                      tooltip="Clear chat"
                      disabled={isLoading}
                      onClick={handleClearChat}
                      className="cursor-pointer text-zinc-600 hover:text-red-400"
                    >
                      <Trash2 className="h-4 w-4" />
                    </PromptInputButton>
                  </TooltipTrigger>
                  <TooltipContent side="top"><p>Clear chat</p></TooltipContent>
                </Tooltip>
              )}
            </PromptInputTools>

            <PromptInputSubmit
              status={isUploading ? 'submitted' : status}
              onStop={stop}
              disabled={isUploading}
              className="bg-[#39FF14]/90 text-black hover:bg-[#39FF14] disabled:opacity-40"
            />
          </PromptInputFooter>
        </PromptInput>

        <p className="mt-1.5 text-center text-[11px] text-zinc-600">
          Shift+Enter for new line · Enter to send · Drag &amp; drop files anywhere
        </p>
      </div>
    </div>
  )
})
