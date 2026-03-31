'use client'

import Image from 'next/image'
import Link from 'next/link'
import { useState, useCallback, useRef, useEffect } from 'react'
import { usePlatformSession } from '@/hooks/use-platform-session'
import { useEnvironment } from '@/hooks/use-environment'
import { ChatPanel, type ChatPanelHandle } from '@/components/chat/chat-panel'
import { ArtifactPanel } from '@/components/artifacts/artifact-panel'
import {
  ResizablePanelGroup,
  ResizablePanel,
  ResizableHandle,
} from '@/components/ui/resizable'
import { Skeleton } from '@/components/ui/skeleton'
import { Badge } from '@/components/ui/badge'
import { Clock, RotateCcw, FlaskConical } from 'lucide-react'

// ── Session banner ────────────────────────────────────────────────────────────

const WARN_SECS = 5 * 60 // 5 minutes

function SessionBanner({
  expiresAt,
  onExtend,
  onNewSession,
}: {
  expiresAt   : number | null
  onExtend    : () => Promise<void>
  onNewSession: () => void
}) {
  const [remaining, setRemaining] = useState<number | null>(null)
  const [extending, setExtending] = useState(false)

  useEffect(() => {
    if (!expiresAt) return
    const tick = () => setRemaining(Math.max(0, expiresAt - Math.floor(Date.now() / 1000)))
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [expiresAt])

  if (remaining === null) return null

  // Expired
  if (remaining === 0) {
    return (
      <button
        onClick={onNewSession}
        className="cursor-pointer flex items-center gap-1.5 text-[11px] text-red-500 hover:text-red-400 transition-colors"
      >
        <Clock className="h-3 w-3" />
        <span>Session expired — start new session</span>
      </button>
    )
  }

  const hh   = Math.floor(remaining / 3600)
  const mm   = Math.floor((remaining % 3600) / 60)
  const ss   = remaining % 60
  const label = hh > 0
    ? `${hh}:${String(mm).padStart(2, '0')}:${String(ss).padStart(2, '0')}`
    : `${String(mm).padStart(2, '0')}:${String(ss).padStart(2, '0')}`

  const isWarning = remaining <= WARN_SECS

  const handleExtend = async () => {
    setExtending(true)
    try { await onExtend() } finally { setExtending(false) }
  }

  return (
    <div className="flex items-center gap-2">
      <div className={`flex items-center gap-1.5 text-[11px] transition-colors ${
        isWarning
          ? 'text-amber-400 animate-pulse'
          : 'text-zinc-600'
      }`}>
        <Clock className="h-3 w-3 shrink-0" />
        <span>Session expires {label}</span>
      </div>
      {isWarning && (
        <button
          onClick={handleExtend}
          disabled={extending}
          className="cursor-pointer rounded border border-amber-500/40 px-1.5 py-0.5 text-[10px] text-amber-400 hover:border-amber-400/70 hover:text-amber-300 disabled:opacity-50 transition-colors"
        >
          {extending ? 'Extending…' : 'Extend'}
        </button>
      )}
    </div>
  )
}

// ── Header ────────────────────────────────────────────────────────────────────

function Header({
  environment,
  expiresAt,
  onExtend,
  onNewSession,
}: {
  environment : { os: string; hardware: string; label: string }
  expiresAt   : number | null
  onExtend    : () => Promise<void>
  onNewSession: () => void
}) {
  const osLabel: Record<string, string> = {
    windows: 'Windows',
    mac    : 'macOS',
    linux  : 'Linux',
    unknown: 'Unknown',
  }

  return (
    <header className="flex h-11 shrink-0 items-center justify-between border-b border-zinc-800 bg-zinc-950 px-5">
      <Image
        src="/logo.svg"
        alt="Obsidian Networks"
        width={200}
        height={36}
        priority
        className="h-9 w-auto"
      />

      <div className="flex items-center gap-3">
        <SessionBanner expiresAt={expiresAt} onExtend={onExtend} onNewSession={onNewSession} />

        <div className="flex items-center gap-1.5">
          <Badge variant="outline" className="h-5 border-zinc-700 px-1.5 text-[10px] text-zinc-400">
            {osLabel[environment.os] ?? environment.os}
          </Badge>
          <Badge variant="outline" className="h-5 border-zinc-700 px-1.5 text-[10px] text-zinc-400">
            {environment.hardware === 'nvidia_gpu' ? 'NVIDIA GPU'
              : environment.hardware === 'apple_silicon' ? 'Apple Silicon'
              : environment.hardware === 'google_colab' ? 'Colab'
              : 'CPU'}
          </Badge>
        </div>

        <Link
          href="/research"
          className="flex items-center gap-1 rounded-md border border-violet-500/30 px-2 py-1 text-[11px] text-violet-400/70 transition-colors hover:border-violet-400/60 hover:text-violet-400"
          title="Research Labs — discover novel architectures with 8 AI agents"
        >
          <FlaskConical className="h-3 w-3" />
          Research Labs
          <span className="rounded bg-violet-500/20 px-1 py-px text-[9px] font-semibold uppercase tracking-wider text-violet-400/80">Experimental</span>
        </Link>

        <button
          onClick={onNewSession}
          title="Start a new session — clears uploaded data and chat"
          className="cursor-pointer flex items-center gap-1 rounded-md border border-[#39FF14]/30 px-2 py-1 text-[11px] text-[#39FF14]/70 transition-colors hover:border-[#39FF14]/60 hover:text-[#39FF14]"
        >
          <RotateCcw className="h-3 w-3" />
          New session
        </button>
      </div>
    </header>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function Home() {
  const { sessionId, loading, error, isValid, expiresAt, refresh, newSession, extendSession } = usePlatformSession()
  const { environment } = useEnvironment()

  // Incrementing this key forces ChatPanel + ArtifactPanel to remount (clears all local state)
  const [resetKey, setResetKey] = useState(0)
  const chatRef = useRef<ChatPanelHandle>(null)

  const handleCompileError = useCallback((error: string) => {
    chatRef.current?.sendError(error)
  }, [])

  const handleNewSession = useCallback(async () => {
    await newSession()
    setResetKey(k => k + 1)
  }, [newSession])

  // ── Loading state ─────────────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="flex h-screen flex-col bg-zinc-950">
        <div className="flex h-11 items-center border-b border-zinc-800 px-5">
          <Skeleton className="h-4 w-36 bg-zinc-800" />
        </div>
        <div className="flex flex-1 items-center justify-center">
          <p className="text-sm text-zinc-500">Starting session…</p>
        </div>
      </div>
    )
  }

  // ── Error / expired ───────────────────────────────────────────────────────
  if (error || !isValid) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4 bg-zinc-950">
        <p className="text-sm text-zinc-400">
          {error ?? 'Session expired.'}
        </p>
        <button
          onClick={refresh}
          className="cursor-pointer rounded-lg border border-zinc-700 px-4 py-2 text-sm text-zinc-300 hover:border-zinc-500"
        >
          Start new session
        </button>
      </div>
    )
  }

  // ── Main layout ───────────────────────────────────────────────────────────
  return (
    <div className="flex h-screen flex-col bg-zinc-950">

      <Header
        environment={environment}
        expiresAt={expiresAt}
        onExtend={extendSession}
        onNewSession={handleNewSession}
      />

      <ResizablePanelGroup orientation="horizontal" className="flex-1 overflow-hidden">

        {/* Left: Chat */}
        <ResizablePanel defaultSize={62} minSize={40}>
          <ChatPanel key={resetKey} sessionId={sessionId} ref={chatRef} />
        </ResizablePanel>

        <ResizableHandle withHandle className="bg-zinc-800 w-px" />

        {/* Right: Artifact panel */}
        <ResizablePanel defaultSize={38} minSize={28}>
          <ArtifactPanel key={resetKey} sessionId={sessionId} onCompileError={handleCompileError} />
        </ResizablePanel>

      </ResizablePanelGroup>

    </div>
  )
}
