'use client'

import Image from 'next/image'
import Link from 'next/link'
import { useState, useCallback } from 'react'
import { type ResearchSession, cancelResearchSession } from '@/app/api/platform'
import { ResearchLaunchForm } from '@/components/research/research-launch-form'
import { ResearchProgressFeed } from '@/components/research/research-progress-feed'
import { CandidateList } from '@/components/research/candidate-list'
import {
  ResizablePanelGroup,
  ResizablePanel,
  ResizableHandle,
} from '@/components/ui/resizable'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import {
  FlaskConical, ArrowLeft, RotateCcw,
  XCircle, CheckCircle2, Loader2,
} from 'lucide-react'

// ── Status badge ──────────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: string }) {
  if (status === 'completed') return (
    <div className="flex items-center gap-1.5 text-[11px] text-[#39FF14]">
      <CheckCircle2 className="h-3.5 w-3.5" /> Complete
    </div>
  )
  if (status === 'error') return (
    <div className="flex items-center gap-1.5 text-[11px] text-red-400">
      <XCircle className="h-3.5 w-3.5" /> Error
    </div>
  )
  if (status === 'cancelled') return (
    <div className="flex items-center gap-1.5 text-[11px] text-zinc-500">
      <XCircle className="h-3.5 w-3.5" /> Cancelled
    </div>
  )
  return (
    <div className="flex items-center gap-1.5 text-[11px] text-blue-400">
      <Loader2 className="h-3.5 w-3.5 animate-spin" /> Running
    </div>
  )
}

// ── Header ────────────────────────────────────────────────────────────────────

function Header({
  session,
  onCancel,
  onReset,
}: {
  session  : ResearchSession | null
  onCancel : () => void
  onReset  : () => void
}) {
  return (
    <header className="flex h-11 shrink-0 items-center justify-between border-b border-zinc-800 bg-zinc-950 px-5">
      <div className="flex items-center gap-4">
        <Image src="/logo.svg" alt="Obsidian Networks" width={160} height={32} priority className="h-8 w-auto" />
        <Separator orientation="vertical" className="h-5 bg-zinc-800" />
        <div className="flex items-center gap-1.5 text-xs text-zinc-400">
          <FlaskConical className="h-3.5 w-3.5 text-[#39FF14]" />
          Research Mode
        </div>
      </div>

      <div className="flex items-center gap-3">
        {session && <StatusBadge status={session.status} />}

        {session && session.status === 'running' && (
          <button
            onClick={onCancel}
            className="cursor-pointer flex items-center gap-1 rounded-md border border-red-900/40 px-2 py-1 text-[11px] text-red-400/70 transition-colors hover:border-red-700 hover:text-red-400"
          >
            <XCircle className="h-3 w-3" />
            Cancel
          </button>
        )}

        {session && session.status !== 'running' && (
          <button
            onClick={onReset}
            className="cursor-pointer flex items-center gap-1 rounded-md border border-[#39FF14]/30 px-2 py-1 text-[11px] text-[#39FF14]/70 transition-colors hover:border-[#39FF14]/60 hover:text-[#39FF14]"
          >
            <RotateCcw className="h-3 w-3" />
            New session
          </button>
        )}

        <Link
          href="/home"
          className="flex items-center gap-1 rounded-md border border-zinc-700 px-2 py-1 text-[11px] text-zinc-400 transition-colors hover:border-zinc-500 hover:text-zinc-200"
        >
          <ArrowLeft className="h-3 w-3" />
          Back
        </Link>
      </div>
    </header>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function ResearchPage() {
  const [session,    setSession]    = useState<ResearchSession | null>(null)
  const [sessionStatus, setStatus] = useState<string>('queued')
  const [error,      setError]      = useState<string | null>(null)

  const handleSessionStarted = useCallback((s: ResearchSession) => {
    setSession({ ...s, status: 'running' })
    setStatus('running')
    setError(null)
  }, [])

  const handleComplete = useCallback(() => {
    setStatus('completed')
    setSession(prev => prev ? { ...prev, status: 'completed' } : prev)
  }, [])

  const handleError = useCallback((msg: string) => {
    setError(msg)
    setStatus('error')
    setSession(prev => prev ? { ...prev, status: 'error' } : prev)
  }, [])

  const handleCancel = useCallback(async () => {
    if (!session) return
    await cancelResearchSession(session.research_session_id)
    setStatus('cancelled')
    setSession(prev => prev ? { ...prev, status: 'cancelled' } : prev)
  }, [session])

  const handleReset = useCallback(() => {
    setSession(null)
    setStatus('queued')
    setError(null)
  }, [])

  const effectiveSession = session ? { ...session, status: sessionStatus } : null

  // ── No session yet: show launch form ────────────────────────────────────────
  if (!session) {
    return (
      <div className="flex h-screen flex-col bg-zinc-950">
        <Header session={null} onCancel={() => {}} onReset={handleReset} />
        <div className="flex flex-1 overflow-hidden">
          <div className="m-auto w-full max-w-2xl px-6 py-10">
            <div className="mb-8 text-center">
              <div className="mb-3 flex justify-center">
                <div className="flex h-12 w-12 items-center justify-center rounded-xl border border-[#39FF14]/30 bg-[#39FF14]/10">
                  <FlaskConical className="h-6 w-6 text-[#39FF14]" />
                </div>
              </div>
              <h1 className="text-xl font-bold text-zinc-100">Autonomous Research Mode</h1>
              <p className="mt-1.5 text-sm text-zinc-500">
                Eight AI agents continuously discover, train, score, and recurse through novel neural architectures.
              </p>
            </div>
            <ResearchLaunchForm onSessionStarted={handleSessionStarted} />
          </div>
        </div>
      </div>
    )
  }

  // ── Active session: three-panel layout ──────────────────────────────────────
  return (
    <div className="flex h-screen flex-col bg-zinc-950">
      <Header
        session={effectiveSession}
        onCancel={handleCancel}
        onReset={handleReset}
      />

      <ResizablePanelGroup orientation="horizontal" className="flex-1 overflow-hidden">

        {/* Left: Live progress feed */}
        <ResizablePanel defaultSize={38} minSize={28}>
          <div className="flex h-full flex-col border-r border-zinc-800">
            <div className="flex items-center gap-2 px-4 py-2.5 border-b border-zinc-800 shrink-0">
              <FlaskConical className="h-3.5 w-3.5 text-[#39FF14]" />
              <span className="text-xs font-medium text-zinc-400 uppercase tracking-wider">Agent Progress</span>
              {session && (
                <span className="ml-auto font-mono text-[10px] text-zinc-600 truncate max-w-[120px]">
                  {session.research_session_id.slice(0, 8)}…
                </span>
              )}
            </div>
            <ResearchProgressFeed
              researchId={session.research_session_id}
              onComplete={handleComplete}
              onError={handleError}
            />
          </div>
        </ResizablePanel>

        <ResizableHandle withHandle className="bg-zinc-800 w-px" />

        {/* Right: Candidates */}
        <ResizablePanel defaultSize={62} minSize={35}>
          <ScrollArea className="h-full">
            <div className="p-5 space-y-4">

              {/* Session metadata */}
              <div className="flex flex-wrap gap-2">
                <Badge variant="outline" className="h-5 border-zinc-700 px-1.5 text-[10px] text-zinc-400 font-mono">
                  {session.domain}
                </Badge>
                <Badge variant="outline" className="h-5 border-zinc-700 px-1.5 text-[10px] text-zinc-400">
                  {session.category}
                </Badge>
                <Badge variant="outline" className="h-5 border-zinc-700 px-1.5 text-[10px] text-zinc-400">
                  started {new Date(session.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </Badge>
              </div>

              {error && (
                <div className="rounded-lg border border-red-900/50 bg-red-950/30 px-3 py-2">
                  <p className="text-xs text-red-400">{error}</p>
                </div>
              )}

              <CandidateList
                researchId={session.research_session_id}
                polling={sessionStatus === 'running'}
              />
            </div>
          </ScrollArea>
        </ResizablePanel>

      </ResizablePanelGroup>
    </div>
  )
}
