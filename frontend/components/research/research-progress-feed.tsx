'use client'

import { useEffect, useRef, useState } from 'react'
import { AppRoutes } from '@/app/api/routes'
import { continueResearchSession, cancelResearchSession } from '@/app/api/platform'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  FlaskConical, Brain, Code2, Cpu, BarChart3,
  CheckCircle2, XCircle, Loader2, ChevronRight,
  BookOpen, FunctionSquare, Layers, Zap, ChevronDown,
  HelpCircle, ShieldCheck,
} from 'lucide-react'

interface ProgressEvent {
  event_type          : string
  research_session_id : string
  generation         ?: number
  depth              ?: number
  message            ?: string
  data               ?: Record<string, unknown>
  timestamp           : string
}

interface DecisionPrompt {
  consecutiveFailures : number
  bestScore           : number
  message             : string
}

const EVENT_ICONS: Record<string, React.ElementType> = {
  researcher_start    : BookOpen,
  researcher_done     : BookOpen,
  mathematician_start : FunctionSquare,
  mathematician_done  : FunctionSquare,
  architect_start     : Layers,
  architect_done      : Layers,
  coder_start              : Code2,
  coder_done               : Code2,
  code_validator_start     : ShieldCheck,
  code_validator_done      : ShieldCheck,
  trainer_start       : Cpu,
  trainer_done        : Cpu,
  evaluator_start     : BarChart3,
  evaluator_done      : BarChart3,
  validator_start     : CheckCircle2,
  validator_done      : CheckCircle2,
  critic_start        : Brain,
  critic_done         : Brain,
  agent_start         : Loader2,
  agent_done          : Zap,
  generation_start    : FlaskConical,
  generation_complete      : FlaskConical,
  gen0_retry               : FlaskConical,
  session_complete         : CheckCircle2,
  session_error            : XCircle,
  session_cancelled        : XCircle,
  awaiting_user_decision   : HelpCircle,
  session_resumed          : FlaskConical,
}

const EVENT_COLORS: Record<string, string> = {
  session_complete       : 'text-[#39FF14]',
  session_error          : 'text-red-400',
  session_cancelled      : 'text-zinc-500',
  generation_start       : 'text-blue-400',
  generation_complete    : 'text-blue-300',
  gen0_retry             : 'text-amber-300',
  agent_done             : 'text-[#39FF14]/80',
  code_validator_done    : 'text-cyan-400',
  awaiting_user_decision : 'text-amber-400',
  session_resumed        : 'text-blue-400',
}

function AgentBadge({ name }: { name: string }) {
  return (
    <span className="inline-flex items-center gap-1 rounded border border-zinc-700 bg-zinc-800 px-1.5 py-0.5 font-mono text-[10px] text-zinc-400">
      {name}
    </span>
  )
}

function EventRow({
  event,
  idx,
  allEvents,
}: {
  event    : ProgressEvent
  idx      : number
  allEvents: ProgressEvent[]
}) {
  const [expanded, setExpanded] = useState(false)
  const type    = event.event_type ?? ''
  const Icon    = EVENT_ICONS[type] ?? ChevronRight
  const color   = EVENT_COLORS[type] ?? 'text-zinc-400'
  const time    = new Date(event.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })

  // Spinner only while the corresponding _done event hasn't arrived yet
  const isStartEvent = type === 'agent_start' || type.endsWith('_start')
  const doneType     = isStartEvent
    ? (type === 'agent_start' ? 'agent_done' : type.replace('_start', '_done'))
    : null
  const isCompleted  = doneType
    ? allEvents.slice(idx + 1).some(e => e.event_type === doneType)
    : false
  const shouldSpin   = isStartEvent && !isCompleted

  const hasData = event.data && Object.keys(event.data).length > 0

  return (
    <div className="border-b border-zinc-800/50 last:border-0">
      <div
        className={`flex items-start gap-2.5 py-1.5 ${hasData ? 'cursor-pointer hover:bg-zinc-800/30' : ''}`}
        onClick={() => hasData && setExpanded(x => !x)}
      >
        <span className="mt-0.5 shrink-0">
          <Icon className={`h-3.5 w-3.5 ${color} ${shouldSpin ? 'animate-spin' : ''}`} />
        </span>
        <div className="flex-1 min-w-0">
          <p className={`text-xs leading-snug ${color}`}>
            {event.message || type.replace(/_/g, ' ')}
          </p>
          {/* Preview: first 2 data keys inline, rest hidden until expanded */}
          {hasData && !expanded && (
            <div className="mt-1 flex flex-wrap gap-1">
              {Object.entries(event.data!).slice(0, 2).map(([k, v]) => (
                <span key={k} className="text-[10px] text-zinc-600">
                  {k}: <span className="text-zinc-400 font-mono">{String(v).slice(0, 40)}</span>
                </span>
              ))}
              {Object.keys(event.data!).length > 2 && (
                <span className="text-[10px] text-zinc-700">+{Object.keys(event.data!).length - 2} more</span>
              )}
            </div>
          )}
          {event.generation !== undefined && (
            <span className="text-[10px] text-zinc-700">
              gen <span className="font-mono text-zinc-600">{event.generation}</span>
            </span>
          )}
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          <span className="font-mono text-[10px] text-zinc-700">{time}</span>
          {hasData && (
            <ChevronDown className={`h-3 w-3 text-zinc-600 transition-transform ${expanded ? 'rotate-180' : ''}`} />
          )}
        </div>
      </div>

      {/* Expanded detail panel */}
      {expanded && hasData && (
        <div className="ml-6 mb-2 rounded border border-zinc-700/50 bg-zinc-900 px-3 py-2">
          {Object.entries(event.data!).map(([k, v]) => {
            const displayVal = Array.isArray(v)
              ? (v as unknown[]).join(', ')
              : typeof v === 'object' && v !== null
                ? JSON.stringify(v, null, 2)
                : String(v)
            return (
              <div key={k} className="mb-1.5 last:mb-0">
                <span className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wide">{k}</span>
                <pre className="mt-0.5 whitespace-pre-wrap break-all font-mono text-[10px] text-zinc-300 leading-relaxed">
                  {displayVal}
                </pre>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

interface ResearchProgressFeedProps {
  researchId : string
  onComplete ?: () => void
  onError    ?: (msg: string) => void
}

export function ResearchProgressFeed({ researchId, onComplete, onError }: ResearchProgressFeedProps) {
  const [events,         setEvents]         = useState<ProgressEvent[]>([])
  const [connected,      setConnected]      = useState(false)
  const [done,           setDone]           = useState(false)
  const [decisionPrompt, setDecisionPrompt] = useState<DecisionPrompt | null>(null)
  const [deciding,       setDeciding]       = useState(false)
  const bottomRef                           = useRef<HTMLDivElement>(null)
  const esRef                               = useRef<EventSource | null>(null)

  useEffect(() => {
    const url = AppRoutes.ResearchStream(researchId)
    const es  = new EventSource(url)
    esRef.current = es

    es.addEventListener('connected', async () => {
      setConnected(true)
      // Recover banner if the session was already paused when we (re)connected
      try {
        const res = await fetch(AppRoutes.ResearchStatus(researchId))
        if (res.ok) {
          const doc = await res.json()
          if (doc.status === 'awaiting_decision') {
            const failures = doc.consecutive_failures ?? 3
            const score    = doc.best_score ?? 0
            setDecisionPrompt({
              consecutiveFailures: failures,
              bestScore          : Math.round(score * 100),
              message            : `Tried ${failures} times to improve without finding strong candidates (best score: ${Math.round(score * 100)}%). Continue exploring or stop?`,
            })
          }
        }
      } catch { /* ignore — banner will appear via live event if stream is healthy */ }
    })

    es.addEventListener('progress', (e) => {
      try {
        const data: ProgressEvent = JSON.parse((e as MessageEvent).data)
        if (!data.event_type) return
        setEvents(prev => [...prev, data])

        if (data.event_type === 'session_complete') {
          setDone(true)
          es.close()
          onComplete?.()
        } else if (data.event_type === 'session_error') {
          setDone(true)
          es.close()
          onError?.(data.message ?? 'Research session failed')
        } else if (data.event_type === 'session_cancelled') {
          setDone(true)
          es.close()
        } else if (data.event_type === 'awaiting_user_decision') {
          setDecisionPrompt({
            consecutiveFailures: (data as any).consecutive_failures ?? 3,
            bestScore          : Math.round(((data as any).best_score ?? 0) * 100),
            message            : data.message ?? '',
          })
        } else if (data.event_type === 'session_resumed') {
          setDecisionPrompt(null)
        }
      } catch { /* malformed — skip */ }
    })

    es.onerror = () => {
      if (!done) setConnected(false)
    }

    return () => { es.close() }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [researchId])

  const handleContinue = async () => {
    setDeciding(true)
    await continueResearchSession(researchId)
    setDeciding(false)
    // banner clears when session_resumed event arrives
  }

  const handleStop = async () => {
    setDeciding(true)
    await cancelResearchSession(researchId)
    setDecisionPrompt(null)
    setDone(true)
    esRef.current?.close()
    setDeciding(false)
  }

  // Auto-scroll to bottom
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [events])

  // Group events by generation for clarity
  const lastGen = events.reduce<number>((max, e) => Math.max(max, e.generation ?? 0), 0)

  return (
    <div className="flex flex-col h-full">
      {/* Status bar */}
      <div className="flex items-center gap-2 px-4 py-2 border-b border-zinc-800 shrink-0">
        {done ? (
          events.some(e => e.event_type === 'session_complete')
            ? <CheckCircle2 className="h-3.5 w-3.5 text-[#39FF14]" />
            : <XCircle className="h-3.5 w-3.5 text-red-400" />
        ) : connected ? (
          <>
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full rounded-full bg-[#39FF14]/40 animate-ping" />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-[#39FF14]" />
            </span>
          </>
        ) : (
          <Loader2 className="h-3.5 w-3.5 text-zinc-600 animate-spin" />
        )}
        <span className="text-xs text-zinc-400">
          {done
            ? events.some(e => e.event_type === 'session_complete') ? 'Session complete' : 'Session ended'
            : connected ? `Live · Generation ${lastGen}` : 'Connecting…'}
        </span>
        <span className="ml-auto text-[10px] text-zinc-700 font-mono">{events.length} events</span>
      </div>

      {/* User decision banner */}
      {decisionPrompt && !done && (
        <div className="shrink-0 mx-4 my-2 rounded-lg border border-amber-500/40 bg-amber-950/30 px-4 py-3">
          <div className="flex items-start gap-2 mb-3">
            <HelpCircle className="h-4 w-4 text-amber-400 mt-0.5 shrink-0" />
            <div>
              <p className="text-xs font-semibold text-amber-300">Research needs your input</p>
              <p className="text-xs text-amber-200/70 mt-0.5">
                After {decisionPrompt.consecutiveFailures} improvement attempts, the best score is{' '}
                <span className="font-mono text-amber-300">{decisionPrompt.bestScore}%</span>.
                Continue with fresh mathematical mechanisms, or stop here?
              </p>
            </div>
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleContinue}
              disabled={deciding}
              className="flex-1 rounded border border-[#39FF14]/40 bg-[#39FF14]/8 px-3 py-1.5 text-xs font-semibold text-[#39FF14] transition-all hover:border-[#39FF14]/70 hover:bg-[#39FF14]/15 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {deciding ? <Loader2 className="h-3 w-3 animate-spin mx-auto" /> : 'Continue Exploring'}
            </button>
            <button
              onClick={handleStop}
              disabled={deciding}
              className="flex-1 rounded border border-zinc-700 bg-zinc-800 px-3 py-1.5 text-xs font-semibold text-zinc-400 transition-all hover:border-zinc-600 hover:text-zinc-300 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Stop Research
            </button>
          </div>
        </div>
      )}

      {/* Event log */}
      <ScrollArea className="flex-1 px-4 py-2">
        {events.length === 0 && (
          <div className="flex flex-col items-center gap-2 py-8 text-center">
            <Loader2 className="h-5 w-5 text-zinc-700 animate-spin" />
            <p className="text-xs text-zinc-600">Waiting for first agent…</p>
          </div>
        )}
        {events.map((event, i) => (
          <EventRow key={i} event={event} idx={i} allEvents={events} />
        ))}
        <div ref={bottomRef} />
      </ScrollArea>
    </div>
  )
}
