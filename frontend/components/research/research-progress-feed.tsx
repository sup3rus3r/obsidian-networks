'use client'

import { useEffect, useRef, useState } from 'react'
import { AppRoutes } from '@/app/api/routes'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  FlaskConical, Brain, Code2, Cpu, BarChart3,
  CheckCircle2, XCircle, Loader2, ChevronRight,
  BookOpen, FunctionSquare, Layers, Zap,
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

const EVENT_ICONS: Record<string, React.ElementType> = {
  researcher_start    : BookOpen,
  researcher_done     : BookOpen,
  mathematician_start : FunctionSquare,
  mathematician_done  : FunctionSquare,
  architect_start     : Layers,
  architect_done      : Layers,
  coder_start         : Code2,
  coder_done          : Code2,
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
  generation_complete : FlaskConical,
  session_complete    : CheckCircle2,
  session_error       : XCircle,
  session_cancelled   : XCircle,
}

const EVENT_COLORS: Record<string, string> = {
  session_complete  : 'text-[#39FF14]',
  session_error     : 'text-red-400',
  session_cancelled : 'text-zinc-500',
  generation_start  : 'text-blue-400',
  generation_complete: 'text-blue-300',
  agent_done        : 'text-[#39FF14]/80',
}

function AgentBadge({ name }: { name: string }) {
  return (
    <span className="inline-flex items-center gap-1 rounded border border-zinc-700 bg-zinc-800 px-1.5 py-0.5 font-mono text-[10px] text-zinc-400">
      {name}
    </span>
  )
}

function EventRow({ event, idx }: { event: ProgressEvent; idx: number }) {
  const type = event.event_type ?? ''
  const Icon = EVENT_ICONS[type] ?? ChevronRight
  const color = EVENT_COLORS[type] ?? 'text-zinc-400'
  const isSpinner = type === 'agent_start' || type.endsWith('_start')
  const time = new Date(event.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })

  return (
    <div className="flex items-start gap-2.5 py-1.5 border-b border-zinc-800/50 last:border-0">
      <span className="mt-0.5 shrink-0">
        <Icon className={`h-3.5 w-3.5 ${color} ${isSpinner && event.event_type !== 'agent_done' ? 'animate-spin' : ''}`} />
      </span>
      <div className="flex-1 min-w-0">
        <p className={`text-xs leading-snug ${color}`}>
          {event.message || type.replace(/_/g, ' ')}
        </p>
        {event.data && Object.keys(event.data).length > 0 && (
          <div className="mt-1 flex flex-wrap gap-1">
            {Object.entries(event.data).slice(0, 4).map(([k, v]) => (
              <span key={k} className="text-[10px] text-zinc-600">
                {k}: <span className="text-zinc-400 font-mono">{String(v).slice(0, 30)}</span>
              </span>
            ))}
          </div>
        )}
        {event.generation !== undefined && (
          <div className="mt-0.5 flex items-center gap-2">
            <span className="text-[10px] text-zinc-700">
              gen <span className="font-mono text-zinc-600">{event.generation}</span>
            </span>
          </div>
        )}
      </div>
      <span className="shrink-0 font-mono text-[10px] text-zinc-700">{time}</span>
    </div>
  )
}

interface ResearchProgressFeedProps {
  researchId : string
  onComplete ?: () => void
  onError    ?: (msg: string) => void
}

export function ResearchProgressFeed({ researchId, onComplete, onError }: ResearchProgressFeedProps) {
  const [events,    setEvents]    = useState<ProgressEvent[]>([])
  const [connected, setConnected] = useState(false)
  const [done,      setDone]      = useState(false)
  const bottomRef                 = useRef<HTMLDivElement>(null)
  const esRef                     = useRef<EventSource | null>(null)

  useEffect(() => {
    const url = AppRoutes.ResearchStream(researchId)
    const es  = new EventSource(url)
    esRef.current = es

    es.addEventListener('connected', () => setConnected(true))

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
        }
      } catch { /* malformed — skip */ }
    })

    es.onerror = () => {
      if (!done) setConnected(false)
    }

    return () => { es.close() }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [researchId])

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

      {/* Event log */}
      <ScrollArea className="flex-1 px-4 py-2">
        {events.length === 0 && (
          <div className="flex flex-col items-center gap-2 py-8 text-center">
            <Loader2 className="h-5 w-5 text-zinc-700 animate-spin" />
            <p className="text-xs text-zinc-600">Waiting for first agent…</p>
          </div>
        )}
        {events.map((event, i) => (
          <EventRow key={i} event={event} idx={i} />
        ))}
        <div ref={bottomRef} />
      </ScrollArea>
    </div>
  )
}
