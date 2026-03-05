'use client'

import { useCallback, useEffect, useRef, useState, useMemo } from 'react'
import { ENVIRONMENT_OPTIONS, type HardwareTier } from '@/lib/environment'
import { useEnvironment } from '@/hooks/use-environment'
import {
  getDatasetAnalysis,
  getArtifactStatus,
  getPlatformLimits,
  downloadNotebook,
  downloadModelFile,
  downloadDatasetFile,
  triggerCompile,
  type DatasetAnalysis,
  type ArtifactStatus,
  type PlatformLimits,
} from '@/app/api/platform'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
import { Badge } from '@/components/ui/badge'
import {
  NotebookText, Brain, Monitor, Cpu,
  Table2, Clock, Type, ImageIcon,
  AlertTriangle, CheckCircle2,
  Download, Play, XCircle, Loader2, Zap, Info, FileText,
} from 'lucide-react'
import { TrainingChart, type EpochMetrics } from './training-chart'
import { PlotGallery } from './plot-gallery'

const TYPE_CONFIG: Record<string, { label: string; Icon: React.ElementType; color: string }> = {
  tabular     : { label: 'Tabular',      Icon: Table2,    color: 'text-[#39FF14] border-[#39FF14]/30 bg-[#39FF14]/10' },
  time_series : { label: 'Time Series',  Icon: Clock,     color: 'text-sky-400 border-sky-400/30 bg-sky-400/10' },
  nlp         : { label: 'NLP',          Icon: Type,      color: 'text-violet-400 border-violet-400/30 bg-violet-400/10' },
  image       : { label: 'Image',        Icon: ImageIcon, color: 'text-orange-400 border-orange-400/30 bg-orange-400/10' },
}

const TASK_LABEL: Record<string, string> = {
  binary_classification    : 'Binary classif.',
  multiclass_classification: 'Multi-class',
  regression               : 'Regression',
}

function DatasetSummaryCard({ analysis }: { analysis: DatasetAnalysis }) {
  const cfg        = TYPE_CONFIG[analysis.dataset_type] ?? TYPE_CONFIG.tabular
  const TypeIcon   = cfg.Icon
  const missingPct = (analysis.fraction_missing * 100).toFixed(1)
  const catPct     = (analysis.fraction_categorical * 100).toFixed(0)

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-4 space-y-3">

      {/* Header: type badge + task */}
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <TypeIcon className={`h-3.5 w-3.5 ${cfg.color.split(' ')[0]}`} />
          <span className={`rounded border px-1.5 py-0.5 text-[10px] font-medium ${cfg.color}`}>
            {cfg.label}
          </span>
        </div>
        <span className="text-[10px] text-zinc-500">{TASK_LABEL[analysis.task_type] ?? analysis.task_type}</span>
      </div>

      {/* Row / feature counts */}
      <div className="flex items-center gap-4 text-sm">
        <div>
          <p className="text-xs text-zinc-500">Rows</p>
          <p className="font-mono font-medium text-zinc-200">{analysis.n_rows.toLocaleString()}</p>
        </div>
        <div>
          <p className="text-xs text-zinc-500">Features</p>
          <p className="font-mono font-medium text-zinc-200">{analysis.n_features}</p>
        </div>
        <div>
          <p className="text-xs text-zinc-500">Classes</p>
          <p className="font-mono font-medium text-zinc-200">{analysis.n_classes}</p>
        </div>
      </div>

      {/* Target column */}
      <div className="flex items-center justify-between text-xs">
        <span className="text-zinc-500">Target</span>
        <code className="rounded bg-zinc-800 px-1.5 py-0.5 text-[11px] text-[#39FF14]/80 max-w-[60%] truncate">
          {analysis.target_col}
        </code>
      </div>

      <Separator className="bg-zinc-800" />

      {/* Quality flags */}
      <div className="space-y-1.5">
        <p className="text-[10px] font-medium uppercase tracking-wider text-zinc-600">Quality</p>

        <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
          <div className="flex items-center gap-1.5">
            {analysis.fraction_missing > 0.05
              ? <AlertTriangle className="h-3 w-3 text-amber-400 shrink-0" />
              : <CheckCircle2 className="h-3 w-3 text-[#39FF14]/60 shrink-0" />}
            <span className="text-zinc-400">Missing</span>
            <span className="ml-auto font-mono text-zinc-300">{missingPct}%</span>
          </div>

          <div className="flex items-center gap-1.5">
            {(analysis.class_imbalance_ratio ?? 1) > 5
              ? <AlertTriangle className="h-3 w-3 text-amber-400 shrink-0" />
              : <CheckCircle2 className="h-3 w-3 text-[#39FF14]/60 shrink-0" />}
            <span className="text-zinc-400">Imbalance</span>
            <span className="ml-auto font-mono text-zinc-300">
              {analysis.class_imbalance_ratio != null ? `${analysis.class_imbalance_ratio}×` : 'n/a'}
            </span>
          </div>

          <div className="flex items-center gap-1.5 col-span-2">
            <span className="text-zinc-400">Categorical</span>
            <span className="ml-auto font-mono text-zinc-300">{catPct}%</span>
          </div>
        </div>
      </div>
    </div>
  )
}

function DownloadButton({
  icon: Icon,
  label,
  sublabel,
  onClick,
}: {
  icon     : React.ElementType
  label    : string
  sublabel : string
  onClick  : () => void
}) {
  return (
    <button
      onClick={onClick}
      className="cursor-pointer flex w-full items-center gap-3 rounded-lg border border-zinc-800 bg-zinc-900/60 px-3 py-2.5 text-left transition-colors hover:border-[#39FF14]/40 hover:bg-zinc-900"
    >
      <Icon className="h-4 w-4 shrink-0 text-[#39FF14]" />
      <div className="min-w-0 flex-1">
        <p className="text-sm font-medium text-zinc-200">{label}</p>
        <p className="text-[11px] text-zinc-500">{sublabel}</p>
      </div>
      <Download className="h-3.5 w-3.5 shrink-0 text-zinc-500" />
    </button>
  )
}

function DownloadsSection({
  sessionId,
  status,
}: {
  sessionId: string
  status   : ArtifactStatus
}) {
  const hasArtifacts = status.notebook || status.models.length > 0 || status.images.length > 0 || status.datasets.length > 0

  if (!hasArtifacts) {
    return (
      <div className="flex flex-col items-center gap-3 rounded-lg border border-dashed border-zinc-800 p-6 text-center">
        <div className="flex gap-3 text-[#39FF14]/30">
          <NotebookText className="h-5 w-5" />
          <Brain className="h-5 w-5" />
        </div>
        <p className="text-xs text-zinc-500">
          Your notebook and model will appear here once generated
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {(status.notebook || status.models.length > 0) && (
        <div className="space-y-2">
          {status.notebook && (
            <DownloadButton
              icon={NotebookText}
              label="Training Notebook"
              sublabel="training_notebook.ipynb"
              onClick={() => downloadNotebook(sessionId)}
            />
          )}
          {status.models.map(filename => (
            <DownloadButton
              key={filename}
              icon={Brain}
              label={filename.replace('.keras', '').replace(/_/g, ' ')}
              sublabel={filename}
              onClick={() => downloadModelFile(sessionId, filename)}
            />
          ))}
          {status.datasets.map(filename => (
            <DownloadButton
              key={filename}
              icon={FileText}
              label={filename.replace('.csv', '').replace(/_/g, ' ')}
              sublabel={filename}
              onClick={() => downloadDatasetFile(sessionId, filename)}
            />
          ))}
        </div>
      )}
      {status.epochs_run != null && (
        <div className="flex items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-900/60 px-3 py-2">
          <Zap className="h-3.5 w-3.5 shrink-0 text-[#39FF14]" />
          <span className="text-xs text-zinc-400">Trained for</span>
          <span className="font-mono text-xs font-medium text-zinc-200">
            {status.epochs_run} epoch{status.epochs_run !== 1 ? 's' : ''}
          </span>
          {status.epochs_max != null && status.epochs_run < status.epochs_max && (
            <span className="ml-auto text-[10px] text-zinc-600">
              early stopped / {status.epochs_max} max
            </span>
          )}
        </div>
      )}
      <PlotGallery sessionId={sessionId} images={status.images} />
    </div>
  )
}

interface CompileState {
  phase   : 'idle' | 'running' | 'error'
  progress: number
  step    : string
  error   : string | null
}

// ── Stage definitions ──────────────────────────────────────────────────────
const STAGES = [
  { key: 'queue',   label: 'Queued',    maxProgress: 5   },
  { key: 'loading', label: 'Loading',   maxProgress: 15  },
  { key: 'build',   label: 'Building',  maxProgress: 18  },
  { key: 'train',   label: 'Training',  maxProgress: 91  },
  { key: 'saving',  label: 'Saving',    maxProgress: 100 },
]

function stepToStage(step: string, progress: number): number {
  if (step.includes('Saving'))                                    return 4
  if (step.includes('Evaluating'))                                return 4
  if (step.startsWith('Epoch') || progress > 18)                  return 3
  if (step.includes('Building'))                                  return 2
  if (step.includes('Loading'))                                   return 1
  return 0
}

function CompileRunningState({
  compile,
  metrics,
}: {
  compile: CompileState
  metrics: EpochMetrics[]
}) {
  const stageIdx = stepToStage(compile.step, compile.progress)

  // Elapsed timer
  const [elapsed, setElapsed] = useState(0)
  const startRef       = useRef(Date.now())
  const trainStartRef  = useRef<number | null>(null)
  useEffect(() => {
    startRef.current = Date.now()
    setElapsed(0)
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - startRef.current) / 1000)), 1000)
    return () => clearInterval(id)
  }, [])

  // Track when training stage began (for epoch rate + ETA)
  useEffect(() => {
    if (stageIdx === 3 && trainStartRef.current === null) {
      trainStartRef.current = Date.now()
    }
  }, [stageIdx])

  const elapsedStr = useMemo(() => {
    const m = Math.floor(elapsed / 60)
    const s = elapsed % 60
    return m > 0 ? `${m}m ${s}s` : `${s}s`
  }, [elapsed])

  // Epoch rate + ETA — computed from metrics timestamps
  const { epochRate, eta } = useMemo(() => {
    const lastMetric  = metrics[metrics.length - 1]
    if (!lastMetric || !trainStartRef.current || lastMetric.epoch < 2) return { epochRate: null, eta: null }
    const trainElapsed = (Date.now() - trainStartRef.current) / 1000   // seconds
    const rate         = lastMetric.epoch / trainElapsed                // epochs/sec
    if (rate <= 0) return { epochRate: null, eta: null }
    const remaining    = (lastMetric.total_epochs - lastMetric.epoch) / rate
    const etaMin       = Math.floor(remaining / 60)
    const etaSec       = Math.floor(remaining % 60)
    const rateStr      = rate >= 1 / 60
      ? `${(rate * 60).toFixed(1)} ep/min`
      : `${(1 / rate / 60).toFixed(1)} min/ep`
    const etaStr       = etaMin > 0 ? `~${etaMin}m ${etaSec}s left` : `~${etaSec}s left`
    return { epochRate: rateStr, eta: etaStr }
  }, [metrics, elapsed]) // eslint-disable-line react-hooks/exhaustive-deps

  const isEarlyStage = stageIdx < 3

  return (
    <div className="rounded-lg border border-[#39FF14]/20 bg-[#39FF14]/3 p-4 space-y-4 relative overflow-hidden">

      {/* Animated border shimmer */}
      <div className="pointer-events-none absolute inset-0 rounded-lg">
        <div className="absolute inset-x-0 top-0 h-px bg-linear-to-r from-transparent via-[#39FF14]/60 to-transparent animate-[shimmer_2.5s_ease-in-out_infinite]" />
      </div>

      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="relative flex h-2.5 w-2.5 items-center justify-center">
            <span className="absolute inline-flex h-full w-full rounded-full bg-[#39FF14]/40 animate-ping" />
            <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-[#39FF14]" />
          </div>
          <span className="text-xs font-medium text-zinc-300">
            {compile.step || 'Starting…'}
          </span>
        </div>
        <div className="flex items-center gap-2">
          {epochRate && (
            <span className="font-mono text-[10px] text-zinc-600">{epochRate}</span>
          )}
          {eta && (
            <span className="font-mono text-[10px] text-zinc-500">{eta}</span>
          )}
          {!eta && (
            <span className="font-mono text-[10px] text-zinc-500">{elapsedStr}</span>
          )}
          <span className="font-mono text-xs font-medium text-[#39FF14]">{compile.progress}%</span>
        </div>
      </div>

      {/* Stage track */}
      <div className="flex items-center gap-1">
        {STAGES.map((stage, i) => {
          const done    = i < stageIdx
          const active  = i === stageIdx
          return (
            <div key={stage.key} className="flex flex-1 flex-col items-center gap-1">
              <div className={`h-0.5 w-full rounded-full transition-all duration-700 ${
                done   ? 'bg-[#39FF14]' :
                active ? 'bg-[#39FF14]/50 animate-pulse' :
                         'bg-zinc-800'
              }`} />
              <span className={`text-[9px] font-medium transition-colors duration-300 ${
                done   ? 'text-[#39FF14]/70' :
                active ? 'text-[#39FF14]' :
                         'text-zinc-700'
              }`}>{stage.label}</span>
            </div>
          )
        })}
      </div>

      {/* Progress bar with shimmer */}
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-800/80">
        <div
          className="relative h-full rounded-full bg-[#39FF14] transition-all duration-700 ease-out overflow-hidden"
          style={{ width: `${Math.max(compile.progress, 3)}%` }}
        >
          <div className="absolute inset-0 bg-linear-to-r from-transparent via-white/30 to-transparent animate-[shimmer_1.8s_ease-in-out_infinite]" />
        </div>
      </div>

      {/* Training progress row — visible during training and saving */}
      {stageIdx >= 3 && metrics.length > 0 && (() => {
        const last = metrics[metrics.length - 1]
        return (
          <div className="flex items-center justify-between text-[10px] text-zinc-500 border-t border-zinc-800/60 pt-2">
            <span>
              Epoch <span className="font-mono text-zinc-300">{last.epoch}</span>
              <span className="text-zinc-700"> / {last.total_epochs}</span>
            </span>
            <span className="font-mono text-zinc-500">{elapsedStr} elapsed</span>
          </div>
        )
      })()}

      {/* Stage-specific status hint */}
      {stageIdx === 4 && (
        <div className="flex items-center gap-1.5">
          <Loader2 className="h-3 w-3 text-zinc-600 animate-spin" />
          <span className="text-[10px] text-zinc-600 animate-pulse">
            {compile.step || 'Evaluating & saving outputs…'}
          </span>
        </div>
      )}
      {isEarlyStage && metrics.length === 0 && (
        <div className="flex items-center gap-1.5">
          <Loader2 className="h-3 w-3 text-zinc-600 animate-spin" />
          <span className="text-[10px] text-zinc-600 animate-pulse">
            {stageIdx === 0 ? 'Waiting for worker…' : 'Initialising runtime, this can take 30–60s…'}
          </span>
        </div>
      )}

      {/* Chart once we have data */}
      <TrainingChart metrics={metrics} />
    </div>
  )
}

function CompileSection({
  sessionId,
  status,
  onCompileSuccess,
  onCompileError,
}: {
  sessionId        : string
  status           : ArtifactStatus
  onCompileSuccess : () => void
  onCompileError   : (error: string) => void
}) {
  const [compile, setCompile] = useState<CompileState>({
    phase: 'idle', progress: 0, step: '', error: null,
  })
  const [metrics, setMetrics] = useState<EpochMetrics[]>([])
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Clean up poll on unmount
  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current) }, [])

  // When the notebook is re-saved (mtime advances):
  //   - reset error state so compile button reappears
  //   - auto-trigger compile only if we were in error state (model fixed the script after a failure)
  const prevMtimeRef    = useRef<number | null>(null)
  const startCompileRef = useRef<(() => Promise<void>) | null>(null)
  const taskIdRef       = useRef<string | null>(null)
  useEffect(() => {
    const mtime = status.notebook_mtime
    if (
      mtime !== null &&
      prevMtimeRef.current !== null &&
      mtime > prevMtimeRef.current
    ) {
      setCompile(c => {
        if (c.phase === 'error') {
          // Auto-recompile after the model fixed a failing script
          setTimeout(() => startCompileRef.current?.(), 0)
          return { phase: 'idle', progress: 0, step: '', error: null }
        }
        return c
      })
    }
    prevMtimeRef.current = mtime ?? null
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status.notebook_mtime])

  // Hide if: no notebook yet, OR (model exists AND task is not actively running/errored)
  if (!status.notebook) return null
  if ((status.models.length > 0 || status.datasets.length > 0) && compile.phase === 'idle') return null

  const startCompile = async () => {
    setCompile({ phase: 'running', progress: 0, step: 'Queuing…', error: null })
    setMetrics([])

    const result = await triggerCompile(sessionId)
    if (!result) {
      setCompile({ phase: 'error', progress: 0, step: '', error: 'Failed to start compilation' })
      return
    }

    taskIdRef.current = result.task_id

    // Poll the one-shot JSON endpoint — EventSource doesn't work through Next.js rewrites
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`/api/platform/progress-once/${result.task_id}`)
        if (!res.ok) return
        const data = await res.json() as {
          state: string; progress: number; step: string; error?: string
          metrics?: EpochMetrics
        }
        if (data.state === 'SUCCESS') {
          clearInterval(pollRef.current!)
          pollRef.current = null
          taskIdRef.current = null
          setCompile({ phase: 'idle', progress: 100, step: 'Done', error: null })
          onCompileSuccess()
        } else if (data.state === 'FAILURE') {
          clearInterval(pollRef.current!)
          pollRef.current = null
          taskIdRef.current = null
          const errMsg = data.error ?? 'Compilation failed'
          setCompile({ phase: 'error', progress: 0, step: '', error: errMsg })
          onCompileError(errMsg)
        } else {
          setCompile(prev => ({
            ...prev,
            progress: data.progress ?? prev.progress,
            step    : data.step    ?? prev.step,
          }))
          if (data.metrics) {
            setMetrics(prev => [...prev, data.metrics!])
          }
        }
      } catch {
        // transient fetch error — keep polling
      }
    }, 500)
  }

  // Wire ref for auto-compile trigger from mtime watcher
  startCompileRef.current = startCompile

  const stopCompile = async () => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
    const tid = taskIdRef.current
    taskIdRef.current = null
    setCompile({ phase: 'idle', progress: 0, step: '', error: null })
    if (tid) {
      try { await fetch(`/api/platform/revoke/${tid}`, { method: 'POST' }) } catch { /* best-effort */ }
    }
  }

  return (
    <>
      <Separator className="bg-zinc-800" />

      <div>
        <p className="mb-3 text-xs font-medium uppercase tracking-wider text-zinc-500">
          Compile &amp; Train
        </p>

        {compile.phase === 'idle' && (
          <button
            onClick={startCompile}
            className="cursor-pointer flex w-full items-center justify-center gap-2 rounded-lg border border-[#39FF14]/30 bg-[#39FF14]/5 px-4 py-3 text-sm font-medium text-[#39FF14] transition-colors hover:bg-[#39FF14]/10 hover:border-[#39FF14]/60"
          >
            <Play className="h-4 w-4 fill-[#39FF14]" />
            Compile &amp; Train Model
          </button>
        )}

        {compile.phase === 'running' && (
          <div className="space-y-2">
            <CompileRunningState compile={compile} metrics={metrics} />
            <button
              onClick={stopCompile}
              className="cursor-pointer flex w-full items-center justify-center gap-1.5 rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-xs text-zinc-400 transition-colors hover:border-red-800 hover:bg-red-950/30 hover:text-red-400"
            >
              <XCircle className="h-3.5 w-3.5" />
              Stop
            </button>
          </div>
        )}

        {compile.phase === 'error' && (
          <div className="space-y-2 rounded-lg border border-red-900/50 bg-red-950/30 p-3">
            <div className="flex items-start gap-2">
              <XCircle className="mt-0.5 h-4 w-4 shrink-0 text-red-400" />
              <p className="text-xs text-red-300">{compile.error}</p>
            </div>
            <button
              onClick={() => setCompile({ phase: 'idle', progress: 0, step: '', error: null })}
              className="cursor-pointer text-[10px] text-zinc-500 underline hover:text-zinc-300"
            >
              Dismiss and retry
            </button>
          </div>
        )}
      </div>
    </>
  )
}

function EnvironmentSelector() {
  const { environment, setHardware } = useEnvironment()

  const osLabel: Record<string, string> = {
    windows: 'Windows / WSL2',
    mac    : 'macOS',
    linux  : 'Linux',
    unknown: 'Unknown',
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 text-xs font-medium text-zinc-400 uppercase tracking-wider">
        <Monitor className="h-3.5 w-3.5 text-[#39FF14]" />
        Your Environment
      </div>

      <div className="flex items-center justify-between text-sm">
        <span className="text-zinc-500">OS</span>
        <Badge variant="secondary" className="text-xs">
          {osLabel[environment.os]}
        </Badge>
      </div>

      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-1.5 text-sm text-zinc-500">
          <Cpu className="h-3.5 w-3.5 text-[#39FF14]" />
          Hardware
        </div>
        <Select
          value={environment.hardware}
          onValueChange={val => setHardware(val as HardwareTier)}
        >
          <SelectTrigger className="h-8 w-44 cursor-pointer bg-zinc-900 border-zinc-700 text-xs hover:border-[#39FF14]/50 focus:border-[#39FF14]/50 focus:ring-[#39FF14]/20 transition-colors">
            <SelectValue />
          </SelectTrigger>
          <SelectContent className="bg-zinc-900 border-zinc-700">
            {ENVIRONMENT_OPTIONS.map(opt => (
              <SelectItem key={opt.value} value={opt.value} className="cursor-pointer text-xs focus:bg-[#39FF14]/10 focus:text-[#39FF14]">
                {opt.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <p className="text-[11px] text-zinc-600">
        Affects the{' '}
        <code className="text-zinc-500">!pip install</code> cell in your notebook
      </p>
    </div>
  )
}

interface ArtifactPanelProps {
  sessionId     : string | null
  onCompileError: (error: string) => void
}

const POLL_INTERVAL = 4_000   // ms between status checks

export function ArtifactPanel({ sessionId, onCompileError }: ArtifactPanelProps) {
  const [analysis,      setAnalysis]      = useState<DatasetAnalysis | null>(null)
  const [status,        setStatus]        = useState<ArtifactStatus>({ notebook: false, notebook_mtime: null, models: [], images: [], datasets: [], epochs_run: null, epochs_max: null })
  const [limits,        setLimits]        = useState<PlatformLimits | null>(null)
  const [awaitingPlots, setAwaitingPlots] = useState(false)
  const pollRef                           = useRef<ReturnType<typeof setInterval> | null>(null)
  const fastPollRef                       = useRef<ReturnType<typeof setInterval> | null>(null)
  const fastPollStart                     = useRef<number>(0)
  const FAST_POLL_MS                      = 500
  const FAST_POLL_TIMEOUT_MS              = 30_000  // stop fast-polling after 30s

  useEffect(() => {
    getPlatformLimits().then(l => { if (l) setLimits(l) })
  }, [])

  useEffect(() => {
    if (!sessionId) return

    const loadAnalysis = async () => {
      const result = await getDatasetAnalysis(sessionId)
      if (result) setAnalysis(result)
    }

    loadAnalysis()
    const handler = () => loadAnalysis()
    window.addEventListener('dataset-uploaded', handler)
    return () => window.removeEventListener('dataset-uploaded', handler)
  }, [sessionId])

  useEffect(() => {
    if (!sessionId) return

    const checkStatus = async () => {
      const s = await getArtifactStatus(sessionId)
      if (s) setStatus(s)
    }

    checkStatus()
    pollRef.current = setInterval(checkStatus, POLL_INTERVAL)
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [sessionId])

  // Called by CompileSection when the SSE stream reports SUCCESS
  const onCompileSuccess = useCallback(() => {
    if (!sessionId) return
    setAwaitingPlots(true)
    fastPollStart.current = Date.now()

    const tick = async () => {
      const s = await getArtifactStatus(sessionId)
      if (!s) return
      setStatus(s)
      const elapsed = Date.now() - fastPollStart.current
      // Stop fast-polling once we have images OR after timeout
      if (s.images.length > 0 || elapsed > FAST_POLL_TIMEOUT_MS) {
        clearInterval(fastPollRef.current!)
        fastPollRef.current = null
        setAwaitingPlots(false)
      }
    }

    fastPollRef.current = setInterval(tick, FAST_POLL_MS)
  }, [sessionId])

  useEffect(() => () => { if (fastPollRef.current) clearInterval(fastPollRef.current) }, [])

  return (
    <div className="flex h-full flex-col gap-5 overflow-y-auto bg-zinc-950 p-5">

      {/* Dataset analysis */}
      {analysis && (
        <>
          <div>
            <p className="mb-3 text-xs font-medium uppercase tracking-wider text-zinc-500">
              Dataset
            </p>
            <DatasetSummaryCard analysis={analysis} />
          </div>
          <Separator className="bg-zinc-800" />
        </>
      )}

      {/* Downloads */}
      <div>
        <div className="mb-3 flex items-center gap-2">
          <p className="text-xs font-medium uppercase tracking-wider text-zinc-500">Downloads</p>
          {awaitingPlots && (
            <div className="flex items-center gap-1.5 ml-auto">
              <Loader2 className="h-3 w-3 animate-spin text-zinc-600" />
              <span className="text-[10px] text-zinc-600">Saving plots…</span>
            </div>
          )}
        </div>
        {sessionId
          ? <DownloadsSection sessionId={sessionId} status={status} />
          : (
            <div className="flex flex-col items-center gap-3 rounded-lg border border-dashed border-zinc-800 p-6 text-center">
              <div className="flex gap-3 text-[#39FF14]/30">
                <NotebookText className="h-5 w-5" />
                <Brain className="h-5 w-5" />
              </div>
              <p className="text-xs text-zinc-500">
                Your notebook and model will appear here once generated
              </p>
            </div>
          )
        }
      </div>

      {/* Compile & Train — shown after notebook is ready, before model exists */}
      {sessionId && (
        <CompileSection sessionId={sessionId} status={status} onCompileSuccess={onCompileSuccess} onCompileError={onCompileError} />
      )}

      <Separator className="bg-zinc-800" />

      {/* Setup */}
      <div>
        <p className="mb-3 text-xs font-medium uppercase tracking-wider text-zinc-500">
          Setup
        </p>
        <EnvironmentSelector />
      </div>

      <Separator className="bg-zinc-800" />

      {/* Platform limits */}
      <div>
        <div className="mb-3 flex items-center gap-2">
          <Info className="h-3.5 w-3.5 text-zinc-600" />
          <p className="text-xs font-medium uppercase tracking-wider text-zinc-500">Platform Limits</p>
        </div>
        <div className="space-y-1.5 text-[11px] text-zinc-500">
          <div className="flex justify-between">
            <span>Max training time</span>
            <span className="font-mono text-zinc-400">{limits ? `${limits.max_training_minutes} min` : '…'}</span>
          </div>
          <div className="flex justify-between">
            <span>Max dataset size</span>
            <span className="font-mono text-zinc-400">{limits ? `${limits.max_dataset_mb} MB` : '…'}</span>
          </div>
          <div className="flex justify-between">
            <span>Max memory</span>
            <span className="font-mono text-zinc-400">{limits ? `${limits.max_memory_gb} GB` : '…'}</span>
          </div>
          <div className="flex justify-between">
            <span>Max output files</span>
            <span className="font-mono text-zinc-400">{limits ? `${limits.max_output_gb} GB` : '…'}</span>
          </div>
          <div className="flex justify-between">
            <span>Session TTL</span>
            <span className="font-mono text-zinc-400">{limits ? `${limits.session_ttl_hours} hrs` : '…'}</span>
          </div>
          <div className="flex justify-between">
            <span>Max epochs</span>
            <span className="font-mono text-zinc-400">{limits ? limits.max_epochs : '…'}</span>
          </div>
        </div>
      </div>

    </div>
  )
}
