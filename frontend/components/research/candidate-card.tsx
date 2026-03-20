'use client'

import { useState } from 'react'
import { type ResearchCandidate, compileResearchCandidate } from '@/app/api/platform'
import { Badge } from '@/components/ui/badge'
import {
  Download, ChevronDown, ChevronUp,
  Zap, Brain, Sparkles, Shield,
  Cpu, Timer, Layers, Loader2, Info, GitBranch,
  BookOpen, FlaskConical, ExternalLink,
} from 'lucide-react'

const ACTION_COLORS: Record<string, string> = {
  recurse : 'text-[#39FF14] border-[#39FF14]/40 bg-[#39FF14]/10',
  archive : 'text-blue-400 border-blue-400/40 bg-blue-400/10',
  discard : 'text-zinc-600 border-zinc-700 bg-zinc-800/40',
}

const ACTION_LABELS: Record<string, string> = {
  recurse : 'Recurse',
  archive : 'Archived',
  discard : 'Discarded',
}

interface ScoreBarProps {
  label    : string
  value    : number
  Icon     : React.ElementType
  color    : string   // Tailwind text-* class for icon + label
  barColor : string   // explicit hex for the bar fill (avoids Tailwind purge issues)
}

function ScoreBar({ label, value, Icon, color, barColor }: ScoreBarProps) {
  const pct = Math.round(value * 100)
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-[10px]">
        <div className="flex items-center gap-1 text-zinc-500">
          <Icon className={`h-3 w-3 ${color}`} />
          {label}
        </div>
        <span className={`font-mono font-medium ${color}`}>{pct}%</span>
      </div>
      <div className="h-1 w-full rounded-full bg-zinc-800">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: barColor }}
        />
      </div>
    </div>
  )
}

interface CandidateCardProps {
  candidate   : ResearchCandidate
  researchId  : string
  rank        : number
}

export function CandidateCard({ candidate, researchId, rank }: CandidateCardProps) {
  const [expanded,      setExpanded]      = useState(rank === 0)
  const [showAbout,     setShowAbout]     = useState(false)
  const [showLineage,   setShowLineage]   = useState(false)
  const [downloading,   setDownloading]   = useState(false)

  const hasLineage = (candidate.research_papers?.length ?? 0) > 0 || (candidate.mechanisms?.length ?? 0) > 0

  const actionClass = ACTION_COLORS[candidate.next_action] ?? ACTION_COLORS.discard
  const score       = Math.round(candidate.composite_score * 100)

  // Ring color based on score
  const ringColor = score >= 75 ? 'border-[#39FF14]/50' :
                    score >= 50 ? 'border-blue-500/40'   :
                                  'border-zinc-700'

  const handleDownload = async () => {
    setDownloading(true)
    const result = await compileResearchCandidate(researchId, candidate.architecture_name)
    setDownloading(false)
    if (!result) return

    const blob = new Blob([result.code], { type: 'text/plain' })
    const url  = URL.createObjectURL(blob)
    const a    = document.createElement('a')
    a.href     = url
    a.download = result.filename
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className={`rounded-lg border bg-zinc-900/60 transition-all duration-200 ${ringColor}`}>

      {/* Header */}
      <button
        onClick={() => setExpanded(e => !e)}
        className="cursor-pointer flex w-full items-center gap-3 px-4 py-3 text-left"
      >
        {/* Rank */}
        <span className={`shrink-0 flex h-6 w-6 items-center justify-center rounded-full text-[10px] font-bold
          ${rank === 0 ? 'bg-[#39FF14]/20 text-[#39FF14]' : rank === 1 ? 'bg-blue-500/20 text-blue-400' : 'bg-zinc-800 text-zinc-500'}
        `}>
          {rank + 1}
        </span>

        {/* Name */}
        <div className="flex-1 min-w-0">
          <p className="truncate text-sm font-semibold text-zinc-200">
            {candidate.architecture_name}
          </p>
          <div className="flex items-center gap-2 mt-0.5">
            <span className={`rounded border px-1.5 py-0 text-[10px] font-medium ${actionClass}`}>
              {ACTION_LABELS[candidate.next_action]}
            </span>
            {candidate.param_count > 0 && (
              <span className="text-[10px] text-zinc-600 font-mono">
                {(candidate.param_count / 1e6).toFixed(1)}M params
              </span>
            )}
          </div>
        </div>

        {/* Composite score ring */}
        <div className="shrink-0 flex flex-col items-center gap-0.5">
          <span className={`font-mono text-lg font-bold leading-none ${score >= 75 ? 'text-[#39FF14]' : score >= 50 ? 'text-blue-400' : 'text-zinc-500'}`}>
            {score}
          </span>
          <span className="text-[9px] text-zinc-600 uppercase tracking-wider">score</span>
        </div>

        {expanded ? <ChevronUp className="h-4 w-4 shrink-0 text-zinc-600" /> : <ChevronDown className="h-4 w-4 shrink-0 text-zinc-600" />}
      </button>

      {/* Expanded detail */}
      {expanded && (
        <div className="border-t border-zinc-800 px-4 py-3 space-y-4">

          {/* About this architecture — toggleable */}
          {(candidate.rationale || (candidate.mutations && candidate.mutations.length > 0)) && (
            <div>
              <button
                onClick={() => setShowAbout(x => !x)}
                className="cursor-pointer flex w-full items-center gap-1.5 text-[10px] font-medium uppercase tracking-wider text-zinc-500 hover:text-zinc-300 transition-colors"
              >
                <Info className="h-3 w-3" />
                About this architecture
                <ChevronDown className={`ml-auto h-3 w-3 transition-transform ${showAbout ? 'rotate-180' : ''}`} />
              </button>

              {showAbout && (
                <div className="mt-2 rounded border border-zinc-700/50 bg-zinc-900/60 px-3 py-2.5 space-y-2.5">
                  {candidate.base_template && (
                    <div>
                      <p className="text-[10px] font-semibold uppercase tracking-wide text-zinc-600">Base architecture</p>
                      <p className="mt-0.5 font-mono text-xs text-zinc-300">{candidate.base_template}</p>
                    </div>
                  )}

                  {candidate.mutations && candidate.mutations.length > 0 && (
                    <div>
                      <p className="text-[10px] font-semibold uppercase tracking-wide text-zinc-600">Mutations applied</p>
                      <div className="mt-1 flex flex-wrap gap-1">
                        {candidate.mutations.map(m => (
                          <span key={m} className="flex items-center gap-1 rounded border border-zinc-700 bg-zinc-800 px-1.5 py-0.5 font-mono text-[10px] text-zinc-300">
                            <GitBranch className="h-2.5 w-2.5 text-zinc-500" />
                            {m.replace(/_/g, ' ')}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {candidate.rationale && (
                    <div>
                      <p className="text-[10px] font-semibold uppercase tracking-wide text-zinc-600">Rationale</p>
                      <p className="mt-0.5 text-xs leading-relaxed text-zinc-400">{candidate.rationale}</p>
                    </div>
                  )}

                  {candidate.generation !== undefined && (
                    <div>
                      <p className="text-[10px] font-semibold uppercase tracking-wide text-zinc-600">Discovered in</p>
                      <p className="mt-0.5 text-xs text-zinc-400">Generation {candidate.generation}</p>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Research Lineage */}
          {hasLineage && (
            <div>
              <button
                onClick={() => setShowLineage(x => !x)}
                className="cursor-pointer flex w-full items-center gap-1.5 text-[10px] font-medium uppercase tracking-wider text-zinc-500 hover:text-zinc-300 transition-colors"
              >
                <BookOpen className="h-3 w-3" />
                Research lineage
                <ChevronDown className={`ml-auto h-3 w-3 transition-transform ${showLineage ? 'rotate-180' : ''}`} />
              </button>

              {showLineage && (
                <div className="mt-2 space-y-3">

                  {/* Papers */}
                  {candidate.research_papers && candidate.research_papers.length > 0 && (
                    <div className="rounded border border-zinc-700/50 bg-zinc-900/60 px-3 py-2.5 space-y-2">
                      <p className="text-[10px] font-semibold uppercase tracking-wide text-zinc-500 flex items-center gap-1">
                        <BookOpen className="h-3 w-3" /> Papers that informed this design
                      </p>
                      {candidate.research_papers.map((paper, i) => (
                        <div key={i} className="space-y-0.5">
                          <a
                            href={`https://arxiv.org/abs/${paper.arxiv_id}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-start gap-1 text-[11px] font-medium text-blue-400 hover:text-blue-300 transition-colors leading-tight"
                          >
                            <ExternalLink className="h-2.5 w-2.5 mt-0.5 shrink-0" />
                            {paper.title}
                          </a>
                          {paper.abstract && (
                            <p className="text-[10px] text-zinc-500 leading-relaxed pl-3.5 line-clamp-2">{paper.abstract}</p>
                          )}
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Mathematical mechanisms */}
                  {candidate.mechanisms && candidate.mechanisms.length > 0 && (
                    <div className="rounded border border-zinc-700/50 bg-zinc-900/60 px-3 py-2.5 space-y-2">
                      <p className="text-[10px] font-semibold uppercase tracking-wide text-zinc-500 flex items-center gap-1">
                        <FlaskConical className="h-3 w-3" /> Derived mathematical mechanisms
                      </p>
                      {candidate.mechanisms.map((mech, i) => (
                        <div key={i} className="space-y-1 border-t border-zinc-800 pt-2 first:border-t-0 first:pt-0">
                          <div className="flex items-center gap-1.5">
                            <span className="font-mono text-[11px] font-semibold text-purple-400">{mech.name.replace(/_/g, ' ')}</span>
                            {mech.sympy_valid && (
                              <span className="rounded border border-purple-500/30 bg-purple-500/10 px-1 py-0 text-[9px] text-purple-400">✓ valid math</span>
                            )}
                          </div>
                          <p className="text-[10px] text-zinc-400 leading-relaxed">{mech.description}</p>
                          {mech.sympy_expression && (
                            <div className="rounded bg-zinc-950 border border-zinc-800 px-2 py-1">
                              <code className="font-mono text-[10px] text-amber-300">{mech.sympy_expression}</code>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}

                </div>
              )}
            </div>
          )}

          {/* Score breakdown */}
          <div className="space-y-2">
            <p className="text-[10px] font-medium uppercase tracking-wider text-zinc-600">Score Breakdown</p>
            <ScoreBar
              label="Novelty"
              value={candidate.novelty_score}
              Icon={Sparkles}
              color="text-purple-400"
              barColor="#c084fc"
            />
            <p className="text-[10px] text-zinc-600 leading-relaxed -mt-1 pl-4">
              How different this architecture is from previously discovered ones. Measured by embedding distance in the FAISS novelty index.
            </p>
            <ScoreBar
              label="Efficiency"
              value={candidate.efficiency_score}
              Icon={Zap}
              color="text-amber-400"
              barColor="#fbbf24"
            />
            <p className="text-[10px] text-zinc-600 leading-relaxed -mt-1 pl-4">
              Resource efficiency — lower memory usage, faster inference, fewer parameters, and shorter training time all increase this score.
            </p>
            <ScoreBar
              label="Soundness"
              value={candidate.soundness_score}
              Icon={Shield}
              color="text-blue-400"
              barColor="#60a5fa"
            />
            <p className="text-[10px] text-zinc-600 leading-relaxed -mt-1 pl-4">
              LLM judge rating of theoretical correctness — does the architecture make sense, are there obvious design flaws, is the training loss reasonable?
            </p>
            <ScoreBar
              label="Generalization"
              value={candidate.generalization_score}
              Icon={Brain}
              color="text-[#39FF14]"
              barColor="#39FF14"
            />
            <p className="text-[10px] text-zinc-600 leading-relaxed -mt-1 pl-4">
              How well the trained model performs on a held-out validation set. High generalization means the architecture avoids overfitting.
            </p>
          </div>

          {/* Hardware metrics */}
          {(candidate.memory_mb > 0 || candidate.inference_time_ms > 0) && (
            <div className="grid grid-cols-3 gap-3 rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
              <div className="space-y-0.5">
                <div className="flex items-center gap-1 text-[10px] text-zinc-600">
                  <Cpu className="h-3 w-3" /> Memory
                </div>
                <p className="font-mono text-xs font-medium text-zinc-300">
                  {candidate.memory_mb > 0 ? `${candidate.memory_mb.toFixed(0)} MB` : '—'}
                </p>
              </div>
              <div className="space-y-0.5">
                <div className="flex items-center gap-1 text-[10px] text-zinc-600">
                  <Timer className="h-3 w-3" /> Inference
                </div>
                <p className="font-mono text-xs font-medium text-zinc-300">
                  {candidate.inference_time_ms < 9000 ? `${candidate.inference_time_ms.toFixed(0)} ms` : '—'}
                </p>
              </div>
              <div className="space-y-0.5">
                <div className="flex items-center gap-1 text-[10px] text-zinc-600">
                  <Layers className="h-3 w-3" /> Params
                </div>
                <p className="font-mono text-xs font-medium text-zinc-300">
                  {candidate.param_count > 0 ? `${(candidate.param_count / 1e6).toFixed(2)}M` : '—'}
                </p>
              </div>
            </div>
          )}

          {/* Synthetic metrics */}
          {Object.keys(candidate.synthetic_metrics).length > 0 && (
            <div className="space-y-1">
              <p className="text-[10px] font-medium uppercase tracking-wider text-zinc-600">Training Metrics</p>
              <div className="flex flex-wrap gap-3">
                {Object.entries(candidate.synthetic_metrics)
                  .filter(([k]) => !['error'].includes(k))
                  .map(([k, v]) => (
                    <div key={k} className="space-y-0">
                      <p className="text-[10px] text-zinc-600">{k}</p>
                      <p className="font-mono text-xs text-zinc-300">
                        {typeof v === 'number' ? v.toFixed(4) : String(v)}
                      </p>
                    </div>
                  ))}
              </div>
            </div>
          )}

          {/* Download button */}
          {candidate.next_action !== 'discard' && (
            <button
              onClick={handleDownload}
              disabled={downloading}
              className="cursor-pointer flex w-full items-center justify-center gap-2 rounded-lg border border-[#39FF14]/30 bg-[#39FF14]/5 px-3 py-2 text-xs font-medium text-[#39FF14] transition-colors hover:border-[#39FF14]/60 hover:bg-[#39FF14]/10 disabled:opacity-50"
            >
              {downloading
                ? <><Loader2 className="h-3.5 w-3.5 animate-spin" /> Compiling…</>
                : <><Download className="h-3.5 w-3.5" /> Download Training Script</>
              }
            </button>
          )}
        </div>
      )}
    </div>
  )
}
