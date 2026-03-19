'use client'

import { useEffect, useState, useCallback } from 'react'
import { getResearchCandidates, type ResearchCandidate } from '@/app/api/platform'
import { CandidateCard } from './candidate-card'
import { Loader2, Trophy } from 'lucide-react'

interface CandidateListProps {
  researchId  : string
  /** When true, polls for new candidates every 10s (session still running) */
  polling     : boolean
}

export function CandidateList({ researchId, polling }: CandidateListProps) {
  const [candidates, setCandidates] = useState<ResearchCandidate[]>([])
  const [loading,    setLoading]    = useState(true)

  const load = useCallback(async () => {
    const data = await getResearchCandidates(researchId)
    if (data) setCandidates(data)
    setLoading(false)
  }, [researchId])

  // Initial load
  useEffect(() => { load() }, [load])

  // Polling while session is active
  useEffect(() => {
    if (!polling) return
    const id = setInterval(load, 10_000)
    return () => clearInterval(id)
  }, [polling, load])

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-5 w-5 text-zinc-700 animate-spin" />
      </div>
    )
  }

  if (candidates.length === 0) {
    return (
      <div className="flex flex-col items-center gap-3 rounded-lg border border-dashed border-zinc-800 py-10 text-center">
        <Trophy className="h-6 w-6 text-zinc-700" />
        <p className="text-xs text-zinc-600">Candidates will appear here as agents complete scoring</p>
      </div>
    )
  }

  const toShow    = candidates.filter(c => c.next_action !== 'discard')
  const discarded = candidates.filter(c => c.next_action === 'discard')

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Trophy className="h-4 w-4 text-[#39FF14]" />
        <p className="text-xs font-medium uppercase tracking-wider text-zinc-500">
          Discovered Architectures
        </p>
        <span className="ml-auto rounded border border-zinc-700 bg-zinc-800 px-1.5 py-0.5 text-[10px] font-mono text-zinc-400">
          {candidates.length}
        </span>
      </div>

      {/* Recurse + Archive candidates */}
      <div className="space-y-2">
        {toShow.map((c, i) => (
          <CandidateCard
            key={c.architecture_name}
            candidate={c}
            researchId={researchId}
            rank={i}
          />
        ))}
      </div>

      {/* Discarded — collapsed summary */}
      {discarded.length > 0 && (
        <details className="group">
          <summary className="cursor-pointer list-none">
            <div className="flex items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-900/30 px-3 py-2 text-xs text-zinc-600 hover:text-zinc-400 transition-colors">
              <span className="flex-1">
                {discarded.length} discarded candidate{discarded.length !== 1 ? 's' : ''} (score ≤ 50%)
              </span>
              <span className="group-open:rotate-90 transition-transform">›</span>
            </div>
          </summary>
          <div className="mt-2 space-y-2">
            {discarded.map((c, i) => (
              <CandidateCard
                key={c.architecture_name}
                candidate={c}
                researchId={researchId}
                rank={toShow.length + i}
              />
            ))}
          </div>
        </details>
      )}
    </div>
  )
}
