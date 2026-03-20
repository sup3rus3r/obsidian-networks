'use client'

import { useState, useEffect } from 'react'
import {
  getResearchCategories,
  startResearchSession,
  type ResearchCategory,
  type ResearchSession,
} from '@/app/api/platform'
import { CategorySelector } from './category-selector'
import { Textarea } from '@/components/ui/textarea'
import { Loader2, FlaskConical } from 'lucide-react'

// Default prompts per category — pre-filled when user selects a category
const CATEGORY_DEFAULT_PROMPTS: Record<string, string> = {
  vision                : 'Discover efficient image classification architectures under 10M parameters that maximize accuracy on CIFAR-10.',
  text                  : 'Find novel transformer variants for long-context text classification with reduced memory footprint.',
  audio                 : 'Explore compact architectures for environmental sound classification optimized for real-time inference.',
  timeseries            : 'Discover forecasting architectures for multivariate time series with minimal lag and high accuracy.',
  graph                 : 'Find attention-based GNN architectures for node classification on citation networks.',
  multimodal_text_image : 'Discover cross-modal architectures that align image and text representations for zero-shot classification.',
  tabular               : 'Explore deep learning architectures that consistently outperform XGBoost on structured tabular data.',
  recommendation        : 'Find embedding-based architectures for collaborative filtering with cold-start resilience.',
  generative            : 'Discover VAE variants with disentangled latent representations for controllable image generation.',
  reinforcement_learning: 'Discover efficient policy network architectures for continuous control tasks like LunarLander and CartPole.',
}

// Domain options per category
const CATEGORY_DOMAINS: Record<string, string[]> = {
  vision                : ['vision'],
  text                  : ['language'],
  audio                 : ['audio'],
  timeseries            : ['timeseries'],
  graph                 : ['graph'],
  multimodal_text_image : ['multimodal'],
  tabular               : ['tabular'],
  recommendation        : ['recommendation'],
  generative            : ['generative'],
  reinforcement_learning: ['rl'],
}

interface ResearchLaunchFormProps {
  onSessionStarted: (session: ResearchSession) => void
}

export function ResearchLaunchForm({ onSessionStarted }: ResearchLaunchFormProps) {
  const [categories,   setCategories]   = useState<ResearchCategory[]>([])
  const [category,     setCategory]     = useState('vision')
  const [description,  setDescription]  = useState(CATEGORY_DEFAULT_PROMPTS['vision'] ?? '')
  const [population,    setPopulation]    = useState(3)
  const [maxGen,        setMaxGen]        = useState(3)
  const [gen0Retries,   setGen0Retries]   = useState(3)
  const [submitting,    setSubmitting]    = useState(false)
  const [error,        setError]        = useState<string | null>(null)

  useEffect(() => {
    getResearchCategories().then(cats => { if (cats) setCategories(cats) })
  }, [])

  const handleCategorySelect = (id: string) => {
    setCategory(id)
    setDescription(CATEGORY_DEFAULT_PROMPTS[id] ?? '')
  }

  const domain = CATEGORY_DOMAINS[category]?.[0] ?? 'vision'

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!description.trim()) { setError('Describe what you want to discover'); return }
    setError(null)
    setSubmitting(true)

    const session = await startResearchSession({
      domain,
      category,
      task_description           : description.trim(),
      population_size            : population,
      max_generations            : maxGen,
      max_gen0_retries           : gen0Retries,
      enable_real_data_validation: false,
    })

    setSubmitting(false)
    if (!session) {
      setError('Failed to start research session. Is the backend running?')
      return
    }
    onSessionStarted(session)
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-5">

      {/* Category */}
      <div className="space-y-2">
        <label className="text-xs font-medium uppercase tracking-wider text-zinc-500">
          Modality
        </label>
        <CategorySelector
          categories={categories}
          selected={category}
          onSelect={handleCategorySelect}
        />
      </div>

      {/* Task description */}
      <div className="space-y-2">
        <label className="text-xs font-medium uppercase tracking-wider text-zinc-500">
          Research Goal
        </label>
        <Textarea
          value={description}
          onChange={e => setDescription(e.target.value)}
          placeholder={
            category === 'vision'    ? 'e.g. Discover efficient image classification architectures under 10M params...' :
            category === 'text'      ? 'e.g. Find novel transformer variants for long-context understanding...' :
            category === 'audio'     ? 'e.g. Explore architectures for environmental sound classification...' :
            category === 'timeseries'? 'e.g. Discover forecasting architectures for multivariate time series...' :
            category === 'graph'     ? 'e.g. Find attention-based GNN architectures for node classification...' :
            category === 'tabular'   ? 'e.g. Explore deep learning architectures that beat XGBoost on tabular data...' :
            category === 'generative'? 'e.g. Discover VAE variants with disentangled latent representations...' :
            'Describe the architecture discovery goal...'
          }
          rows={3}
          className="resize-none bg-zinc-900 border-zinc-700 text-sm text-zinc-200 placeholder:text-zinc-600 focus:border-[#39FF14]/50 focus:ring-[#39FF14]/20"
        />
      </div>

      {/* Population + Generations */}
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <label className="text-xs font-medium uppercase tracking-wider text-zinc-500">
            Candidates / Generation
          </label>
          <div className="flex items-center gap-2">
            <input
              type="range"
              min={2} max={20} step={1}
              value={Math.min(population, 20)}
              onChange={e => setPopulation(Number(e.target.value))}
              className="flex-1 accent-[#39FF14]"
            />
            <input
              type="number"
              min={2}
              value={population}
              onChange={e => setPopulation(Math.max(2, Number(e.target.value) || 2))}
              className="w-14 rounded border border-zinc-700 bg-zinc-900 px-1.5 py-0.5 text-center font-mono text-sm text-zinc-200 focus:border-[#39FF14]/50 focus:outline-none"
            />
          </div>
          <p className="text-[10px] text-zinc-600">
            More candidates → broader search, longer runtime
          </p>
        </div>

        <div className="space-y-2">
          <label className="text-xs font-medium uppercase tracking-wider text-zinc-500">
            Max Generations
          </label>
          <div className="flex items-center gap-2">
            <input
              type="range"
              min={1} max={20} step={1}
              value={Math.min(maxGen, 20)}
              onChange={e => setMaxGen(Number(e.target.value))}
              className="flex-1 accent-[#39FF14]"
            />
            <input
              type="number"
              min={1}
              value={maxGen}
              onChange={e => setMaxGen(Math.max(1, Number(e.target.value) || 1))}
              className="w-14 rounded border border-zinc-700 bg-zinc-900 px-1.5 py-0.5 text-center font-mono text-sm text-zinc-200 focus:border-[#39FF14]/50 focus:outline-none"
            />
          </div>
          <p className="text-[10px] text-zinc-600">
            Each generation refines top candidates
          </p>
        </div>
      </div>

      {/* Gen 0 Retries */}
      <div className="space-y-2">
        <label className="text-xs font-medium uppercase tracking-wider text-zinc-500">
          Gen 0 Max Attempts
        </label>
        <div className="flex items-center gap-2">
          <input
            type="range"
            min={1} max={10} step={1}
            value={Math.min(gen0Retries, 10)}
            onChange={e => setGen0Retries(Number(e.target.value))}
            className="flex-1 accent-[#39FF14]"
          />
          <input
            type="number"
            min={1}
            value={gen0Retries}
            onChange={e => setGen0Retries(Math.max(1, Number(e.target.value) || 1))}
            className="w-14 rounded border border-zinc-700 bg-zinc-900 px-1.5 py-0.5 text-center font-mono text-sm text-zinc-200 focus:border-[#39FF14]/50 focus:outline-none"
          />
        </div>
        <p className="text-[10px] text-zinc-600">
          Gen 0 keeps retrying with fresh research until a candidate hits ≥50% or attempts run out
        </p>
      </div>

      {/* Est. cost hint */}
      <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 px-3 py-2">
        <p className="text-[11px] text-zinc-500">
          Approx. <span className="font-mono text-zinc-300">{population * (gen0Retries + maxGen - 1)}</span> training runs
          &nbsp;·&nbsp;
          <span className="text-[#39FF14]/70">Free</span> on local CPU
          &nbsp;·&nbsp;
          <span className="text-zinc-400">~{(population * maxGen * 5).toFixed(0)} min</span> estimated
        </p>
      </div>

      {error && (
        <p className="text-xs text-red-400">{error}</p>
      )}

      <button
        type="submit"
        disabled={submitting}
        className="cursor-pointer flex w-full items-center justify-center gap-2 rounded-lg border border-[#39FF14]/40 bg-[#39FF14]/8 px-4 py-3 text-sm font-semibold text-[#39FF14] transition-all hover:border-[#39FF14]/70 hover:bg-[#39FF14]/15 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {submitting
          ? <><Loader2 className="h-4 w-4 animate-spin" /> Launching…</>
          : <><FlaskConical className="h-4 w-4" /> Launch Research Session</>
        }
      </button>
    </form>
  )
}
