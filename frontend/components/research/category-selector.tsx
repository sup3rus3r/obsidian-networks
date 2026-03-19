'use client'

import { type ResearchCategory } from '@/app/api/platform'
import {
  Eye, Type, Music2, TrendingUp, GitFork,
  Layers, Table2, Star, Sparkles,
} from 'lucide-react'

const CATEGORY_ICONS: Record<string, React.ElementType> = {
  vision                : Eye,
  text                  : Type,
  audio                 : Music2,
  timeseries            : TrendingUp,
  graph                 : GitFork,
  multimodal_text_image : Layers,
  tabular               : Table2,
  recommendation        : Star,
  generative            : Sparkles,
}

const CATEGORY_COLORS: Record<string, string> = {
  vision                : 'border-blue-500/30 bg-blue-500/5 hover:border-blue-500/60 hover:bg-blue-500/10 [&.selected]:border-blue-500 [&.selected]:bg-blue-500/15',
  text                  : 'border-violet-500/30 bg-violet-500/5 hover:border-violet-500/60 hover:bg-violet-500/10 [&.selected]:border-violet-500 [&.selected]:bg-violet-500/15',
  audio                 : 'border-pink-500/30 bg-pink-500/5 hover:border-pink-500/60 hover:bg-pink-500/10 [&.selected]:border-pink-500 [&.selected]:bg-pink-500/15',
  timeseries            : 'border-amber-500/30 bg-amber-500/5 hover:border-amber-500/60 hover:bg-amber-500/10 [&.selected]:border-amber-500 [&.selected]:bg-amber-500/15',
  graph                 : 'border-cyan-500/30 bg-cyan-500/5 hover:border-cyan-500/60 hover:bg-cyan-500/10 [&.selected]:border-cyan-500 [&.selected]:bg-cyan-500/15',
  multimodal_text_image : 'border-orange-500/30 bg-orange-500/5 hover:border-orange-500/60 hover:bg-orange-500/10 [&.selected]:border-orange-500 [&.selected]:bg-orange-500/15',
  tabular               : 'border-[#39FF14]/30 bg-[#39FF14]/5 hover:border-[#39FF14]/60 hover:bg-[#39FF14]/10 [&.selected]:border-[#39FF14] [&.selected]:bg-[#39FF14]/15',
  recommendation        : 'border-rose-500/30 bg-rose-500/5 hover:border-rose-500/60 hover:bg-rose-500/10 [&.selected]:border-rose-500 [&.selected]:bg-rose-500/15',
  generative            : 'border-purple-500/30 bg-purple-500/5 hover:border-purple-500/60 hover:bg-purple-500/10 [&.selected]:border-purple-500 [&.selected]:bg-purple-500/15',
}

const CATEGORY_ICON_COLORS: Record<string, string> = {
  vision                : 'text-blue-400',
  text                  : 'text-violet-400',
  audio                 : 'text-pink-400',
  timeseries            : 'text-amber-400',
  graph                 : 'text-cyan-400',
  multimodal_text_image : 'text-orange-400',
  tabular               : 'text-[#39FF14]',
  recommendation        : 'text-rose-400',
  generative            : 'text-purple-400',
}

// Static fallback if API categories aren't loaded yet
const STATIC_CATEGORIES: ResearchCategory[] = [
  { id: 'vision',                label: 'Vision',         description: 'CNNs, ViT, object detection, segmentation', domains: ['vision'],        default_architectures: ['CNN', 'ViT'] },
  { id: 'text',                  label: 'Text / NLP',     description: 'Transformers, LSTM, text classification',   domains: ['language'],      default_architectures: ['Transformer', 'LSTM'] },
  { id: 'audio',                 label: 'Audio',          description: 'Conformer, CNN-audio, speech, music',       domains: ['audio'],         default_architectures: ['Conformer', 'CNN-Audio'] },
  { id: 'timeseries',            label: 'Time Series',    description: 'LSTM-TS, Transformer-TS, forecasting',      domains: ['timeseries'],    default_architectures: ['LSTM-TS', 'Transformer-TS'] },
  { id: 'graph',                 label: 'Graph',          description: 'GCN, GAT, node/link/graph classification',  domains: ['graph'],         default_architectures: ['GCN', 'GAT'] },
  { id: 'multimodal_text_image', label: 'Multimodal',     description: 'CLIP, Flamingo, cross-modal learning',      domains: ['multimodal'],    default_architectures: ['CLIP', 'Flamingo'] },
  { id: 'tabular',               label: 'Tabular',        description: 'MLP, ResNet-Tabular, structured data',      domains: ['tabular'],       default_architectures: ['MLP', 'ResNet-Tabular'] },
  { id: 'recommendation',        label: 'Recommendation', description: 'Embedding-CF, attention-based rec systems', domains: ['recommendation'],default_architectures: ['Embedding-CF', 'Attention-Rec'] },
  { id: 'generative',            label: 'Generative',     description: 'VAE, GAN, diffusion, image synthesis',      domains: ['generative'],    default_architectures: ['VAE', 'GAN'] },
]

interface CategorySelectorProps {
  categories  : ResearchCategory[]
  selected    : string
  onSelect    : (id: string) => void
}

export function CategorySelector({ categories, selected, onSelect }: CategorySelectorProps) {
  const cats = categories.length > 0 ? categories : STATIC_CATEGORIES

  return (
    <div className="grid grid-cols-3 gap-2">
      {cats.map(cat => {
        const Icon       = CATEGORY_ICONS[cat.id] ?? Table2
        const colorClass = CATEGORY_COLORS[cat.id] ?? CATEGORY_COLORS.tabular
        const iconColor  = CATEGORY_ICON_COLORS[cat.id] ?? 'text-zinc-400'
        const isSelected = selected === cat.id

        return (
          <button
            key={cat.id}
            onClick={() => onSelect(cat.id)}
            className={`selected:border-current cursor-pointer rounded-lg border p-3 text-left transition-all duration-150 ${colorClass} ${isSelected ? 'selected' : ''}`}
          >
            <div className="flex items-start gap-2">
              <Icon className={`mt-0.5 h-4 w-4 shrink-0 ${iconColor}`} />
              <div className="min-w-0">
                <p className="text-xs font-semibold text-zinc-200 leading-tight">{cat.label}</p>
                <p className="mt-0.5 text-[10px] text-zinc-500 leading-snug line-clamp-2">{cat.description}</p>
              </div>
            </div>
          </button>
        )
      })}
    </div>
  )
}
