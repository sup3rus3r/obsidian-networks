'use client'

import { useState } from 'react'
import Image from 'next/image'
import { Maximize2, ImageIcon } from 'lucide-react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { AppRoutes } from '@/app/api/routes'

function PlotThumbnail({
  sessionId,
  filename,
}: {
  sessionId: string
  filename : string
}) {
  const [open, setOpen]     = useState(false)
  const [error, setError]   = useState(false)
  const src = AppRoutes.ViewImage(sessionId, filename)
  const label = filename.replace(/\.[^.]+$/, '').replace(/_/g, ' ')

  return (
    <>
      {/* Thumbnail card */}
      <button
        onClick={() => setOpen(true)}
        className="group relative flex flex-col overflow-hidden rounded-lg border border-zinc-800 bg-zinc-900/60 transition-colors hover:border-[#39FF14]/40 hover:bg-zinc-900 w-full text-left"
      >
        {/* Image area */}
        <div className="relative h-28 w-full bg-zinc-950 flex items-center justify-center overflow-hidden">
          {error ? (
            <ImageIcon className="h-8 w-8 text-zinc-700" />
          ) : (
            <Image
              src={src}
              alt={label}
              fill
              className="object-contain p-1"
              onError={() => setError(true)}
              unoptimized
            />
          )}
          {/* Expand hint on hover */}
          <div className="absolute inset-0 flex items-center justify-center bg-black/50 opacity-0 transition-opacity group-hover:opacity-100">
            <Maximize2 className="h-5 w-5 text-white" />
          </div>
        </div>

        {/* Label */}
        <div className="px-3 py-2">
          <p className="truncate text-xs font-medium text-zinc-300 capitalize">{label}</p>
          <p className="truncate text-[10px] text-zinc-600">{filename}</p>
        </div>
      </button>

      {/* Full-size Dialog */}
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="max-w-3xl border-zinc-800 bg-zinc-950 p-4">
          <DialogHeader className="pb-2">
            <DialogTitle className="text-sm font-medium text-zinc-200 capitalize">
              {label}
            </DialogTitle>
          </DialogHeader>
          <div className="relative w-full" style={{ aspectRatio: '16/9' }}>
            {error ? (
              <div className="flex h-full items-center justify-center">
                <ImageIcon className="h-12 w-12 text-zinc-700" />
              </div>
            ) : (
              <Image
                src={src}
                alt={label}
                fill
                className="object-contain rounded"
                unoptimized
              />
            )}
          </div>
        </DialogContent>
      </Dialog>
    </>
  )
}

export function PlotGallery({
  sessionId,
  images,
}: {
  sessionId: string
  images   : string[]
}) {
  if (images.length === 0) return null

  return (
    <div className="space-y-2">
      <p className="text-[10px] font-medium uppercase tracking-wider text-zinc-600">Plots</p>
      <div className="grid grid-cols-2 gap-2">
        {images.map(filename => (
          <PlotThumbnail key={filename} sessionId={sessionId} filename={filename} />
        ))}
      </div>
    </div>
  )
}
