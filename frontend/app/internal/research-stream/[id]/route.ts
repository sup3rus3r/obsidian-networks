import { NextRequest } from 'next/server'

const API_BASE = process.env.INTERNAL_API_URL ?? 'http://localhost:8000'

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params

  let upstream: Response
  try {
    upstream = await fetch(`${API_BASE}/platform/research/${id}/stream`, {
      headers: { Accept: 'text/event-stream', 'Cache-Control': 'no-cache' },
      signal: req.signal,
    })
  } catch (err: unknown) {
    const aborted = err instanceof Error && err.name === 'AbortError'
    return new Response(aborted ? 'Client disconnected' : 'Upstream connection failed', {
      status: aborted ? 499 : 502,
    })
  }

  if (!upstream.ok || !upstream.body) {
    return new Response('Failed to connect to research stream', { status: upstream.status })
  }

  return new Response(upstream.body, {
    headers: {
      'Content-Type'     : 'text/event-stream',
      'Cache-Control'    : 'no-cache, no-transform',
      'X-Accel-Buffering': 'no',
      Connection         : 'keep-alive',
    },
  })
}
