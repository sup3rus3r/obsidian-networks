import { NextRequest, NextResponse } from 'next/server'

const apiBase = process.env.INTERNAL_API_URL ?? 'http://localhost:8000'

export async function GET(req: NextRequest) {
  const sessionId = req.nextUrl.searchParams.get('sessionId')
  if (!sessionId) return NextResponse.json({ error: 'Missing sessionId' }, { status: 400 })
  const res = await fetch(`${apiBase}/platform/session/${sessionId}/phase`, { signal: AbortSignal.timeout(5_000) })
  const data = await res.json()
  return NextResponse.json(data, { status: res.status })
}

export async function POST(req: NextRequest) {
  const { sessionId, phase } = await req.json()
  if (!sessionId || !phase) return NextResponse.json({ error: 'Missing sessionId or phase' }, { status: 400 })
  const res = await fetch(`${apiBase}/platform/session/${sessionId}/phase`, {
    method : 'POST',
    headers: { 'Content-Type': 'application/json' },
    body   : JSON.stringify({ phase }),
    signal : AbortSignal.timeout(5_000),
  })
  const data = await res.json()
  return NextResponse.json(data, { status: res.status })
}
