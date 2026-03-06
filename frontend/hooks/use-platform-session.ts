'use client'

import { useState, useEffect, useCallback } from 'react'
import { createSession } from '@/app/api/platform'

/** Ping the backend to confirm the cookie-bound session is still alive. */
async function verifySession(): Promise<boolean> {
  try {
    const res = await fetch('/api/platform/session', { credentials: 'include' })
    return res.ok
  } catch {
    return false
  }
}

interface SessionState {
  sessionId  : string | null
  expiresAt  : number | null   // Unix timestamp (seconds)
  loading    : boolean
  error      : string | null
}

interface UsePlatformSessionReturn extends SessionState {
  /** Manually request a new session (e.g. after expiry banner dismiss) */
  refresh: () => Promise<void>
  /** Force-create a brand new session, discarding the current one */
  newSession: () => Promise<void>
  /** True if session exists and has not expired */
  isValid: boolean
}

const SESSION_STORAGE_KEY = 'ob_session_id'
const EXPIRES_STORAGE_KEY = 'ob_session_expires'

/**
 * Manages an anonymous platform session.
 * - Creates one on first load (or if stored session is expired).
 * - Persists the session ID in sessionStorage so page refreshes don't create duplicates.
 * - The server also sets an httponly cookie, but we keep the ID client-side
 *   to know which session to reference for API calls.
 */
export function usePlatformSession(): UsePlatformSessionReturn {
  const [state, setState] = useState<SessionState>({
    sessionId : null,
    expiresAt : null,
    loading   : true,
    error     : null,
  })

  const initSession = useCallback(async () => {
    setState(prev => ({ ...prev, loading: true, error: null }))

    // Check existing stored session
    const storedId      = sessionStorage.getItem(SESSION_STORAGE_KEY)
    const storedExpires = sessionStorage.getItem(EXPIRES_STORAGE_KEY)

    if (storedId && storedExpires) {
      const exp = parseInt(storedExpires, 10)
      if (Date.now() / 1000 < exp - 60) {           // still valid locally (60s buffer)
        const alive = await verifySession()          // confirm backend still knows about it
        if (alive) {
          setState({ sessionId: storedId, expiresAt: exp, loading: false, error: null })
          return
        }
        // Backend lost the session (e.g. server restart) — clear stale storage
        sessionStorage.removeItem(SESSION_STORAGE_KEY)
        sessionStorage.removeItem(EXPIRES_STORAGE_KEY)
      }
    }

    // Create a new session
    const data = await createSession()
    if (!data) {
      setState({ sessionId: null, expiresAt: null, loading: false, error: 'Failed to start session' })
      return
    }

    sessionStorage.setItem(SESSION_STORAGE_KEY, data.session_id)
    if ('expires_at' in data) {
      sessionStorage.setItem(EXPIRES_STORAGE_KEY, String((data as { session_id: string; expires_at: number }).expires_at))
    }

    setState({
      sessionId : data.session_id,
      expiresAt : 'expires_at' in data ? (data as { session_id: string; expires_at: number }).expires_at : null,
      loading   : false,
      error     : null,
    })
  }, [])

  useEffect(() => {
    initSession()
  }, [initSession])

  const newSession = useCallback(async () => {
    sessionStorage.removeItem(SESSION_STORAGE_KEY)
    sessionStorage.removeItem(EXPIRES_STORAGE_KEY)
    try { await fetch('/api/platform/purge-queue', { method: 'POST' }) } catch { /* best-effort */ }
    await initSession()
  }, [initSession])

  const isValid =
    state.sessionId !== null &&
    (state.expiresAt === null || Date.now() / 1000 < state.expiresAt - 60)

  return { ...state, refresh: initSession, newSession, isValid }
}
