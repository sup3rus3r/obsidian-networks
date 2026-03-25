import { getProvider } from '@/lib/model'
import { NextResponse } from 'next/server'

// Known vision-capable models
const VISION_CAPABLE: Record<string, string[]> = {
  anthropic: ['claude-opus-4-6', 'claude-sonnet-4-6', 'claude-haiku-4-5'],
  openai    : ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-4-vision-preview'],
  gemini    : ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-3.1-pro-preview', 'gemini-3-flash-preview', 'gemini-3.1-flash-lite-preview'],
  lmstudio  : [], // local models vary — assume not vision-capable
}

export async function GET() {
  const provider = getProvider()
  const modelId  = process.env.AI_MODEL || ''
  const capable  = VISION_CAPABLE[provider] ?? []
  // If no model override, use provider default — anthropic/openai defaults are vision-capable
  const supportsVision = provider === 'lmstudio'
    ? false
    : modelId === '' || capable.some(m => modelId.includes(m))

  return NextResponse.json({ provider, model: modelId || null, supportsVision })
}
