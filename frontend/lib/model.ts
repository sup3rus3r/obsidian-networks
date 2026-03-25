/**
 * Model provider resolver.
 *
 * Controlled entirely via environment variables — no UI required.
 *
 * AI_PROVIDER   anthropic | openai | lmstudio | gemini   (default: anthropic)
 * AI_MODEL      model id string                          (default: provider default)
 *
 * Provider-specific keys:
 *   ANTHROPIC_API_KEY          — required for anthropic
 *   OPENAI_API_KEY             — required for openai
 *   GOOGLE_GENERATIVE_AI_API_KEY — required for gemini
 *   LMSTUDIO_BASE_URL          — LM Studio server URL (default: http://localhost:1234/v1)
 *   LMSTUDIO_API_KEY           — optional, ignored by LM Studio but the SDK requires a value
 */

import { anthropic } from '@ai-sdk/anthropic'
import { createGoogleGenerativeAI } from '@ai-sdk/google'
import { createOpenAI } from '@ai-sdk/openai'

export type Provider = 'anthropic' | 'openai' | 'lmstudio' | 'gemini'

const DEFAULTS: Record<Provider, string> = {
  anthropic: 'claude-sonnet-4-6',
  openai    : 'gpt-4o',
  lmstudio  : 'local-model',
  gemini    : 'gemini-2.5-flash',
}

export function getProvider(): Provider {
  return (process.env.AI_PROVIDER ?? 'anthropic') as Provider
}

export function getModel() {
  const provider = getProvider()
  const modelId  = process.env.AI_MODEL || DEFAULTS[provider] || DEFAULTS.anthropic

  switch (provider) {
    case 'openai': {
      const openai = createOpenAI({
        apiKey: process.env.OPENAI_API_KEY,
      })
      return openai(modelId)
    }

    case 'lmstudio': {
      const lmstudio = createOpenAI({
        baseURL: process.env.LMSTUDIO_BASE_URL ?? 'http://localhost:1234/v1',
        // LM Studio doesn't validate the key, but the SDK requires a non-empty value
        apiKey : process.env.LMSTUDIO_API_KEY ?? 'lm-studio',
      })
      // LM Studio only supports v1/chat/completions, not the newer v1/responses API.
      // extractReasoningMiddleware in route.ts handles <think> tags from reasoning models.
      return lmstudio.chat(modelId)
    }

    case 'gemini': {
      const google = createGoogleGenerativeAI({
        apiKey: process.env.GOOGLE_GENERATIVE_AI_API_KEY,
      })
      return google(modelId)
    }

    case 'anthropic':
    default: {
      return anthropic(modelId)
    }
  }
}
