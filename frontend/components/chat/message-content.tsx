'use client'

import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useEffect, useState, useCallback, useMemo, memo } from 'react'
import { Copy, Check } from 'lucide-react'
import { cn } from '@/lib/utils'

// ── Shiki lazy loader ─────────────────────────────────────────────────────────

let highlighter: Awaited<ReturnType<typeof import('shiki').createHighlighter>> | null = null

async function getHighlighter() {
  if (highlighter) return highlighter
  const { createHighlighter } = await import('shiki')
  highlighter = await createHighlighter({
    themes : ['one-dark-pro'],
    langs  : ['python', 'typescript', 'javascript', 'json', 'bash', 'shell', 'text'],
  })
  return highlighter
}

// ── Inline copy hook ──────────────────────────────────────────────────────────

function useCopy(text: string) {
  const [copied, setCopied] = useState(false)
  const copy = useCallback(async () => {
    await navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }, [text])
  return { copied, copy }
}

// ── Code block with copy button ───────────────────────────────────────────────

interface CodeBlockProps {
  language   : string
  code       : string
  isStreaming: boolean
}

const CodeBlock = memo(function CodeBlock({ language, code, isStreaming }: CodeBlockProps) {
  const [html, setHtml] = useState<string | null>(null)
  const { copied, copy } = useCopy(code)

  // Only highlight once streaming stops — prevents per-chunk flicker
  useEffect(() => {
    if (isStreaming) return
    let cancelled = false
    getHighlighter().then(hl => {
      if (cancelled) return
      const lang = hl.getLoadedLanguages().includes(language as never) ? language : 'text'
      setHtml(hl.codeToHtml(code, { lang, theme: 'one-dark-pro' }))
    })
    return () => { cancelled = true }
  }, [code, language, isStreaming])

  // Clear highlight when a new stream starts (e.g. model update request)
  useEffect(() => {
    if (isStreaming) setHtml(null)
  }, [isStreaming])

  return (
    <div className="group/code relative my-2 rounded-lg overflow-hidden border border-zinc-800">
      {/* Header bar: language label + copy button */}
      <div className="flex items-center justify-between bg-zinc-800/80 px-3 py-1.5">
        <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
          {language || 'code'}
        </span>
        <button
          onClick={copy}
          className="cursor-pointer flex items-center gap-1.5 rounded px-1.5 py-0.5 text-[11px] text-zinc-500 transition-colors hover:bg-zinc-700 hover:text-[#39FF14]"
        >
          {copied
            ? <><Check className="h-3 w-3 text-[#39FF14]" /><span className="text-[#39FF14]">Copied!</span></>
            : <><Copy className="h-3 w-3" /><span>Copy</span></>
          }
        </button>
      </div>

      {/* Syntax-highlighted code */}
      {html
        ? (
          <div
            className="overflow-x-auto text-sm [&>pre]:p-4 [&>pre]:overflow-x-auto"
            dangerouslySetInnerHTML={{ __html: html }}
          />
        )
        : (
          <pre className="overflow-x-auto bg-zinc-900 p-4 text-sm font-mono text-zinc-100">
            <code>{code}</code>
          </pre>
        )
      }
    </div>
  )
})

// ── Message content ───────────────────────────────────────────────────────────

interface MessageContentProps {
  content    : string
  isStreaming?: boolean
  className? : string
}

export function MessageContent({ content, isStreaming = false, className }: MessageContentProps) {
  // Memoized so ReactMarkdown doesn't recreate all renderers on every streaming chunk
  const components = useMemo(() => ({
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    code({ className: cls, children, ...props }: any) {
      const match   = /language-(\w+)/.exec(cls ?? '')
      const lang    = match?.[1] ?? 'text'
      const code    = String(children).replace(/\n$/, '')
      const isBlock = code.includes('\n') || match !== null

      if (!isBlock) {
        return (
          <code
            className="rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-xs text-zinc-200"
            {...props}
          >
            {children}
          </code>
        )
      }

      return <CodeBlock language={lang} code={code} isStreaming={isStreaming} />
    },

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    p({ children }: any) {
      return <p className="mb-3 last:mb-0 leading-relaxed">{children}</p>
    },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ul({ children }: any) {
      return <ul className="my-2 list-disc pl-5 space-y-1">{children}</ul>
    },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ol({ children }: any) {
      return <ol className="my-2 list-decimal pl-5 space-y-1">{children}</ol>
    },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    h1({ children }: any) {
      return <h1 className="mb-2 mt-4 text-lg font-semibold">{children}</h1>
    },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    h2({ children }: any) {
      return <h2 className="mb-2 mt-3 text-base font-semibold">{children}</h2>
    },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    h3({ children }: any) {
      return <h3 className="mb-1 mt-2 text-sm font-semibold">{children}</h3>
    },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    table({ children }: any) {
      return (
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-sm">{children}</table>
        </div>
      )
    },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    th({ children }: any) {
      return (
        <th className="border border-zinc-700 bg-zinc-800 px-3 py-1.5 text-left font-medium text-zinc-200">
          {children}
        </th>
      )
    },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    td({ children }: any) {
      return <td className="border border-zinc-700 px-3 py-1.5 text-zinc-300">{children}</td>
    },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    blockquote({ children }: any) {
      return (
        <blockquote className="border-l-2 border-[#39FF14]/40 pl-4 italic text-zinc-400">
          {children}
        </blockquote>
      )
    },
  // isStreaming must be in deps so CodeBlock receives the updated prop when streaming ends
  }), [isStreaming])

  return (
    <div className={cn('prose prose-sm prose-invert max-w-none', className)}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {content}
      </ReactMarkdown>

      {isStreaming && content && (
        <span className="inline-flex items-center gap-0.5 ml-0.5 align-middle">
          <span className="h-1 w-1 rounded-full bg-zinc-500 animate-bounce [animation-delay:0ms]" />
          <span className="h-1 w-1 rounded-full bg-zinc-500 animate-bounce [animation-delay:150ms]" />
          <span className="h-1 w-1 rounded-full bg-zinc-500 animate-bounce [animation-delay:300ms]" />
        </span>
      )}
    </div>
  )
}
