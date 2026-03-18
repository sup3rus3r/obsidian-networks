import { streamText, convertToModelMessages, pruneMessages, tool, stepCountIs, wrapLanguageModel, extractReasoningMiddleware, type UIMessage } from 'ai'
import { z } from 'zod'
import { getModel, getProvider } from '@/lib/model'

export const runtime = 'nodejs'
export const maxDuration = 120

// Anthropic: ~200k tokens; OpenAI: ~128k; LM Studio: varies (often 8–32k)
const MAX_HISTORY_MSGS: Record<string, number> = {
  anthropic: 40,
  openai   : 30,
  lmstudio : 16,
}

const FETCH_TIMEOUT     = 10_000
const PDF_FETCH_TIMEOUT = 45_000

async function fetchText(url: string): Promise<string> {
  const res = await fetch(url, {
    headers: { 'User-Agent': 'Mozilla/5.0 (compatible; obsidian-networks-research/1.0)' },
    signal : AbortSignal.timeout(FETCH_TIMEOUT),
  })
  return res.text()
}

async function fetchPdfText(url: string): Promise<string | null> {
  try {
    const res = await fetch(url, {
      headers: { 'User-Agent': 'Mozilla/5.0 (compatible; obsidian-networks-research/1.0)' },
      signal : AbortSignal.timeout(PDF_FETCH_TIMEOUT),
    })
    if (!res.ok) return null
    const buf = Buffer.from(await res.arrayBuffer())
    // Import the inner module directly — pdf-parse/index.js runs a self-test on
    // load that requires a fixture file, which crashes in Next.js builds.
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const pdfParse: (buf: Buffer) => Promise<{ text: string }> = require('pdf-parse/lib/pdf-parse.js')
    const result = await pdfParse(buf)
    return result.text ?? null
  } catch {
    return null
  }
}

function htmlToText(html: string): string {
  return html
    .replace(/<script[\s\S]*?<\/script>/gi, '')
    .replace(/<style[\s\S]*?<\/style>/gi, '')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/\s+/g, ' ')
    .trim()
}

function wordTruncate(text: string, max: number): string {
  const words = text.split(/\s+/)
  return words.length > max ? words.slice(0, max).join(' ') + ' …' : text
}

interface SearchResult {
  title  : string
  url    : string
  snippet: string
}

async function searchDDG(query: string, maxResults = 3): Promise<SearchResult[]> {
  const html = await fetchText(
    `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`
  )

  const results: SearchResult[] = []

  const titleRe   = /<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/g
  const snippetRe = /<[^>]+class="result__snippet"[^>]*>([\s\S]*?)<\/(?:a|div|span)>/g

  const titles   = [...html.matchAll(titleRe)]
  const snippets = [...html.matchAll(snippetRe)]

  for (let i = 0; i < Math.min(titles.length, snippets.length, maxResults); i++) {
    const rawHref = titles[i][1]
    let url = rawHref
    const uddg = rawHref.match(/[?&]uddg=([^&]+)/)
    if (uddg) url = decodeURIComponent(uddg[1])

    results.push({
      title  : htmlToText(titles[i][2]),
      url,
      snippet: htmlToText(snippets[i][1]),
    })
  }

  return results
}

interface Paper {
  title   : string
  authors : string
  abstract: string
  url     : string
}

function parseArxiv(xml: string): Paper[] {
  const papers: Paper[] = []
  for (const m of xml.matchAll(/<entry>([\s\S]*?)<\/entry>/g)) {
    const e   = m[1]
    const url = e.match(/<id>(https?:\/\/arxiv\.org\/abs\/[^<]+)<\/id>/)?.[1]
    if (!url) continue
    papers.push({
      title   : (e.match(/<title>([\s\S]*?)<\/title>/)?.[1]   ?? '').trim().replace(/\s+/g, ' '),
      authors : [...e.matchAll(/<name>([\s\S]*?)<\/name>/g)]
                  .slice(0, 3).map(a => a[1].trim()).join(', '),
      abstract: (e.match(/<summary>([\s\S]*?)<\/summary>/)?.[1] ?? '').trim().replace(/\s+/g, ' ').slice(0, 2000),
      url,
    })
  }
  return papers
}

const ALLOWED_DOMAINS = [
  'tensorflow.org',
  'keras.io',
  'arxiv.org',
  'paperswithcode.com',
  'huggingface.co',
]

function isAllowedUrl(raw: string): boolean {
  try {
    const { hostname } = new URL(raw)
    return ALLOWED_DOMAINS.some(d => hostname === d || hostname.endsWith('.' + d))
  } catch {
    return false
  }
}

function createResearchTools(sessionId: string | null) {
  const apiBase = process.env.INTERNAL_API_URL ?? 'http://localhost:8000'

  return {
    search_tensorflow_docs: tool({
      description:
        'Search the TensorFlow and Keras documentation for API references, guides, and tutorials. ' +
        'Always call this before writing any model architecture or preprocessing code to verify ' +
        'current Keras 3 API signatures and avoid deprecated patterns.',
      inputSchema: z.object({
        query: z.string().describe(
          'Search query, e.g. "Conv2D layer parameters keras 3" or "EarlyStopping callback"'
        ),
      }),
      execute: async (input: { query: string }) => {
        try {
          const results = await searchDDG(
            `site:keras.io OR site:tensorflow.org/api_docs ${input.query}`,
            3
          )
          let topContent: string | null = null
          for (const r of results) {
            if (isAllowedUrl(r.url)) {
              try {
                const raw = await fetchText(r.url)
                if (raw) { topContent = wordTruncate(htmlToText(raw), 4000); break }
              } catch { /* skip */ }
            }
          }
          return { query: input.query, results, topContent }
        } catch (err) {
          return { query: input.query, results: [] as SearchResult[], topContent: null, error: String(err) }
        }
      },
    }),

    fetch_arxiv_papers: tool({
      description:
        'Fetch recent research papers from arXiv for a given ML problem domain. ' +
        'Call this when the user describes a specific task to ground recommendations in recent literature.',
      inputSchema: z.object({
        query      : z.string().describe('arXiv search terms, e.g. "tabular classification neural network 2024"'),
        max_results: z.number().int().min(1).max(5).default(3),
      }),
      execute: async (input: { query: string; max_results: number }) => {
        try {
          const xml = await fetchText(
            `https://export.arxiv.org/api/query` +
            `?search_query=all:${encodeURIComponent(input.query)}` +
            `&start=0&max_results=${input.max_results}` +
            `&sortBy=relevance&sortOrder=descending`
          )
          const papers = parseArxiv(xml)
          return { query: input.query, papers }
        } catch (err) {
          return { query: input.query, papers: [] as Paper[], error: String(err) }
        }
      },
    }),

    fetch_url: tool({
      description:
        'Fetch and read the full plain-text content of a specific paper or documentation page. ' +
        'Allowed domains: tensorflow.org, keras.io, arxiv.org, paperswithcode.com, huggingface.co. ' +
        'For arxiv URLs, automatically tries the full HTML version (/html/) before falling back to PDF then abstract. ' +
        'After calling fetch_url, ALWAYS call ingest_url with the same URL to index it into the vector store.',
      inputSchema: z.object({
        url: z.string().url().describe('The URL to fetch and read'),
      }),
      execute: async (input: { url: string }) => {
        if (!isAllowedUrl(input.url)) {
          return { error: `Domain not allowed. Permitted: ${ALLOWED_DOMAINS.join(', ')}` }
        }
        const isArxivAbs = /^https?:\/\/arxiv\.org\/abs\//.test(input.url)

        async function tryFetch(url: string): Promise<string | null> {
          try {
            const text = await fetchText(url)
            if (text.includes('No HTML for') || text.includes('HTML is not available')) return null
            return text
          } catch { return null }
        }

        let fetchedUrl = input.url
        let raw: string | null = null

        if (isArxivAbs) {
          const htmlUrl = input.url.replace('/abs/', '/html/')
          raw = await tryFetch(htmlUrl)
          if (raw) {
            fetchedUrl = htmlUrl
          } else {
            const pdfUrl = input.url.replace('/abs/', '/pdf/')
            const pdfText = await fetchPdfText(pdfUrl)
            if (pdfText) {
              return { url: pdfUrl, content: wordTruncate(pdfText.replace(/\s+/g, ' ').trim(), 12000), source: 'pdf' }
            }
            raw = await tryFetch(input.url)
            fetchedUrl = input.url
          }
        } else {
          raw = await tryFetch(input.url)
        }

        if (!raw) return { url: fetchedUrl, error: 'Could not fetch page.' }
        return { url: fetchedUrl, content: wordTruncate(htmlToText(raw), 12000), source: 'html' }
      },
    }),

    ingest_url: tool({
      description:
        'Index a fetched URL into the session vector store. ' +
        'MUST be called after every fetch_url call with the same URL and the paper title. ' +
        'The backend downloads the full document independently (no truncation) and embeds it into FAISS. ' +
        'This is what powers query_research in the planning phase — skip it and planning has no data.',
      inputSchema: z.object({
        url  : z.string().url().describe('The URL that was just fetched with fetch_url'),
        title: z.string().describe('The paper or page title'),
      }),
      execute: async (input: { url: string; title: string }) => {
        if (!sessionId) return { error: 'No active session' }
        try {
          const res = await fetch(`${apiBase}/platform/vectorstore/${sessionId}/ingest`, {
            method : 'POST',
            headers: { 'Content-Type': 'application/json' },
            body   : JSON.stringify(input),
            signal : AbortSignal.timeout(90_000),
          })
          return await res.json()
        } catch (e) {
          return { error: String(e) }
        }
      },
    }),

    finalize_research: tool({
      description:
        'Signal that research is complete and transition to the PLANNING phase. ' +
        'Call ONLY after all fetch_url + ingest_url calls are done (minimum 3 papers indexed). ' +
        'After this call, query_research and produce_plan become available.',
      inputSchema: z.object({}),
      execute: async () => {
        if (!sessionId) return { error: 'No active session' }
        try {
          const res = await fetch(`${apiBase}/platform/session/${sessionId}/phase`, {
            method : 'POST',
            headers: { 'Content-Type': 'application/json' },
            body   : JSON.stringify({ phase: 'planning' }),
            signal : AbortSignal.timeout(10_000),
          })
          return await res.json()
        } catch (e) {
          return { error: String(e) }
        }
      },
    }),
  }
}

function createPlanningTools(sessionId: string | null) {
  const apiBase = process.env.INTERNAL_API_URL ?? 'http://localhost:8000'

  return {
    query_research: tool({
      description:
        'Query the session vector store to retrieve relevant chunks from ingested papers and docs. ' +
        'Use during planning to ground every architecture decision in specific retrieved evidence. ' +
        'Call at least 6 times with different queries before writing the plan. ' +
        'Returns top-k chunks with source URL and relevance score.',
      inputSchema: z.object({
        query: z.string().describe('The question to search for, e.g. "optimal learning rate tabular data"'),
        k    : z.number().int().min(1).max(12).default(6),
      }),
      execute: async (input: { query: string; k: number }) => {
        if (!sessionId) return { error: 'No active session' }
        try {
          const res = await fetch(`${apiBase}/platform/vectorstore/${sessionId}/query`, {
            method : 'POST',
            headers: { 'Content-Type': 'application/json' },
            body   : JSON.stringify(input),
            signal : AbortSignal.timeout(15_000),
          })
          return await res.json()
        } catch (e) {
          return { error: String(e) }
        }
      },
    }),

    produce_plan: tool({
      description:
        'Submit the structured plan document to the user for approval. ' +
        'Write the complete markdown plan using the <plan_template> structure from the system prompt. ' +
        'Every architecture decision and hyperparameter MUST cite a source URL from retrieved chunks. ' +
        'After calling this, STOP and wait for user approval. Do NOT call edit_script or create_notebook.',
      inputSchema: z.object({
        plan_markdown: z.string().describe('Full markdown plan document following the plan_template structure'),
      }),
      execute: async (input: { plan_markdown: string }) => {
        if (!sessionId) return { error: 'No active session' }
        try {
          const res = await fetch(`${apiBase}/platform/session/${sessionId}/phase`, {
            method : 'POST',
            headers: { 'Content-Type': 'application/json' },
            body   : JSON.stringify({ phase: 'planning', plan_doc: input.plan_markdown }),
            signal : AbortSignal.timeout(10_000),
          })
          return await res.json()
        } catch (e) {
          return { error: String(e) }
        }
      },
    }),

    approve_plan: tool({
      description:
        'Call this when the user approves the plan (says "looks good", "approved", "proceed", "go ahead", etc.). ' +
        'Transitions the session to approved state and unlocks edit_script, run_code, and create_notebook.',
      inputSchema: z.object({}),
      execute: async () => {
        if (!sessionId) return { error: 'No active session' }
        try {
          const res = await fetch(`${apiBase}/platform/session/${sessionId}/phase`, {
            method : 'POST',
            headers: { 'Content-Type': 'application/json' },
            body   : JSON.stringify({ phase: 'approved' }),
            signal : AbortSignal.timeout(10_000),
          })
          return await res.json()
        } catch (e) {
          return { error: String(e) }
        }
      },
    }),
  }
}

function createScriptTools(sessionId: string | null) {
  const apiBase = process.env.INTERNAL_API_URL ?? 'http://localhost:8000'

  const run_code = tool({
    description:
      'Execute a Python snippet in the session working directory and return the output. ' +
      'Use this to test data loading, inspect column names/shapes/dtypes, and validate ' +
      'logic before writing the full script. Has access to dataset.csv and all installed packages. ' +
      '30-second timeout — not for full training runs.',
    inputSchema: z.object({
      code: z.string().describe('Python snippet to execute'),
    }),
    execute: async (input: { code: string }) => {
      if (!sessionId) return { stdout: 'No active session.', exit_code: 1 }
      try {
        const res  = await fetch(`${apiBase}/platform/run_code/${sessionId}`, {
          method : 'POST',
          headers: { 'Content-Type': 'application/json' },
          body   : JSON.stringify({ code: input.code }),
          signal : AbortSignal.timeout(35_000),
        })
        return await res.json()
      } catch (e) {
        return { stdout: String(e), exit_code: 1 }
      }
    },
  })

  const read_script = tool({
    description:
      'Read the current generated_script.py with line numbers. ' +
      'Use this after a compile failure to inspect the exact code before editing.',
    inputSchema: z.object({}),
    execute: async () => {
      if (!sessionId) return { error: 'No active session.' }
      try {
        const res = await fetch(`${apiBase}/platform/script/${sessionId}`, {
          signal: AbortSignal.timeout(10_000),
        })
        return await res.json()
      } catch (e) {
        return { error: String(e) }
      }
    },
  })

  const edit_script = tool({
    description:
      'Replace an exact string in the current script — like a str-replace editor. ' +
      'Use this to fix specific errors after read_script + run_code identify the problem. ' +
      'old_str must match exactly once including whitespace and indentation. ' +
      'Special: set old_str to "__REPLACE_ALL__" to replace the entire script with new_str ' +
      '(use this when there are multiple errors or an indentation/syntax error that is hard to target precisely).',
    inputSchema: z.object({
      old_str: z.string().describe('Exact text to replace (must be unique), or "__REPLACE_ALL__" to replace the entire script'),
      new_str: z.string().describe('Replacement text'),
    }),
    execute: async (input: { old_str: string; new_str: string }) => {
      if (!sessionId) return { ok: false, error: 'No active session.' }
      try {
        const res = await fetch(`${apiBase}/platform/edit_script/${sessionId}`, {
          method : 'POST',
          headers: { 'Content-Type': 'application/json' },
          body   : JSON.stringify(input),
          signal : AbortSignal.timeout(10_000),
        })
        return await res.json()
      } catch (e) {
        return { ok: false, error: String(e) }
      }
    },
  })

  return { run_code, read_script, edit_script }
}

function createNotebookTool(sessionId: string | null) {
  const apiBase = process.env.INTERNAL_API_URL ?? 'http://localhost:8000'

  // Per-request retry counter — resets each time the route is called (i.e. each user turn).
  // Caps at 3 attempts to prevent infinite fix-and-retry loops.
  let attempts    = 0
  let lastError   = ''
  const MAX_ATTEMPTS = 3

  return tool({
    description:
      'Validate and save the current script as a downloadable Jupyter notebook (.ipynb). ' +
      'IMPORTANT: Do NOT pass the script here. First write the script using ' +
      'edit_script(old_str="__REPLACE_ALL__", new_str=<full script>), then call create_notebook ' +
      'with only a description. The backend reads the already-saved script automatically. ' +
      'The notebook will appear in the Downloads panel for the user to download.',
    inputSchema: z.object({
      description: z.string().describe(
        'One-line title for the notebook, e.g. "House Price Regression — Wide & Deep"'
      ),
    }),
    execute: async (input: { description: string }) => {
      if (!sessionId) return { error: 'No active session — cannot save notebook' }

      attempts++

      if (attempts > MAX_ATTEMPTS) {
        return {
          error : `HARD STOP — create_notebook has failed ${MAX_ATTEMPTS} times in a row.`,
          action: 'Do NOT call create_notebook again. Apologise to the user and ask them to describe a simpler architecture so you can start fresh.',
        }
      }

      try {
        const res = await fetch(`${apiBase}/platform/notebook/${sessionId}`, {
          method : 'POST',
          headers: { 'Content-Type': 'application/json' },
          body   : JSON.stringify({ description: input.description }),
          signal : AbortSignal.timeout(15_000),
        })
        if (!res.ok) {
          const body = await res.json().catch(() => null)
          if (res.status === 422 && body?.detail?.errors) {
            const errors      = body.detail.errors as string[]
            const errorKey    = errors.join('|')
            const repeated    = errorKey === lastError
            lastError         = errorKey
            const attemptsLeft = MAX_ATTEMPTS - attempts
            const isSyntax    = errors.some(e => e.startsWith('SyntaxError'))

            // If same error repeated, or it's a syntax error — targeted edits won't help.
            // Force a full rewrite.
            const action = attemptsLeft <= 0
              ? 'HARD STOP — no attempts remaining. Tell the user to describe a simpler approach.'
              : (repeated || isSyntax)
                ? `MANDATORY: Use edit_script(old_str="__REPLACE_ALL__", new_str=<complete corrected script>) to rewrite the entire script from scratch. Targeted edits cannot fix this. Then call create_notebook again (${attemptsLeft} attempt(s) remaining).`
                : `Fix ALL listed errors with edit_script, then call create_notebook again (${attemptsLeft} attempt(s) remaining).`

            return { errors, action }
          }
          const text = body ? JSON.stringify(body) : res.statusText
          return { error: `Backend error ${res.status}: ${text}` }
        }
        return { ok: true, message: 'Notebook saved. The user can now download it from the Downloads panel.' }
      } catch (err) {
        return { error: String(err) }
      }
    },
  })
}

// ── Base system prompt (role, format, code constraints) ──────────────────────
// Phase-specific behaviour is injected at runtime via buildSystemPrompt().
const SYSTEM_BASE = `\
<role>
You are Obsidian Networks — an expert ML research engineer specialising in TensorFlow/Keras.
Your purpose is to help users design, research, and generate production-ready deep learning models from their datasets.
You operate in three strict phases: RESEARCH → PLAN → BUILD.
You NEVER skip phases. You NEVER write code before the plan is approved.
</role>

<format>
- NEVER show the training script in your chat reply — it goes ONLY into create_notebook
- Chat replies must be concise — bullet points, not paragraphs
- Include sources as markdown links under a "References" heading
- Structure scripts internally with section comments: # ── Section Name ──────────────
- Inline script comments must explain *why* an architectural decision was made, citing the approved plan
- Keep conversational replies short and direct
</format>

<constraints>
- The dataset is ALWAYS available as "dataset.csv" (or "dataset.json" for JSON uploads). NEVER use the original uploaded filename.
- All model output files (.keras, .h5) MUST be saved inside the "output/" subdirectory
- If your script creates a derived dataset a later step must read back, save it to "output/filename.csv". NEVER use a bare filename for derived files — the platform rewrites bare filenames to "dataset.csv".
- DO NOT write any matplotlib/seaborn plot code or plt.savefig calls — the platform auto-generates canonical diagnostic plots
- Always start scripts with "import tensorflow" and access Keras as tensorflow.keras. NEVER use standalone "import keras" or bare "keras.X" references.
- Use the Functional API for all models — no Sequential for anything non-trivial
- CRITICAL — ALL tabular features are numeric by the time the model sees them. The worker automatically encodes every non-numeric column to float32 integer codes.
- THEREFORE: Use a SINGLE keras.Input of shape (n_features,) and a SINGLE keras.layers.Normalization layer. NEVER use Embedding layers, StringLookup, or separate categorical/numerical branches for tabular data.
- The feature matrix is always: X = df[feature_cols].to_numpy(dtype='float32')
- CRITICAL — Normalization layer: ALWAYS call normalizer.adapt(X_train) on the TRAINING split only BEFORE building the model. Never adapt on the full dataset (leakage).
- CRITICAL — NaN safety: ALWAYS add after loading data: df = df.replace([np.inf, -np.inf], np.nan).dropna()
- Every supervised script must include EarlyStopping(patience=20, restore_best_weights=True) + ModelCheckpoint
- Always set epochs=200 in model.fit() — EarlyStopping will cut it short. NEVER use epochs=1 or any low value.
- Never use deprecated Keras 2 APIs
- CRITICAL — residual/skip connections with layers.Add() REQUIRE matching shapes. Always project the shortcut with a Dense layer of matching units before Add().
- For time series: use keras.utils.timeseries_dataset_from_array() — NEVER manually roll windows
- For RL: custom training loop with env.step() / env.reset() — do NOT use model.fit()
</constraints>

<plan_template>
When writing the plan document, use exactly this structure:

# ML Plan: [Task Name]

## 1. Problem Type & Task Framing
[Binary/multi-class/regression/time-series/RL — why, based on dataset analysis]

## 2. Feature Engineering
[Column-by-column: what transformations, why, which columns are dropped/combined. Source: URL]

## 3. Architecture
### Selected Model: [Name]
**Justification** (cite retrieved chunks):
- Input → [shape] — why
- Layer 1: Dense([N], activation='[X]') — why N units, why X activation [source: URL]
- [continue for each layer]
- Output: Dense([K], activation='[Y]') — why Y matches the task

## 4. Hyperparameters
| Parameter | Value | Justification | Source |
|---|---|---|---|
| learning_rate | ... | ... | [URL] |
| batch_size | ... | ... | [URL] |
| dropout | ... | ... | [URL] |

## 5. Training Strategy
- Optimizer: [name + config, with source citation]
- LR schedule: [reason]
- Early stopping: patience=20, restore_best_weights=True
- Epochs: 200 (capped by EarlyStopping)
- Train/val split: [ratio, reason]

## 6. Expected Outputs & Evaluation
- Primary metric: [AUC/RMSE/accuracy — why]
- Output file: output/model.keras
</plan_template>`

// ── Phase-aware system prompt builder ────────────────────────────────────────
function buildSystemPrompt(phase: string, planDoc: string | null): string {
  if (phase === 'idle' || phase === 'researching') {
    return SYSTEM_BASE + `

<phase current="${phase}">
CURRENT PHASE: RESEARCH

Available tools: fetch_arxiv_papers, fetch_url, search_tensorflow_docs, ingest_url, finalize_research
LOCKED tools (not available yet): query_research, produce_plan, approve_plan, run_code, edit_script, create_notebook

BEHAVIOUR:
1. DATASET JUST UPLOADED, NO CLEAR GOAL YET?
   → Do NOT run tools. Greet the dataset (1 sentence). Ask ONE open-ended question about the goal.
   → Do NOT name any architectures.

2. USER HAS STATED A CLEAR GOAL?
   → STEP 1: Call fetch_arxiv_papers with a domain-specific query (e.g. "tabular regression deep learning 2024")
   → STEP 2: Call fetch_arxiv_papers AGAIN with a different angle (e.g. "neural network benchmark tabular data 2024")
   → STEP 3: Call fetch_url on the TOP 3 most relevant paper URLs IN PARALLEL to read full text (methods, architecture, hyperparameters)
   → STEP 4: For EACH fetch_url call, immediately call ingest_url with the same URL and title to index it into the vector store
   → STEP 5: Call search_tensorflow_docs to verify the Keras 3 API for the top candidate architecture
   → STEP 6: Call finalize_research to transition to the PLAN phase
   → Do NOT form any architectural opinion yet. Do NOT describe what you found. Just move to planning.

NEVER skip ingest_url after fetch_url — the vector store must be populated before planning.
NEVER call finalize_research before fetching and ingesting at least 3 paper URLs.
NEVER ask more than one question at a time.
</phase>`
  }

  if (phase === 'planning') {
    return SYSTEM_BASE + `

<phase current="planning">
CURRENT PHASE: PLANNING

Available tools: query_research, produce_plan
LOCKED tools (not available yet): fetch_arxiv_papers, fetch_url, ingest_url, run_code, edit_script, create_notebook

BEHAVIOUR:
1. Query the vector store at least 6 times with different targeted questions before writing the plan:
   - "best architecture for [task type] from literature"
   - "optimal layer sizes and depth for [task type]"
   - "activation functions for [task type] output layer"
   - "optimizer learning rate schedule [task type]"
   - "regularization dropout rates [task type]"
   - "evaluation metrics [task type]"
   Query with different phrasings to maximise coverage.

2. Synthesise ALL retrieved chunks into the plan document using the <plan_template> structure above.
   - Every architecture decision MUST cite a specific source URL from the retrieved chunks.
   - Every hyperparameter value MUST cite a source URL.
   - Do NOT invent values from training knowledge — ground EVERYTHING in retrieved evidence.

3. Call produce_plan with the complete markdown plan.

4. STOP. Write exactly: "Here is the proposed plan based on the research. Let me know if you'd like any changes before I start building."

5. Do NOT call edit_script, create_notebook, or run_code under any circumstances in this phase.
</phase>`
  }

  if (phase === 'approved' || phase === 'building') {
    const planBlock = planDoc
      ? `\n<approved_plan>\n${planDoc}\n</approved_plan>`
      : ''
    return SYSTEM_BASE + `

<phase current="${phase}">
CURRENT PHASE: BUILD
${planBlock}

BEHAVIOUR:
1. If the user just approved the plan (says "looks good", "proceed", "approved", "go ahead", etc.):
   → Call approve_plan first to unlock code generation tools.

2. BUILD FLOW (after approve_plan or if already in building phase):
   → STEP 1: Call run_code to inspect the dataset:
       run_code("import pandas as pd; df = pd.read_csv('dataset.csv'); print(df.shape); print(df.dtypes); print(df.head(2))")
   → STEP 2: Write the complete script via edit_script(old_str="__REPLACE_ALL__", new_str=<full script>)
       - EVERY architecture decision in code must trace back to the approved plan above
       - EVERY in-code comment must reference the plan section or paper source
   → STEP 3: Call create_notebook with ONLY a description (no script argument)
   → STEP 4: If create_notebook returns validation errors, use read_script + edit_script to fix, then retry
   → STEP 5: After create_notebook succeeds, reply with 3–5 bullet points: architecture, key hyperparameters, expected output. No code shown.

3. USER ASKS TO CHANGE/IMPROVE THE MODEL?
   → Acknowledge the change, call run_code / read_script, apply targeted edit_script changes
   → Call create_notebook again with updated description
   → Reply with 2–3 sentences describing what changed

4. create_notebook RETURNS ERRORS?
   → Fix ALL listed errors immediately without asking the user. Retry until ok: true.
   → On HARD STOP: apologise, ask user to describe a simpler architecture.

NEVER deviate from the approved plan's architecture without explicit user instruction.
NEVER call approve_plan more than once.
</phase>`
  }

  // Fallback — should not normally be reached
  return SYSTEM_BASE
}

export async function POST(req: Request) {
  const { messages, sessionId }: { messages: UIMessage[]; sessionId: string | null } =
    await req.json()

  const provider   = getProvider()
  const maxHistory = MAX_HISTORY_MSGS[provider] ?? 40
  const apiBase    = process.env.INTERNAL_API_URL ?? 'http://localhost:8000'

  // Fetch current session phase from backend
  let sessionPhase = 'idle'
  let sessionPlan: string | null = null
  if (sessionId) {
    try {
      const phaseRes = await fetch(`${apiBase}/platform/session/${sessionId}/phase`, {
        signal: AbortSignal.timeout(5_000),
      })
      if (phaseRes.ok) {
        const pd  = await phaseRes.json()
        sessionPhase = pd.phase   ?? 'idle'
        sessionPlan  = pd.plan_doc ?? null
      }
    } catch { /* non-fatal — default to idle */ }
  }

  // Keep last N message turns to stay within context window limits
  const rawMessages = await convertToModelMessages(messages.slice(-maxHistory))

  // For OpenAI / LM Studio: prune old tool call results from history to keep
  // context lean (arXiv abstracts and TF doc fetches are large).
  // For Anthropic: use server-side contextManagement instead (see providerOptions).
  const prunedMessages = provider !== 'anthropic'
    ? pruneMessages({ messages: rawMessages, toolCalls: 'before-last-5-messages' })
    : rawMessages

  // LM Studio: strip all tool call/result messages from history — LM Studio struggles
  // to handle tool message history across turns and rejects subsequent requests.
  // Keep only user and assistant (text-only) messages; tool exchanges are ephemeral.
  const modelMessages = provider === 'lmstudio'
    ? prunedMessages
        .filter(msg => msg.role === 'user' || msg.role === 'assistant')
        .map(msg => {
          // Flatten array content to plain string
          if (!Array.isArray(msg.content)) return msg
          const text = (msg.content as Array<{ type: string; text?: string }>)
            .filter(p => p.type === 'text')
            .map(p => p.text ?? '')
            .join('')
          return { ...msg, content: text } as typeof msg
        })
        .filter(msg => typeof msg.content === 'string' && msg.content.trim() !== '')
    : prunedMessages

  // Phase-gated tool set — script tools only available once plan is approved
  const isBuilding = sessionPhase === 'approved' || sessionPhase === 'building'
  const phaseTools = {
    ...createResearchTools(sessionId),
    ...createPlanningTools(sessionId),
    ...(isBuilding ? { ...createScriptTools(sessionId), create_notebook: createNotebookTool(sessionId) } : {}),
  }

  // Wrap all models with reasoning middleware — extracts <think>...</think> blocks
  // from the text stream into proper reasoning parts for the ThinkingBlock UI.
  // Anthropic/OpenAI native reasoning is handled separately but the middleware
  // is harmless when no <think> tags are present.
  const model = wrapLanguageModel({
    model     : getModel(),
    middleware: extractReasoningMiddleware({ tagName: 'think' }),
  })

  const result = streamText({
    model          : model,
    system         : buildSystemPrompt(sessionPhase, sessionPlan),
    messages       : modelMessages,
    tools          : phaseTools,
    stopWhen       : stepCountIs(40),
    // Anthropic: cache the system prompt + auto-clear old tool uses when context grows large.
    // cacheControl marks the system prompt for ephemeral caching (reduces cost + TPM usage).
    // contextManagement clears old tool-use results at 60k tokens, keeping last 3 tool turns.
    providerOptions: provider === 'anthropic' ? {
      anthropic: {
        cacheControl: { type: 'ephemeral' },
        contextManagement: {
          edits: [
            {
              type           : 'clear_tool_uses_20250919',
              trigger        : { type: 'input_tokens', value: 60000 },
              keep           : { type: 'tool_uses', value: 3 },
              clearToolInputs: true,
            },
          ],
        },
      },
    } : undefined,
    onError: ({ error }) => {
      console.error('[streamText error]', error)
    },
  })

  return result.toUIMessageStreamResponse({ sendReasoning: true })
}
