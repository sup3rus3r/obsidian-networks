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

const FETCH_TIMEOUT = 10_000

async function fetchText(url: string): Promise<string> {
  const res = await fetch(url, {
    headers: { 'User-Agent': 'Mozilla/5.0 (compatible; obsidian-networks-research/1.0)' },
    signal : AbortSignal.timeout(FETCH_TIMEOUT),
  })
  return res.text()
}

async function fetchContext7Docs(topic: string): Promise<string> {
  // Fetch from keras.io (primary — highest trust score, most relevant API docs)
  const kerasRes = await fetch(
    `https://context7.com/api/v1/websites/keras_io?tokens=8000&topic=${encodeURIComponent(topic)}`,
    { signal: AbortSignal.timeout(15_000) }
  )
  const kerasText = kerasRes.ok ? await kerasRes.text() : ''

  // Fetch from tensorflow/docs (secondary — covers tf.data, callbacks, training APIs)
  const tfRes = await fetch(
    `https://context7.com/api/v1/tensorflow/docs?tokens=4000&topic=${encodeURIComponent(topic)}`,
    { signal: AbortSignal.timeout(15_000) }
  )
  const tfText = tfRes.ok ? await tfRes.text() : ''

  return [kerasText, tfText].filter(Boolean).join('\n\n---\n\n')
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


function createResearchTools(sessionId: string | null) {
  const apiBase = process.env.INTERNAL_API_URL ?? 'http://localhost:8000'

  return {
    search_arxiv: tool({
      description:
        'Search arXiv for research papers relevant to the ML task. ' +
        'Returns paper titles, authors, abstracts, and arXiv IDs. ' +
        'Read the abstracts carefully and select the most relevant papers. ' +
        'Then call ingest_arxiv_paper for each selected paper to download and index the full PDF. ' +
        'Call this 2 times with different query angles to get broad coverage.',
      inputSchema: z.object({
        query      : z.string().describe('arXiv search terms, e.g. "tabular classification deep learning 2024"'),
        max_results: z.number().int().min(1).max(6).default(5),
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
          // Strip version suffix from IDs so download always gets latest
          const papersWithIds = papers.map(p => ({
            ...p,
            arxiv_id: p.url.split('/abs/')[1]?.replace(/v\d+$/, '') ?? '',
          }))
          return { query: input.query, papers: papersWithIds }
        } catch (err) {
          return { query: input.query, papers: [] as (Paper & { arxiv_id: string })[], error: String(err) }
        }
      },
    }),

    ingest_arxiv_paper: tool({
      description:
        'Download the full PDF of an arXiv paper by its ID and index it into the session vector store. ' +
        'This gives the planning phase access to full paper content: methods, architectures, hyperparameters. ' +
        'Use the arxiv_id returned by search_arxiv (e.g. "2112.02962"). ' +
        'Call this for every paper you select from search_arxiv results.',
      inputSchema: z.object({
        arxiv_id: z.string().describe('arXiv paper ID without version, e.g. "2112.02962"'),
        title   : z.string().describe('Paper title from search_arxiv results'),
      }),
      execute: async (input: { arxiv_id: string; title: string }) => {
        if (!sessionId) return { error: 'No active session' }
        // Strip any version suffix the model may have included
        const cleanId = input.arxiv_id.replace(/v\d+$/, '')
        const pdfUrl  = `https://arxiv.org/pdf/${cleanId}`
        try {
          const res = await fetch(`${apiBase}/platform/vectorstore/${sessionId}/ingest`, {
            method : 'POST',
            headers: { 'Content-Type': 'application/json' },
            body   : JSON.stringify({ url: pdfUrl, title: input.title }),
            signal : AbortSignal.timeout(120_000),
          })
          const data = await res.json()
          return { arxiv_id: cleanId, pdf_url: pdfUrl, ...data }
        } catch (e) {
          return { arxiv_id: cleanId, error: String(e) }
        }
      },
    }),

    fetch_tensorflow_docs: tool({
      description:
        'Fetch authoritative Keras/TensorFlow API documentation from Context7 for a specific topic. ' +
        'Returns real code snippets and API signatures from keras.io and tensorflow/docs. ' +
        'Ingests the result directly into the session vector store for use during planning. ' +
        'Call this for the specific architecture components you plan to use (e.g. "Normalization layer adapt", "EarlyStopping callback", "Dense layer functional API").',
      inputSchema: z.object({
        topic: z.string().describe('Specific Keras/TF topic, e.g. "Normalization layer adapt tabular" or "EarlyStopping ModelCheckpoint callbacks"'),
      }),
      execute: async (input: { topic: string }) => {
        if (!sessionId) return { error: 'No active session' }
        try {
          const docs = await fetchContext7Docs(input.topic)
          if (!docs.trim()) return { error: 'No documentation found for this topic' }

          // Ingest directly into vector store
          const res = await fetch(`${apiBase}/platform/vectorstore/${sessionId}/ingest`, {
            method : 'POST',
            headers: { 'Content-Type': 'application/json' },
            body   : JSON.stringify({
              url  : `https://keras.io/context7/${encodeURIComponent(input.topic)}`,
              title: `Keras/TF Docs: ${input.topic}`,
              text : docs,
            }),
            signal : AbortSignal.timeout(30_000),
          })
          const data = await res.json()
          // Return a short preview so the model knows what was indexed
          return {
            topic  : input.topic,
            preview: docs.slice(0, 800),
            ...data,
          }
        } catch (e) {
          return { topic: input.topic, error: String(e) }
        }
      },
    }),

    finalize_research: tool({
      description:
        'Signal that research is complete and transition to the PLANNING phase. ' +
        'Call ONLY after all ingest_arxiv_paper + fetch_tensorflow_docs calls are done (minimum 3 papers indexed). ' +
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
        'Transitions the session to approved state. ' +
        'After calling this, immediately proceed with the build sequence — do NOT wait for another user message.',
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
          const data = await res.json()
          return {
            ...data,
            next_action: 'Plan approved. Proceed immediately with the BUILD SEQUENCE: STEP 1 (dataset mode only) run_code to inspect dataset, STEP 2 edit_script to write the full training script, STEP 3 create_notebook. Do not pause or ask the user anything.',
          }
        } catch (e) {
          return { error: String(e) }
        }
      },
    }),
  }
}

function createScriptTools(sessionId: string | null) {
  const apiBase = process.env.INTERNAL_API_URL ?? 'http://localhost:8000'

  const pip_install = tool({
    description:
      'Install or downgrade a Python package in the execution environment. ' +
      'Use this ONLY to fix package version conflicts that prevent the script from running ' +
      '(e.g. numpy/ml_dtypes incompatibility). ' +
      'Example: pip_install({ package: "numpy>=1.26,<2.0" }). ' +
      'After a successful install, retry the failing step.',
    inputSchema: z.object({
      package: z.string().describe('Package spec, e.g. "numpy>=1.26,<2.0" or "tensorflow-cpu==2.16.1"'),
    }),
    execute: async (input: { package: string }) => {
      if (!sessionId) return { ok: false, stdout: 'No active session.' }
      try {
        const res = await fetch(`${apiBase}/platform/pip_install/${sessionId}`, {
          method : 'POST',
          headers: { 'Content-Type': 'application/json' },
          body   : JSON.stringify({ package: input.package }),
          signal : AbortSignal.timeout(130_000),
        })
        return await res.json()
      } catch (e) {
        return { ok: false, stdout: String(e) }
      }
    },
  })

  const run_code = tool({
    description:
      'Execute a Python snippet in the session working directory and return the output. ' +
      'Use this to test data loading, inspect column names/shapes/dtypes, and validate ' +
      'logic before writing the full script. Has access to dataset.csv and all installed packages. ' +
      '30-second timeout — not for full training runs. ' +
      'NOTE: subprocess and os.system are blocked here — use pip_install tool for package fixes.',
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

  return { pip_install, run_code, read_script, edit_script }
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

      // Preflight: verify the script exists and has content before hitting the notebook endpoint.
      // If edit_script was never called, this returns a clear instruction to write it first.
      try {
        const scriptRes = await fetch(`${apiBase}/platform/script/${sessionId}`, {
          signal: AbortSignal.timeout(5_000),
        })
        if (!scriptRes.ok) {
          return {
            error : 'No training script found — generated_script.py does not exist yet.',
            action: 'MANDATORY: Call edit_script(old_str="__REPLACE_ALL__", new_str=<complete Python training script>) to write the full script FIRST. Then call create_notebook again.',
          }
        }
        const scriptData = await scriptRes.json().catch(() => null)
        const lineCount  = scriptData?.line_count ?? 0
        if (lineCount < 10) {
          return {
            error : `Script exists but is nearly empty (${lineCount} lines). It must be written before creating a notebook.`,
            action: 'MANDATORY: Call edit_script(old_str="__REPLACE_ALL__", new_str=<complete Python training script>) to write the full script FIRST. Then call create_notebook again.',
          }
        }
      } catch (prefErr) {
        // Non-fatal — let the actual notebook call surface any real error
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

// ── Locked tool stub ──────────────────────────────────────────────────────────
// When a tool is not available in the current phase, replace it with a stub that
// returns an immediate error + instruction. This is better than omitting the tool
// entirely because: (a) Anthropic's API may require schemas for tools that appear
// in conversation history, and (b) it gives the model a clear recovery path.
function lockedTool(name: string, instruction: string) {
  return tool({
    description: `[LOCKED — unavailable in current phase] Do NOT call this. ${instruction}`,
    inputSchema: z.object({}),
    execute: async () => ({
      error : `PHASE ERROR: '${name}' is not available in the current phase.`,
      action: instruction,
    }),
  })
}

// ── Base system prompt (role, format, code constraints) ──────────────────────
// Phase-specific behaviour is injected at runtime via buildSystemPrompt().
const SYSTEM_BASE = `\
<role>
You are Obsidian Networks — an expert ML research engineer specialising in TensorFlow/Keras.
Your purpose is to help users design, research, and generate production-ready deep learning models.
You operate in three strict phases: RESEARCH → PLAN → BUILD.
You NEVER skip phases. You NEVER write code before the plan is approved.

You support two modes of operation:
1. DATASET MODE — user uploads a tabular/time-series/structured dataset and describes a prediction goal.
2. DESCRIPTION MODE — user describes a vision, video, NLP, or custom architecture task with no dataset upload (CNNs, Video models, Transformers, ViTs, etc.). In this mode you interview the user to gather all required specifications before researching.
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
- CRITICAL — For CNN / image models with no uploaded dataset: generate synthetic image tensors with tf.random.normal(shape=(N, H, W, C)) matching the agreed input resolution. NEVER reference a file path that does not exist.
- CRITICAL — For video models: generate synthetic clip tensors with tf.random.normal(shape=(N, frames, H, W, C)). Use Conv3D or ConvLSTM2D. NEVER manually loop over frames with separate 2D convolutions unless the architecture specifically requires it.
- CRITICAL — For Transformers / ViT: implement the full attention mechanism using tensorflow.keras layers (MultiHeadAttention, LayerNormalization, Dense). For ViT, use Conv2D with stride=patch_size to extract patches — NEVER use a for-loop over patches. For text transformers, use tensorflow.keras.layers.Embedding + positional encoding.
- For all description-mode models (no dataset): the script MUST demonstrate a complete forward pass and at least one training step using the synthetic data, saving the model to output/model.keras.
- CRITICAL — NEVER add subprocess, os.system, os.popen, or eval/exec calls anywhere in the training script or in run_code snippets — these are blocked by the security validator. If you encounter a package version error, use the pip_install tool once, then retry. Do NOT loop through run_code trying different workarounds.
</constraints>

<plan_template>
When writing the plan document, use exactly this structure:

# ML Plan: [Task Name]

## 1. Problem Type & Task Framing
[Binary/multi-class/regression/time-series/RL/CNN/Video/Transformer — why, based on dataset or user description]

## 2. Data & Input Specification
[For tabular: column-by-column transformations, which are dropped/combined, source URL.
For CNN/Video/Transformer: input tensor shape, resolution, channels, sequence length, patch size, data source (real or synthetic), preprocessing pipeline. Source: URL]

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

Available tools: search_arxiv, ingest_arxiv_paper, fetch_tensorflow_docs, finalize_research
LOCKED tools (not available yet): query_research, produce_plan, approve_plan, run_code, edit_script, create_notebook

BEHAVIOUR:

── DATASET MODE (user uploaded a tabular/structured file) ──────────────────────

1. DATASET UPLOADED, NO CLEAR GOAL YET?
   → Do NOT run tools. Greet the dataset (1 sentence). Ask ONE open-ended question about the goal.
   → Do NOT name any architectures.

2. DATASET UPLOADED AND GOAL IS CLEAR? Execute ALL steps:
   STEP 1 — arXiv search (call BOTH in parallel):
     search_arxiv(query="<primary domain query>", max_results=5)
     search_arxiv(query="<different angle>", max_results=5)
   STEP 2 — Read abstracts, select 3–4 most relevant papers. Use ONLY the arxiv_id field as returned.
   STEP 3 — Ingest papers in parallel: ingest_arxiv_paper(arxiv_id, title) for each.
   STEP 4 — Fetch Keras/TF docs in parallel (architecture type, training callbacks, preprocessing).
   STEP 5 — finalize_research()

── DESCRIPTION MODE (CNN / Video / Transformer / NLP — no dataset upload) ──────

3. USER DESCRIBES A VISION, VIDEO, TRANSFORMER, OR CUSTOM ARCHITECTURE TASK WITH NO DATASET?
   → Do NOT run tools yet. You MUST interview the user first. Ask ONE question at a time. Gather ALL of:

   For CNN / image classification / object detection:
     a. What is the input — single images or video frames? What approximate resolution?
     b. How many classes / what is the output (classification, detection, segmentation)?
     c. Is this from scratch or fine-tuning a pretrained backbone (ResNet, EfficientNet, MobileNet)?
     d. What is the deployment target — GPU server, edge device, browser?
     e. Approximate dataset size or data source (even if synthetic/described — affects architecture depth).

   For Video / temporal models:
     a. Is the task clip classification, frame prediction, action recognition, or something else?
     b. What is the input — raw video frames, optical flow, or pre-extracted features?
     c. What temporal architecture is preferred or open to recommendation — 3D CNN, ConvLSTM, VideoTransformer?
     d. Clip length and frame rate expectations?

   For Transformers / ViT / NLP:
     a. Is this text, image patches, or multimodal?
     b. Sequence length and vocabulary size (for text) or patch size (for ViT)?
     c. Task: classification, generation, embedding, translation?
     d. Encoder-only, decoder-only, or encoder-decoder?
     e. Training from scratch or fine-tuning a pretrained checkpoint?

   → Once ALL necessary information is gathered, proceed with the research steps (STEP 1–5 above).
   → The script will use synthetic/procedural data generation internally since no dataset file is provided.
   → In the BUILD phase, the script MUST generate its own sample data (e.g. tf.random.normal, np.random) to demonstrate the full model forward pass and training loop.

── SHARED RULES ────────────────────────────────────────────────────────────────
- ALWAYS use arxiv_id exactly as returned by search_arxiv — never modify, guess, or construct IDs.
- NEVER call finalize_research before at least 3 papers are ingested.
- NEVER call fetch_url or ingest_url — those tools no longer exist.
- NEVER ask more than one question at a time.
- NEVER name or recommend an architecture before research is complete.
</phase>`
  }

  if (phase === 'planning') {
    return SYSTEM_BASE + `

<phase current="planning">
CURRENT PHASE: PLANNING

Available tools: query_research, produce_plan
LOCKED tools (not available yet): search_arxiv, ingest_arxiv_paper, fetch_tensorflow_docs, run_code, edit_script, create_notebook

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

════════════════════════════════════════════════════════════════
MANDATORY BUILD SEQUENCE — YOU MUST FOLLOW THIS EXACTLY
════════════════════════════════════════════════════════════════

STEP 1 — Approve the plan (FIRST CALL, DO THIS ONCE):
   Call approve_plan() → this unlocks edit_script and create_notebook.
   If already in building phase (approve_plan was called in a prior turn), SKIP this step.
   NEVER call approve_plan more than once.

STEP 2 — Inspect the dataset (DATASET MODE ONLY):
   Call run_code("import pandas as pd; df = pd.read_csv('dataset.csv'); print(df.shape); print(df.dtypes); print(df.head(2))")
   SKIP entirely for CNN / Video / Transformer / description-mode tasks (no dataset file exists).

STEP 3 — Write the training script (MANDATORY — ALWAYS DO THIS BEFORE create_notebook):
   Call edit_script(old_str="__REPLACE_ALL__", new_str=<complete Python training script>)
   ⚠ WARNING: create_notebook will FAIL if you skip this step. The script MUST be written first.
   - Implement EXACTLY the architecture from the approved plan above
   - Every in-code comment must reference the plan section or paper source

STEP 4 — Create the notebook:
   Call create_notebook(description="<one-line title>")
   This reads the already-saved script — do NOT pass the script here.

STEP 5 — Handle validation errors (if create_notebook returns errors):
   Fix ALL listed errors with edit_script, then call create_notebook again.
   Repeat until create_notebook returns ok: true.

STEP 6 — Confirm success:
   Reply with 3–5 bullets: architecture summary, key hyperparameters, expected output. No code shown.

════════════════════════════════════════════════════════════════
RULE: The order is ALWAYS: approve_plan → (run_code) → edit_script → create_notebook
      You MUST NOT call create_notebook before edit_script has written the script.
      You MUST NOT call query_research or produce_plan in the BUILD phase.
════════════════════════════════════════════════════════════════

USER ASKS TO CHANGE/IMPROVE THE MODEL?
   → Acknowledge, call read_script + edit_script to apply changes
   → Call create_notebook again with updated description
   → Reply with 2–3 sentences describing what changed

create_notebook RETURNS HARD STOP?
   → Apologise, ask user to describe a simpler architecture.

run_code / create_notebook RETURNS AN ENVIRONMENT ERROR (numpy/ml_dtypes conflict, missing package, version mismatch)?
   → Fix it with run_code — NEVER embed pip install in the training script itself.
   → For numpy/ml_dtypes conflicts: run_code("import subprocess; subprocess.run(['pip', 'install', 'numpy>=1.26,<2.0', '--quiet'], check=True)")
   → Then retry the failing step. If the pip install itself fails or the error persists, stop and tell the user their environment has a package conflict that must be resolved outside the platform.

NEVER deviate from the approved plan's architecture without explicit user instruction.
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

  // Phase-gated tool set — only the tools for the current phase are passed to the
  // model. Passing locked tools even with system-prompt warnings still allows the
  // model to call them; the only reliable gate is to omit them entirely.
  const isBuilding = sessionPhase === 'approved' || sessionPhase === 'building'
  const { approve_plan } = createPlanningTools(sessionId)

  const phaseTools = isBuilding
    // BUILD phase: script tools + create_notebook + approve_plan (needed on first turn).
    // query_research and produce_plan are locked stubs so the model gets a recovery
    // instruction if it tries to call them (e.g. due to conversation history context).
    ? {
        approve_plan,
        ...createScriptTools(sessionId),
        create_notebook : createNotebookTool(sessionId),
        query_research  : lockedTool('query_research',  'Research and planning are complete. Call edit_script to write the training script, then call create_notebook.'),
        produce_plan    : lockedTool('produce_plan',    'Planning is complete. Call edit_script to write the training script, then call create_notebook.'),
      }
    : sessionPhase === 'planning'
    // PLANNING phase: all planning tools + build tools.
    // Build tools are included so that when approve_plan is called within this turn,
    // the model can immediately proceed to edit_script → create_notebook without
    // requiring an extra round-trip from the user.
    // produce_plan is a locked stub in build phase; here it's real.
    ? {
        ...createPlanningTools(sessionId),
        ...createScriptTools(sessionId),
        create_notebook: createNotebookTool(sessionId),
      }
    // RESEARCH phase (idle / researching): research + finalize only
    : createResearchTools(sessionId)

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
    stopWhen       : stepCountIs(20),
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
