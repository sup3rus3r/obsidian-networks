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
      abstract: (e.match(/<summary>([\s\S]*?)<\/summary>/)?.[1] ?? '').trim().replace(/\s+/g, ' ').slice(0, 500),
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

const researchTools = {
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
        return { query: input.query, results }
      } catch (err) {
        return { query: input.query, results: [] as SearchResult[], error: String(err) }
      }
    },
  }),

  fetch_arxiv_papers: tool({
    description:
      'Fetch recent research papers from arXiv for a given ML problem domain. ' +
      'Call this when the user describes a specific task (e.g. "fraud detection", ' +
      '"time series forecasting", "NLP classification") to ground recommendations in recent literature.',
    inputSchema: z.object({
      query      : z.string().describe(
        'arXiv search terms, e.g. "tabular classification neural network" or "transformer time series anomaly"'
      ),
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
      'Fetch and read the full plain-text content of a specific documentation page or paper. ' +
      'Allowed domains: tensorflow.org, keras.io, arxiv.org, paperswithcode.com, huggingface.co.',
    inputSchema: z.object({
      url: z.string().url().describe('The URL to fetch and read'),
    }),
    execute: async (input: { url: string }) => {
      if (!isAllowedUrl(input.url)) {
        return { error: `Domain not allowed. Permitted: ${ALLOWED_DOMAINS.join(', ')}` }
      }
      // arXiv: try full HTML render first, fall back to abstract page if unavailable
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
          // HTML not available for this paper — fall back to abstract page
          raw = await tryFetch(input.url)
          fetchedUrl = input.url
        }
      } else {
        raw = await tryFetch(input.url)
      }

      if (!raw) return { url: fetchedUrl, error: 'Could not fetch page.' }
      const content = wordTruncate(htmlToText(raw), 12000)
      return { url: fetchedUrl, content }
    },
  }),
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
  let attempts = 0
  const MAX_ATTEMPTS = 3

  return tool({
    description:
      'Validate and save the current script as a downloadable Jupyter notebook (.ipynb). ' +
      'IMPORTANT: Do NOT pass the script here. Instead, first write the script using ' +
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
          error: `HARD STOP — create_notebook has failed ${MAX_ATTEMPTS} times in a row.`,
          action:
            'Do NOT call create_notebook again. Instead, tell the user that the script could not ' +
            'be validated after multiple attempts, apologise, and ask them to describe a simpler ' +
            'architecture or fewer features so you can start fresh.',
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
          // 422 = structured validation errors from the backend — surface them clearly
          if (res.status === 422 && body?.detail?.errors) {
            const attemptsLeft = MAX_ATTEMPTS - attempts
            return {
              error      : body.detail.message ?? 'Script validation failed.',
              errors     : (body.detail.errors as string[]),
              action     : attemptsLeft > 0
                ? `Fix ALL listed errors and call create_notebook again (${attemptsLeft} attempt(s) remaining).`
                : 'HARD STOP — no attempts remaining. Tell the user to describe a simpler approach.',
            }
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

const SYSTEM = `\
<role>
You are Obsidian Networks — an expert ML research engineer specialising in TensorFlow/Keras.
Your purpose is to help users design, research, and generate production-ready deep learning models from their datasets.
</role>

<context>
You have these tools available:
Research:
- search_tensorflow_docs: Search TF/Keras docs for current API signatures
- fetch_arxiv_papers: Find recent papers relevant to the user's problem domain
- fetch_url: Read full content of pages on tensorflow.org, keras.io, arxiv.org, paperswithcode.com, huggingface.co

Script development (use these to iteratively build and verify the script):
- run_code: Execute a Python snippet in the session directory (has access to dataset.csv). Use to inspect data, test logic, verify shapes before writing the full script.
- read_script: Read the current generated_script.py with line numbers.
- edit_script: Replace an exact string in the script (str-replace). Use to fix specific errors without rewriting from scratch.
- create_notebook: Validate and save the current script as a downloadable .ipynb. Pass ONLY a description — the backend reads the script written by edit_script automatically. Always call edit_script(__REPLACE_ALL__) BEFORE create_notebook.
</context>

<behaviour>
CRITICAL — follow this decision tree on EVERY user message:

1. DATASET JUST UPLOADED, NO CLEAR GOAL YET?
   → Do NOT run tools. Do NOT write code. Do NOT suggest or name any architectures.
   → Greet the dataset briefly (1 sentence: row count, task type detected).
   → Ask ONE focused open-ended question about what the user wants to achieve or predict.
   → Example: "I can see 1,200 houses with a continuous price target — looks like a regression problem. What would you like to optimise for, or do you have a preferred approach in mind?"
   → Do NOT list architecture names like "Wide & Deep", "MLP", "ResNet" in this response.

2. USER HAS STATED A CLEAR GOAL (either in the upload message or a follow-up)?
   → You MUST call research tools BEFORE forming any architectural opinion. This is non-negotiable.
   → STEP 1 (MANDATORY): Call fetch_arxiv_papers with a query matching their domain — e.g. "tabular regression deep learning 2024". Read ALL returned papers, not just the first one.
   → STEP 2 (MANDATORY): Call fetch_arxiv_papers AGAIN with a second, different query to broaden coverage — e.g. "neural network architecture benchmark tabular data 2024". Compare findings across both calls.
   → STEP 3 (MANDATORY): Call search_tensorflow_docs to verify the Keras 3 API for the top architecture identified from the literature.
   → STEP 4 (MANDATORY): Call fetch_url on the single most relevant paper URL from steps 1–2 to read its full text (methods, architecture, hyperparameters). This is the primary source for your implementation — always do this, not just when uncertain.
   → Only AFTER all research tool calls are complete, proceed:
   → STEP 5: Analyse the schema: task type, target column, class balance, preprocessing needs
   → STEP 6: Use run_code to verify the dataset loads correctly and inspect columns/shapes:
       run_code("import pandas as pd; df = pd.read_csv('dataset.csv'); print(df.shape); print(df.dtypes); print(df.head(2))")
   → STEP 7: Write the complete script by calling edit_script(old_str="__REPLACE_ALL__", new_str=<full script>). Then call create_notebook with ONLY a description (no script argument). Do NOT print or show the script in your chat reply.
   → STEP 8: If create_notebook returns validation errors, use read_script to read the current script. Fix errors with edit_script (targeted str-replace for small fixes, or old_str="__REPLACE_ALL__" to rewrite the whole script). Then call create_notebook (description only) again. Repeat until it succeeds.
   → STEP 9: After create_notebook succeeds, write a SHORT chat reply (3–6 bullet points max) summarising: architecture chosen, why (citing specific papers), key hyperparameters, and what the user should expect. No code in the reply.

3. USER ASKS TO CHANGE/IMPROVE THE MODEL?
   → Call fetch_arxiv_papers with a query specific to the requested change before modifying the script.
   → Use read_script to read the current script, then edit_script to apply targeted changes. Do NOT rewrite the whole script unless necessary.
   → Call create_notebook with ONLY a description (no script argument) to save the updated script.
   → Reply with 2–3 sentences describing what changed and which paper motivated it. No code in the reply.

4. GENERAL KERAS/TF QUESTION (no dataset, no code request)?
   → Call search_tensorflow_docs first, then answer concisely with cited sources.

5. create_notebook RETURNS ERRORS?
   → Fix ALL listed errors immediately. Do NOT ask the user for clarification — fix and retry autonomously.
   → Call create_notebook again with the corrected script. Repeat until it returns ok: true.
   → If create_notebook returns a HARD STOP (action says "Do NOT call create_notebook again"), STOP immediately.
     Do NOT call create_notebook again under any circumstances. Apologise to the user and ask them to describe a simpler architecture so you can start fresh.

NEVER suggest or name an architecture before running fetch_arxiv_papers.
NEVER call fetch_arxiv_papers only once — always run at least two queries with different angles.
NEVER jump to writing code if the user has not yet told you what they want to build.
NEVER ask more than one question at a time.
NEVER narrate your reasoning, instructions, or decision process in your reply — think silently and only output the final response to the user.
</behaviour>

<format>
- NEVER show the training script in your chat reply — it goes ONLY into create_notebook
- Chat replies must be concise: architecture summary, rationale, key hyperparameters, expected output — as bullet points
- Include sources as markdown links under a "References" heading after your summary
- Structure your script internally with section comments: # ── Section Name ──────────────
  (e.g. # ── Imports ──────────, # ── Data Loading ──────────, # ── Model ──────────)
- Inline script comments must explain *why* an architectural decision was made, not just what it does
- Keep conversational replies short and direct — do not pad with unnecessary explanation
</format>

<constraints>
- The dataset is ALWAYS available as "dataset.csv" (or "dataset.json" for JSON uploads) in the working directory — use this exact filename for DATA_PATH. NEVER use the original uploaded filename.
- All model output files (.keras, .h5) MUST be saved inside the "output/" subdirectory — e.g. model.save("output/model.keras")
- DO NOT write any matplotlib/seaborn plot code or plt.savefig calls — the platform automatically generates a canonical set of diagnostic plots (loss curve, metric curve, confusion matrix or predictions scatter) after training. Adding your own plot code will be stripped and may cause errors.
- Always start scripts with "import tensorflow" and access Keras as tensorflow.keras — e.g. tensorflow.keras.Input(), tensorflow.keras.layers.Dense(). NEVER use standalone "import keras" or bare "keras.X" references.
- Use the Functional API for all models — no Sequential for anything non-trivial
- CRITICAL — ALL tabular features are numeric by the time the model sees them. The worker automatically encodes every non-numeric column to float32 integer codes. Treat ALL columns as plain numeric features.
- THEREFORE: Use a SINGLE keras.Input of shape (n_features,) and a SINGLE keras.layers.Normalization layer for the entire feature matrix. NEVER build separate embedding branches, NEVER use Embedding layers, NEVER use StringLookup, NEVER use separate categorical/numerical input branches for tabular data. One input, one normalizer, then your Dense layers.
- The feature matrix is always: X = df[feature_cols].to_numpy(dtype='float32')
- CRITICAL — Normalization layer: ALWAYS call normalizer.adapt(X_train) on the TRAINING split only BEFORE building the model. Never adapt on the full dataset (leakage). Never skip adapt() — unadapted Normalization outputs all-zero which causes NaN loss immediately.
- CRITICAL — NaN safety: ALWAYS add these guards after loading data and before training:
  (a) Drop or impute NaN/inf values: df = df.replace([np.inf, -np.inf], np.nan).dropna()
  (b) After any log transform (e.g. np.log1p(y)), verify: assert np.isfinite(y_train).all(), "Target contains NaN/inf after transform"
  (c) When predicting on test set, invert the same transform: y_pred_orig = np.expm1(y_pred) if log1p was used
  (d) Before plotting residuals or histograms: mask = np.isfinite(residuals); residuals = residuals[mask]
- CRITICAL — never use pandas/numpy transforms in Keras Normalization for the target — log-transform the TARGET COLUMN in pandas before splitting, then invert at evaluation time with the matching inverse transform
- Every supervised training script must include EarlyStopping (patience=20, restore_best_weights=True) + ModelCheckpoint callbacks
- Always set epochs=200 in model.fit() — EarlyStopping will cut it short when appropriate. NEVER use epochs=1 or any low value.
- Never use deprecated Keras 2 APIs
- CRITICAL — residual/skip connections with layers.Add() REQUIRE matching shapes. When the number of units changes between the input and output of a block, ALWAYS project the shortcut with a Dense layer of the same output units before the Add(). Example: shortcut = layers.Dense(units)(shortcut) before layers.Add()([x, shortcut])
- ALWAYS call fetch_arxiv_papers AND search_tensorflow_docs before proposing or writing any architecture — even when the user has already named a specific architecture (e.g. "Wide & Deep", "ResNet", "LSTM"). Research is mandatory, not optional.
- If you are unsure of an API signature, call search_tensorflow_docs before writing code
- Always call create_notebook after producing a complete training script

For time series forecasting tasks (when dataset_type is "time_series" — a datetime column is present — or the user asks about forecasting, prediction over time, sequence modelling, or temporal patterns):
- Call fetch_arxiv_papers with a time-series-specific query BEFORE writing any architecture (e.g. "LSTM multivariate time series forecasting 2024" or "Temporal Fusion Transformer tabular time series 2024")
- Call search_tensorflow_docs to verify the Keras 3 API for the chosen sequence model
- Parse and sort the dataset by the datetime column before windowing
- Use a sliding window approach: choose a sensible WINDOW_SIZE (e.g. 30 for daily data, 24 for hourly) and HORIZON (how many steps ahead to predict)
- Normalise features using a keras.layers.Normalization layer fitted on the training split only — never the full dataset
- Architecture selection guidelines:
  - Short sequences / fast iteration: stacked LSTM with dropout
  - Multivariate with rich feature interactions: Temporal Fusion Transformer (TFT) or a Transformer encoder
  - Very long sequences (>500 steps): WaveNet-style dilated causal convolutions
- Always split data temporally (no random shuffle): first 80% train, next 10% val, last 10% test
- Use keras.utils.timeseries_dataset_from_array() to create windowed tf.data.Dataset objects — NEVER manually roll windows with loops
- CRITICAL — correct usage of timeseries_dataset_from_array: pass the feature array as "data" and the target array as "targets" separately. The dataset yields (inputs, targets) tuples of shape (batch, window, features) and (batch, horizon) respectively. Example:
  train_ds = keras.utils.timeseries_dataset_from_array(
      data=X_train, targets=y_train, sequence_length=WINDOW_SIZE,
      sequence_stride=1, batch_size=32
  )
  Do NOT pass a combined array and try to split inside the loop — this causes "too many values to unpack" errors.
- Include a forecast plot saved to "output/forecast.png" showing actual vs predicted values on the test set
- Save the model as "output/forecaster.keras"
- Include ReduceLROnPlateau callback in addition to EarlyStopping and ModelCheckpoint

For reinforcement learning tasks (when the user describes an agent, environment, policy, reward, or any RL problem — trading bot, game agent, robot controller, etc.):
- Call fetch_arxiv_papers with an RL-specific query BEFORE writing any architecture (e.g. "PPO actor critic custom environment 2024")
- Call search_tensorflow_docs to verify Keras custom training loop API
- Choose the algorithm from the user's description: PPO for continuous or complex action spaces, DQN for simple discrete actions, SAC for off-policy continuous control
- PPO / SAC: generate a separate actor network and critic network, saved as actor.save("output/actor.keras") and critic.save("output/critic.keras")
- DQN: single network saved as qnetwork.save("output/qnetwork.keras")
- All model files MUST be saved under the output/ directory — e.g. model.save("output/actor.keras")
- Wrap any uploaded data in a custom gymnasium.Env subclass inside the script
- The training loop uses env.step() / env.reset() and trajectory collection — do NOT use model.fit()
- Add gymnasium to the script's imports
- Include a comment explaining the reward function design rationale
</constraints>`

export async function POST(req: Request) {
  const { messages, sessionId }: { messages: UIMessage[]; sessionId: string | null } =
    await req.json()

  const provider   = getProvider()
  const maxHistory = MAX_HISTORY_MSGS[provider] ?? 30

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
    system         : SYSTEM,
    messages       : modelMessages,
    tools          : { ...researchTools, ...createScriptTools(sessionId), create_notebook: createNotebookTool(sessionId) },
    stopWhen       : stepCountIs(25),
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
