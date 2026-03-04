import { streamText, convertToModelMessages, pruneMessages, tool, stepCountIs, type UIMessage } from 'ai'
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
      try {
        const html    = await fetchText(input.url)
        const content = wordTruncate(htmlToText(html), 6000)
        return { url: input.url, content }
      } catch (err) {
        return { url: input.url, error: String(err) }
      }
    },
  }),
}

function createNotebookTool(sessionId: string | null) {
  const apiBase = process.env.INTERNAL_API_URL ?? 'http://localhost:8000'

  return tool({
    description:
      'Save the final training script as a downloadable Jupyter notebook (.ipynb). ' +
      'Call this ONCE after you have produced the complete, runnable training script. ' +
      'The notebook will appear in the Downloads panel for the user to download.',
    inputSchema: z.object({
      script     : z.string().describe('The complete Python training script to save as a notebook'),
      description: z.string().describe(
        'One-line title for the notebook, e.g. "House Price Regression — Wide & Deep"'
      ),
    }),
    execute: async (input: { script: string; description: string }) => {
      if (!sessionId) return { error: 'No active session — cannot save notebook' }
      try {
        const res = await fetch(`${apiBase}/platform/notebook/${sessionId}`, {
          method : 'POST',
          headers: { 'Content-Type': 'application/json' },
          body   : JSON.stringify({ script: input.script, description: input.description }),
          signal : AbortSignal.timeout(15_000),
        })
        if (!res.ok) {
          const text = await res.text().catch(() => res.statusText)
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
You have four tools available:
- search_tensorflow_docs: Search TF/Keras docs for current API signatures
- fetch_arxiv_papers: Find recent papers relevant to the user's problem domain
- fetch_url: Read full content of pages on tensorflow.org, keras.io, arxiv.org, paperswithcode.com, huggingface.co
- create_notebook: Save the final training script as a downloadable .ipynb notebook
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
   → STEP 1 (MANDATORY): Call fetch_arxiv_papers with a query matching their domain — e.g. "tabular regression deep learning 2024"
   → STEP 2 (MANDATORY): Call search_tensorflow_docs to verify the Keras 3 API for the approach suggested by the literature
   → Only AFTER both tool calls return results, proceed:
   → STEP 3: Analyse the schema: task type, target column, class balance, preprocessing needs
   → STEP 4: Write the complete script and pass it DIRECTLY to create_notebook — do NOT print or show the script in your chat reply
   → STEP 5: After create_notebook succeeds, write a SHORT chat reply (3–6 bullet points max) summarising: architecture chosen, why, key hyperparameters, and what the user should expect. No code in the reply.

3. USER ASKS TO CHANGE/IMPROVE THE MODEL?
   → Incorporate changes into the full script, call create_notebook again to overwrite.
   → Reply with 2–3 sentences describing what changed. No code in the reply.

4. GENERAL KERAS/TF QUESTION (no dataset, no code request)?
   → Call search_tensorflow_docs first, then answer concisely with cited sources.

NEVER suggest or name an architecture before running fetch_arxiv_papers.
NEVER jump to writing code if the user has not yet told you what they want to build.
NEVER ask more than one question at a time.
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
- If the script generates any matplotlib/seaborn/plotly plots, ALWAYS save them to "output/" using plt.savefig("output/plot_name.png", dpi=150, bbox_inches="tight") and then call plt.close(). NEVER call plt.show(). Use descriptive filenames (e.g. "output/training_history.png", "output/feature_importance.png"). Add import matplotlib; matplotlib.use("Agg") at the top of any script that uses matplotlib to prevent display errors in headless environments.
- Import Keras as: import keras (NEVER import tensorflow.keras or from tensorflow import keras)
- Use the Functional API for all models — no Sequential for anything non-trivial
- CRITICAL — ALL tabular features are numeric by the time the model sees them. The worker pre-encodes every string/object/category column to float32 integer codes automatically. You must treat ALL columns (including originally-categorical ones like Sex, ChestPainType, etc.) as plain numeric features.
- THEREFORE: Use a SINGLE keras.Input of shape (n_features,) and a SINGLE keras.layers.Normalization layer for the entire feature matrix. NEVER build separate embedding branches, NEVER use Embedding layers, NEVER use StringLookup, NEVER use separate categorical/numerical input branches for tabular data. One input, one normalizer, then your Dense layers.
- The feature matrix is always: X = df[feature_cols].to_numpy(dtype='float32') — this always works because all columns are already float32.
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
- PPO / SAC: generate a separate actor network and critic network, saved as actor.save("actor.keras") and critic.save("critic.keras")
- DQN: single network saved as qnetwork.save("qnetwork.keras")
- All model files must be saved to the current working directory (the script runs with cwd = output/)
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
  const modelMessages = provider !== 'anthropic'
    ? pruneMessages({ messages: rawMessages, toolCalls: 'before-last-5-messages' })
    : rawMessages

  const result = streamText({
    model          : getModel(),
    system         : SYSTEM,
    messages       : modelMessages,
    tools          : { ...researchTools, create_notebook: createNotebookTool(sessionId) },
    stopWhen       : stepCountIs(10),
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

  return result.toUIMessageStreamResponse()
}
