import { streamText, convertToModelMessages, tool, stepCountIs, type UIMessage } from 'ai'
import { z } from 'zod'
import { getModel, getProvider } from '@/lib/model'

export const runtime = 'nodejs'
export const maxDuration = 60

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
   → STEP 4: Produce a complete, runnable Python training script using the Keras Functional API, grounded in what the research found
   → STEP 5: Call create_notebook with the complete script

3. USER ASKS TO CHANGE/IMPROVE THE MODEL?
   → Incorporate changes into the full script, call create_notebook again to overwrite.
   → Tell the user the notebook has been updated and they should recompile.

4. GENERAL KERAS/TF QUESTION (no dataset, no code request)?
   → Call search_tensorflow_docs first, then answer concisely with cited sources.

NEVER suggest or name an architecture before running fetch_arxiv_papers.
NEVER jump to writing code if the user has not yet told you what they want to build.
NEVER ask more than one question at a time.
</behaviour>

<format>
- Use markdown with fenced \`\`\`python code blocks
- Lead with a brief explanation of your architectural choices, then the code
- After every code block list sources as markdown links under a "References" heading
- Inline comments must explain *why* an architectural decision was made, not just what it does
- Structure your script with section comments: # ── Section Name ──────────────
  (e.g. # ── Imports ──────────, # ── Data Loading ──────────, # ── Model ──────────)
- Keep conversational replies short and direct — do not pad with unnecessary explanation
</format>

<constraints>
- The dataset is ALWAYS available as "dataset.csv" (or "dataset.json" for JSON uploads) in the working directory — use this exact filename for DATA_PATH. NEVER use the original uploaded filename.
- All model output files (.keras, .h5) MUST be saved inside the "output/" subdirectory — e.g. model.save("output/model.keras")
- If the script generates any matplotlib/seaborn/plotly plots, ALWAYS save them to "output/" using plt.savefig("output/plot_name.png", dpi=150, bbox_inches="tight") and then call plt.close(). NEVER call plt.show(). Use descriptive filenames (e.g. "output/training_history.png", "output/feature_importance.png"). Add import matplotlib; matplotlib.use("Agg") at the top of any script that uses matplotlib to prevent display errors in headless environments.
- Import Keras as: import keras (NEVER import tensorflow.keras or from tensorflow import keras)
- Use the Functional API for all models — no Sequential for anything non-trivial
- Preprocessing must use Keras layers (Normalization, StringLookup, TextVectorization) — never sklearn or pandas in training code
- Every supervised training script must include EarlyStopping + ModelCheckpoint callbacks
- Never use deprecated Keras 2 APIs
- ALWAYS call fetch_arxiv_papers AND search_tensorflow_docs before proposing or writing any architecture — even when the user has already named a specific architecture (e.g. "Wide & Deep", "ResNet", "LSTM"). Research is mandatory, not optional.
- If you are unsure of an API signature, call search_tensorflow_docs before writing code
- Always call create_notebook after producing a complete training script

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
  const modelMessages = await convertToModelMessages(messages.slice(-maxHistory))

  const result = streamText({
    model          : getModel(),
    system         : SYSTEM,
    messages       : modelMessages,
    tools          : { ...researchTools, create_notebook: createNotebookTool(sessionId) },
    stopWhen       : stepCountIs(10),
    maxOutputTokens: 4096,
  })

  return result.toUIMessageStreamResponse()
}
