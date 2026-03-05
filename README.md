<div align="center">

<img src="frontend/public/logo.svg" alt="Obsidian Networks" width="280" />

---

## What is Obsidian Networks?

**Obsidian Networks** is an AI-powered platform that generates **complete machine learning pipelines** from a simple description of your problem.

Instead of manually researching architectures, writing training code, debugging errors, and configuring environments, the platform automates the entire workflow.

Upload your dataset → describe your goal → receive a fully functional training script and trained model.

The generated code follows modern TensorFlow/Keras practices and is designed to be **production-ready, transparent, and editable**.

---

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/sup3rus3r/obsidian-networks?style=social)](https://github.com/sup3rus3r/obsidian-networks/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/sup3rus3r/obsidian-networks?style=social)](https://github.com/sup3rus3r/obsidian-networks/network/members)
[![GitHub Issues](https://img.shields.io/github/issues/sup3rus3r/obsidian-networks)](https://github.com/sup3rus3r/obsidian-networks/issues)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-16-000000?logo=next.js&logoColor=white)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Redis](https://img.shields.io/badge/Redis-7-DC382D?logo=redis&logoColor=white)](https://redis.io/)

---

**If you find this project useful, please consider giving it a star!** It helps others discover the project and motivates continued development.

[**Give it a Star**](https://github.com/sup3rus3r/obsidian-networks) &#11088;

</div>

---
<img src="docs/images/screen.png" width="100%" alt="Obsidian Networks UI" />

---

<img src="docs/images/demo_one.png" width="100%" alt="Obsidian Networks demo" />

---

<img src="docs/images/demo_two.png" width="100%" alt="Obsidian Networks demo" />


## TL;DR — Model Recommendations

Based on initial testing across providers, model quality for this use case ranks as follows:

**Claude (Anthropic) > GPT-4o (OpenAI) > Large local models via LM Studio**

- **Claude** (Sonnet / Opus) produces the most thorough research — it reliably calls all research tools before writing code, follows multi-step instructions precisely, and generates clean, runnable training scripts with minimal patching needed.
- **GPT-4o** performs well and follows tool-use instructions consistently, though it does less unprompted research than Claude.
- **LM Studio** (local models) works best with larger models (13B+). Smaller models tend to skip research tool calls and occasionally produce scripts with minor syntax issues. The platform automatically patches the most common errors, but output quality is noticeably lower than cloud providers.

If you are self-hosting for privacy or cost reasons, use the largest model your hardware supports. For best results, use Claude.

### Self-Healing Scripts

When a generated training script fails to compile, the platform automatically feeds the error message back to the AI model so it can diagnose and fix the problem — then regenerates the script and retries, without any manual intervention. This loop runs silently in the background; you will see the chat update with a fix and the compile button become available again once the corrected script is ready.

---

## Table of Contents

- [Why Obsidian Networks?](#why-obsidian-networks)
- [How It Works](#how-it-works)
- [Features](#features)
  - [Supervised Learning](#supervised-learning)
  - [Time Series Forecasting](#time-series-forecasting)
  - [Reinforcement Learning](#reinforcement-learning)
  - [Training Visualisation](#training-visualisation)
  - [Plot Gallery](#plot-gallery)
  - [Multi-Provider LLM](#multi-provider-llm)
  - [Notebook Export](#notebook-export)
  - [Session Privacy](#session-privacy)
  - [Self-Hosted](#self-hosted)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Docker](#docker-recommended)
  - [Local Development](#local-development)
  - [Environment Variables](#environment-variables)
- [Sample Datasets](#sample-datasets)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Security Model](#security-model)
- [Recent Updates](#recent-updates)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Why Obsidian Networks?

Most ML platforms assume you already know how to build models. **Obsidian Networks** inverts that assumption.

- **No ML expertise required** — Describe your goal in plain English. The AI selects the architecture, verifies the API, writes the code, and trains the model.
- **Research-backed output** — Before generating a single line of code, the agent queries arXiv and the TensorFlow/Keras documentation. Every script is grounded in recent literature.
- **End-to-end in one session** — From raw CSV to a trained `.keras` file without switching tools, writing boilerplate, or managing environments.
- **Time series support** — Upload hourly or daily data and receive a complete LSTM or Temporal Fusion Transformer script with correct temporal windowing and no data leakage.
- **Reinforcement learning support** — Describe an RL problem — trading agent, game controller, robot policy — and receive a complete Gymnasium environment, actor/critic networks, and a training loop.
- **Fully local option** — Point the platform at a local [LM Studio](https://lmstudio.ai/) server. Your data, models, and prompts never touch a cloud API.
- **Self-hosted & open-source** — One `docker compose up --build`. Your infrastructure, your data, your keys.

---

## How It Works

```
Upload dataset  →  Describe your goal  →  Compile & Train  →  Download your model
```

**1. Research**
The AI agent queries arXiv for papers relevant to your problem domain and searches the TensorFlow/Keras docs to verify current API signatures before writing a single line of code.

**2. Generate**
A complete, runnable Python training script is produced using the Keras Functional API — with dataset-aware preprocessing (Normalization, StringLookup, TextVectorization, timeseries windowing), EarlyStopping, ModelCheckpoint, and the correct save pattern for your architecture.

**3. Compile**
Click **Compile & Train**. The backend validates the script at the AST level and runs it in an isolated subprocess inside a hardened Docker sandbox. Progress, per-epoch metrics, and loss/accuracy charts stream back to the UI in real time via SSE.

**4. Download**
Grab your trained `.keras` model file(s), the auto-generated Jupyter notebook, and any training plots from the Downloads panel — ready to deploy or continue iterating anywhere.

---

## Features

### Supervised Learning

Automatic task detection for binary classification, multiclass classification, and regression. Preprocessing is handled entirely through Keras layers — `Normalization`, `StringLookup`, `TextVectorization` — so the trained model is fully portable with no sklearn pipelines or pandas transforms at inference time. EarlyStopping and ModelCheckpoint are included in every generated script.

---

### Time Series Forecasting

Upload any time-indexed CSV and describe a forecasting goal. The platform detects datetime columns, infers the sampling frequency, and selects an appropriate architecture.

| Architecture | When used | Output |
|---|---|---|
| **LSTM** | General sequence modelling, univariate or multivariate | `model.keras` |
| **Stacked LSTM** | Longer sequences, more complex patterns | `model.keras` |
| **Temporal Fusion Transformer** | Multi-horizon forecasting with interpretability | `model.keras` |

Key guarantees in every generated time series script:
- Temporal train/val/test split (no shuffling — prevents data leakage)
- `keras.utils.timeseries_dataset_from_array` for correct sliding-window batching
- EarlyStopping with patience=20 on validation loss
- Up to 200 epochs with checkpointing

---

### Reinforcement Learning

Describe an RL problem in plain English and the platform generates a complete Gymnasium environment, the appropriate network architecture, and a trajectory-based training loop using `env.step()` / `env.reset()` — never `model.fit()`. No file upload required.

| Algorithm | When to use | Output files |
|-----------|-------------|--------------|
| **PPO** | Continuous or complex action spaces | `actor.keras` + `critic.keras` |
| **DQN** | Simple discrete action spaces | `qnetwork.keras` |
| **SAC** | Off-policy continuous control | `actor.keras` + `critic_1.keras` + `critic_2.keras` |

The AI always searches arXiv for RL-specific papers before selecting an algorithm, and the reward function rationale is documented inline in every generated script.

---

### Training Visualisation

Live loss/accuracy charts appear inline during compilation, streamed epoch-by-epoch via SSE — no page refresh needed.

- Dual-axis layout: loss on the left y-axis, accuracy percentage on the right
- Click the expand icon to open a full-size chart with tooltip, legend, and stat pills
- Epochs badge in the Downloads panel shows how many epochs ran and whether EarlyStopping fired (e.g. `Trained for 47 epochs · early stopped / 200 max`)
- Built with Recharts + shadcn-style `ChartContainer`

---

### Plot Gallery

Generated scripts automatically save matplotlib figures to the session output directory. After compilation completes, any PNG/JPG/SVG files appear as a thumbnail gallery in the Downloads panel.

- Click any thumbnail to open a full-size lightbox dialog
- The panel shows a `Saving plots…` spinner after compilation while images are written to disk
- Useful for visualising training history, confusion matrices, feature importance, prediction error, and any other plots your script produces

---

### Multi-Provider LLM

Switch between providers without changing your workflow. Each provider uses the most efficient context management strategy available.

| Provider | Type | Notes |
|----------|------|-------|
| **Anthropic** | Cloud | Claude Opus 4.6, Sonnet 4.6, Haiku 4.5 |
| **OpenAI** | Cloud | GPT-4o, o3, any available model |
| **LM Studio** | Local | Any OpenAI-compatible local model |

**Context efficiency per provider:**

- **Anthropic** — Prompt caching (`cache_control: ephemeral`) is applied to the system prompt on every request, so the ~2k-token system prompt is read from cache rather than re-processed each turn. Server-side `contextManagement` automatically clears old tool-use results (arXiv abstracts, TF doc fetches) when the context approaches 60k input tokens, keeping only the 3 most recent tool turns. This dramatically reduces cost and latency on long sessions.
- **OpenAI / LM Studio** — `pruneMessages` strips tool call results older than the last 5 message turns before sending, keeping the context window lean without losing conversation continuity.

---

### Notebook Export

Every training script is saved as a `.ipynb` Jupyter notebook. Download it and continue iterating locally, on Google Colab, on Kaggle, or anywhere a Jupyter kernel runs.

The notebook includes a **per-platform environment setup cell** at the top — no more hunting for the right install command:

| Platform | What's included |
|---|---|
| **CPU (Linux / WSL2)** | venv setup, `tensorflow-cpu` install |
| **NVIDIA GPU (Linux / WSL2)** | `tensorflow[and-cuda]` install, GPU verification snippet |
| **Google Colab** | Pre-installed TF note, extra deps only |

When you request changes to a model — new architecture, different optimizer, added dropout — the AI updates the full script and calls `create_notebook` again, overwriting the previous version so you always have the latest.

---

### Session Privacy

Sessions are anonymous and ephemeral. Uploaded datasets and generated files are stored in an isolated per-session directory on the host and purged automatically after a configurable TTL (default: 4 hours). Nothing is persisted in a database.

---

### Self-Hosted

One command starts the full stack: Next.js frontend, FastAPI backend, Celery worker, and Redis.

```bash
docker compose up --build
```

No external services required beyond your LLM API key.

---

## Architecture

```
┌─────────────────────────────────┐      ┌─────────────────────────────────┐
│           Frontend              │      │            Backend              │
│     Next.js 16 + React 19       │─────>│      FastAPI + Python 3.11      │
│     Port 3000                   │ /api │      Port 8000                  │
│                                 │proxy │                                 │
│  ┌──────────┐  ┌─────────────┐  │      │  ┌───────────┐  ┌───────────┐  │
│  │ AI SDK 6 │  │  shadcn/ui  │  │      │  │ Sessions  │  │ Dataset   │  │
│  │ useChat  │  │  Tailwind 4 │  │      │  │ + TTL     │  │ Analysis  │  │
│  └──────────┘  └─────────────┘  │      │  └───────────┘  └───────────┘  │
│  ┌──────────┐  ┌─────────────┐  │      │  ┌───────────┐  ┌───────────┐  │
│  │  Shiki   │  │  Resizable  │  │      │  │ Notebook  │  │  Compile  │  │
│  │  (code)  │  │   Panels    │  │      │  │  Export   │  │ Endpoint  │  │
│  └──────────┘  └─────────────┘  │      │  └───────────┘  └───────────┘  │
└─────────────────────────────────┘      └──────────────┬──────────────────┘
                                                        │
                                         ┌──────────────▼──────────────────┐
                                         │    Redis 7 (broker + results)   │
                                         └──────────────┬──────────────────┘
                                                        │
                                         ┌──────────────▼──────────────────┐
                                         │    Celery Worker (--pool=solo)  │
                                         │  AST validation → subprocess    │
                                         │  TensorFlow / Keras / Gymnasium │
                                         │  Outputs: *.keras + plots       │
                                         │  Sandbox: seccomp + cap_drop    │
                                         └─────────────────────────────────┘
```

**How it works:**

1. The Next.js frontend proxies all `/api/platform/*` requests to the FastAPI backend via `next.config.ts` rewrites. Large file uploads bypass the proxy entirely, going directly to the backend at `NEXT_PUBLIC_UPLOAD_URL` to avoid buffering limits.
2. The AI chat route (`app/api/chat/route.ts`) calls `streamText` with research tools (`fetch_arxiv_papers`, `search_tensorflow_docs`, `fetch_url`), script development tools (`run_code`, `read_script`, `edit_script`), and `create_notebook`.
3. The agent writes the training script via `edit_script(old_str="__REPLACE_ALL__", ...)`, then calls `create_notebook` with only a description. The backend reads the saved script, applies AST/regex patches, validates it, and wraps it in a `.ipynb` notebook.
4. The Celery worker validates the script at the AST level, runs it in a subprocess with a stripped environment, and writes `.keras` files and plot images to the session's output directory.
5. Training progress and per-epoch metrics stream back to the frontend via Server-Sent Events; completed model files and plots appear in the Downloads panel.

---

## Quick Start

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| [Docker](https://docs.docker.com/get-docker/) | Latest | Container runtime |
| [Docker Compose](https://docs.docker.com/compose/) | v2+ | Multi-service orchestration |
| LLM API key | — | Anthropic, OpenAI, or LM Studio |

### Docker (Recommended)

```bash
git clone https://github.com/sup3rus3r/obsidian-networks.git
cd obsidian-networks

cp .env.example .env
# Open .env and set:
#   AUTH_SECRET — generate one with: openssl rand -base64 32
#   ANTHROPIC_API_KEY (or OPENAI_API_KEY, or LMSTUDIO_BASE_URL)

docker compose up --build
```

Open [http://localhost:3000](http://localhost:3000).

> **After changing backend code or environment variables**, rebuild only the worker (faster than a full rebuild):
> ```bash
> docker compose build worker && docker compose up -d worker
> ```

### Local Development

**Requirements:** Linux or WSL2, Python 3.11+, [uv](https://docs.astral.sh/uv/), Node.js 18+, Redis (`sudo apt install redis`)

```bash
git clone https://github.com/sup3rus3r/obsidian-networks.git
cd obsidian-networks

# Install dependencies
cd backend && uv sync && cd ..
cd frontend && npm install && cd ..

# Configure environment
cp backend/.env.example backend/.env   # fill in your keys
cp frontend/.env.example frontend/.env.local   # set AI_PROVIDER and API key

# Start everything with one command
npm run dev
```

This starts Next.js, FastAPI, the Celery worker, and Redis in a single terminal with colour-coded output. Open [http://localhost:3000](http://localhost:3000).

The dev server proxies all `/api/platform/*` requests to `http://localhost:8000`.

### Environment Variables

All configuration lives in `.env` at the repo root (Docker) or `frontend/.env.local` (local dev). See `.env.example` and `backend/.env.example` for full documentation.

#### Root / Frontend (`.env` or `frontend/.env.local`)

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_PROVIDER` | `anthropic` | LLM provider: `anthropic`, `openai`, or `lmstudio` |
| `AI_MODEL` | provider default | Override the model (e.g. `claude-opus-4-6`, `gpt-4o`) |
| `ANTHROPIC_API_KEY` | — | Required when `AI_PROVIDER=anthropic` |
| `OPENAI_API_KEY` | — | Required when `AI_PROVIDER=openai` |
| `LMSTUDIO_BASE_URL` | `http://localhost:1234/v1` | Required when `AI_PROVIDER=lmstudio` |

#### Backend / Root (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTH_SECRET` | — | **Required.** Random string for session signing (`openssl rand -base64 32`) |
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection URL |
| `SESSION_TTL_HOURS` | `4` | Hours before session files are purged |
| `MAX_FILE_SIZE_MB` | `500` | Maximum dataset upload size in MB |
| `SESSIONS_DIR` | `/sessions` | Where session files are stored |
| `MAX_TRAINING_MINUTES` | `10` | Hard timeout for the training subprocess |
| `MAX_MEMORY_GB` | `12` | Docker memory cap for the worker container |
| `MAX_OUTPUT_GB` | `10` | Max disk output per training run (`RLIMIT_FSIZE`) |
| `MAX_EPOCHS` | `200` | Maximum epochs the generated script may train for |
| `NEXT_PUBLIC_UPLOAD_URL` | `http://localhost:8000` | Direct browser→backend URL for large uploads (bypasses Next.js proxy). Set to your host's public backend address in production. |

---

## Sample Datasets

Two sample datasets are included in the repository root to get started immediately:

### `heart_failure_dataset.csv` — Binary Classification
918 patients with 11 clinical features including age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise angina, ST depression (Oldpeak), and ST slope. Target column: `HeartDisease` (0 = no disease, 1 = disease).

**Try:** *"Build a binary classifier to predict heart disease risk. Use a deep neural network with dropout regularisation."*
---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Next.js 16, React 19, TypeScript, Tailwind CSS 4, shadcn/ui |
| AI / Streaming | Vercel AI SDK 6, Anthropic / OpenAI / LM Studio |
| Charting | Recharts |
| Backend API | FastAPI 0.128+, Python 3.11, uv |
| Task Queue | Celery 5, Redis 7 |
| ML Runtime | TensorFlow 2.16+, Keras 3, NumPy, Pandas, scikit-learn, Gymnasium |
| Visualisation | Matplotlib, Seaborn, Statsmodels |
| Notebook | nbformat |
| Deployment | Docker, Docker Compose |

---

## Project Structure

```
obsidian-networks/
├── frontend/                       # Next.js 16 application
│   ├── app/
│   │   ├── api/
│   │   │   ├── chat/route.ts       # streamText, research tools, system prompt
│   │   │   ├── platform.ts         # API client helpers (upload, compile, download)
│   │   │   └── routes.ts           # Centralised URL constants
│   │   └── home/                   # Main application page
│   ├── components/
│   │   ├── artifacts/              # Downloads panel, compile section, plot gallery, SSE progress
│   │   ├── chat/                   # Chat UI, tool status chips, file attachments
│   │   └── ui/                     # shadcn/ui primitives
│   ├── hooks/                      # React hooks (session, scroll, environment)
│   └── lib/                        # Multi-provider model resolver, utilities
├── backend/                        # FastAPI application + Celery worker
│   ├── routers/
│   │   └── platform.py             # Session, upload, analysis, notebook, compile, image serving
│   ├── tasks.py                    # Celery task — AST validation + subprocess run + metrics parsing
│   ├── sessions.py                 # Session directory management + TTL cleanup
│   └── main.py                     # FastAPI app, CORS, router registration
├── docker-compose.yml              # Full stack: frontend + api + worker + redis
├── worker-seccomp.json             # Seccomp profile blocking dangerous syscalls in worker
├── .env.example                    # Root environment variables with documentation
├── backend/.env.example            # Backend-specific environment variables
├── heart_failure_dataset.csv       # Sample dataset — binary classification
└── energy_consumption_sample.csv   # Sample dataset — time series forecasting
```

---

## Security Model

### AST Validation

Generated scripts are validated at the AST level before execution. An allowlist restricts imports to `tensorflow`, `keras`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `gymnasium`, `matplotlib`, `seaborn`, `statsmodels`, and standard library modules. Calls to `os.system`, `os.popen`, `eval`, `exec`, and `execve` are explicitly blocked regardless of how they are invoked.

### Subprocess Isolation

Scripts run in a subprocess with a stripped environment — only `PATH`, `HOME`, `PYTHONUNBUFFERED`, and TensorFlow thread-count variables are set — with a configurable hard timeout (default: 10 minutes). Per-process resource limits are applied on Linux via `resource.setrlimit`:

| Limit | Value | Env var | Purpose |
|---|---|---|---|
| CPU time (`RLIMIT_CPU`) | 600s soft / 660s hard | `MAX_TRAINING_MINUTES` | Prevents runaway training loops |
| File size (`RLIMIT_FSIZE`) | 10 GB | `MAX_OUTPUT_GB` | Prevents disk-fill attacks |

Memory is capped at the Docker container level (`mem_limit`) rather than via `RLIMIT_AS` — virtual address space limits cause TensorFlow thread-pool failures when `RLIMIT_AS` is set too low.

### Docker Sandbox

The Celery worker container runs with an additional layer of Docker-level hardening:

- **Seccomp profile** (`worker-seccomp.json`): Blocks dangerous syscalls including `ptrace`, `mount`, `pivot_root`, `bpf`, `kexec_load`, `add_key`, and others. `clone` is permitted (required by `subprocess.Popen`); namespace-related syscalls (`unshare`, `setns`, `pivot_root`) remain blocked.
- **Capability restrictions**: `cap_drop: ALL` with only `DAC_OVERRIDE`, `SETUID`, `SETGID`, and `CHOWN` re-added
- **Process limits**: `nproc=65536`, `nofile=65536` (TensorFlow and data loaders spawn many threads)
- **Memory cap**: `mem_limit` default 12 GB RAM + 16 GB swap (configurable via `MAX_MEMORY_GB`)

Each session's files are isolated in a per-session directory that no other session can access.

---

## Recent Updates

### v0.6.0 — Reinforcement Learning + Large Uploads + Script Reliability
- **RL support**: Full PPO, DQN, and SAC generation with custom Gymnasium environments. Separate validator rules for RL scripts (no `model.fit()` required; saves `actor.keras`/`critic.keras`/`qnetwork.keras`). `patch_canonical_plots` skips RL scripts to avoid crashes on custom training loops.
- **Large file uploads**: Uploads up to 500 MB now go directly from the browser to the backend (`NEXT_PUBLIC_UPLOAD_URL`), bypassing Next.js proxy buffering. Starlette multipart parser raised to 10 GB part limit via manual `MultiPartParser` instantiation.
- **CSV delimiter auto-detection**: `csv.Sniffer` detects comma, semicolon, tab, and pipe delimiters automatically on upload.
- **Script write/validate flow**: The agent now writes scripts via `edit_script(old_str="__REPLACE_ALL__")` before calling `create_notebook` with only a description. This eliminates JSON serialisation corruption of large Python scripts passed through tool arguments.
- **Script always saved on validation failure**: `generated_script.py` is written before validation so `read_script` and `edit_script` are always available after a failed `create_notebook`, breaking the rewrite loop.
- **`edit_script` accepts `__REPLACE_ALL__`**: Lets the agent replace the entire script in one call when facing widespread syntax or indentation errors.
- **arXiv full-paper reading**: `fetch_url` rewrites `arxiv.org/abs/` → `arxiv.org/html/` to fetch full paper text instead of abstract-only pages, with fallback for papers without HTML rendering.
- **`run_code` tool**: Agent can now execute Python snippets in the session directory to inspect data, verify column names/shapes, and test logic before writing the full script.

### v0.5.0 — Context Efficiency
- **Anthropic**: system prompt cached via `cache_control: ephemeral` — reduces cost and latency on every request
- **Anthropic**: server-side `contextManagement` automatically clears old tool-use results (arXiv/TF doc fetches) when context exceeds 60k tokens, keeping the last 3 tool turns
- **OpenAI / LM Studio**: `pruneMessages` strips tool call results older than the last 5 message turns before sending
- Long sessions no longer hit rate limits or degrade in quality as conversation history grows

### v0.4.0 — Docker Sandbox + Plot Gallery
- Worker container hardened with seccomp profile, capability restrictions, and resource limits
- Generated scripts automatically save matplotlib/seaborn figures to session output
- Plot thumbnails appear in the Downloads panel after compilation; click to open full-size lightbox
- `Saving plots…` spinner prevents false "no plots" states during post-compilation file writes
- Epochs badge shows exact training duration and EarlyStopping status

### v0.3.0 — Time Series Forecasting
- Automatic detection of datetime columns and sampling frequency
- LSTM, Stacked LSTM, and Temporal Fusion Transformer architecture selection
- Correct `timeseries_dataset_from_array` windowing (no data leakage)
- `energy_consumption_sample.csv` sample dataset included
- Notebook environment setup cells added for Linux/WSL2 CPU, NVIDIA GPU, and Google Colab

### v0.2.0 — Training Metrics Visualisation
- Live loss/accuracy chart appears inline during compilation, streamed epoch-by-epoch via SSE
- Click the expand icon to open a full-size chart with tooltip, legend, and stat pills
- Dual-axis layout: loss on the left, accuracy percentage on the right
- Built with Recharts + shadcn-style `ChartContainer` component

---

## Roadmap

- [ ] Image dataset support (upload folder of images, auto-generate CNN/ViT architectures)
- [x] Time series forecasting templates (LSTM, Temporal Fusion Transformer)
- [x] Training metrics visualisation (live loss/accuracy charts during compilation)
- [ ] Model comparison — compile multiple architectures and compare results side-by-side
- [x] Docker-isolated compilation sandbox (seccomp profile, capability restrictions, resource limits)
- [x] Plot gallery — view matplotlib/seaborn figures inline after compilation
- [x] Reinforcement learning support (PPO, DQN, SAC with Gymnasium environments)
- [x] Large file uploads up to 500 MB (direct browser→backend, bypasses Next.js proxy)
- [ ] Export to ONNX / TensorFlow Lite for edge deployment

---

## Contributing

Contributions are welcome — bug reports, feature requests, documentation improvements, and pull requests alike. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting.

---

## License

Obsidian Networks is released under the [GNU Affero General Public License v3.0](LICENSE).

You are free to run, modify, and distribute this software. If you deploy a modified version as a network service, the AGPL requires you to make your modified source code available to users under the same terms.

---

<div align="center">

Made with care by [Mohammed Khan](https://github.com/sup3rus3r)

</div>
