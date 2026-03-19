<div align="center">

<img src="frontend/public/logo.svg" alt="Obsidian Networks" width="280" />

---

Obsidian Networks is an open-source platform that takes a problem described in plain language and builds a working ML solution from it — reading the relevant research, selecting an architecture, writing and running the code, and returning something you can actually use.

You describe the problem. It handles the rest: literature search, architecture selection, code generation, training, debugging. The output is a trained model and a Jupyter notebook you can open and modify.

It runs entirely on your own hardware. No data leaves your machine. No API calls to a training service. No account required beyond whatever LLM provider you choose to use.

Autonomous Research Mode goes further — instead of solving one problem, it runs an open-ended architecture search. Eight agents work in a loop across multiple generations: generating candidates, training them, scoring them, and recursing on the ones worth keeping. You set a domain and a goal, and it runs until it has something to show you.

**Built in pursuit of AGI. Open source under AGPL v3.**

---

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/sup3rus3r/obsidian-networks?style=social)](https://github.com/sup3rus3r/obsidian-networks/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/sup3rus3r/obsidian-networks?style=social)](https://github.com/sup3rus3r/obsidian-networks/network/members)
[![GitHub Issues](https://img.shields.io/github/issues/sup3rus3r/obsidian-networks)](https://github.com/sup3rus3r/obsidian-networks/issues)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-16-000000?logo=next.js&logoColor=white)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Redis](https://img.shields.io/badge/Redis-7-DC382D?logo=redis&logoColor=white)](https://redis.io/)

---

**If you find this project useful, please consider giving it a star!** It helps others discover the project and motivates continued development.

[**Give it a Star ⭐**](https://github.com/sup3rus3r/obsidian-networks)

</div>

---


## How It Works

You describe what you need. Everything else happens automatically.

**Research** — The platform goes to the source. It searches academic literature, reads papers, selects the ones that matter for your specific problem, and builds a private knowledge base from them — just for this session. It does not guess. It does not improvise from memory. It reads.

**Plan** — From that knowledge base, it forms a plan. Every decision — what approach to take, how to structure it, what to measure — is drawn from what it found in the literature and presented to you with citations. You read it. You approve it. Nothing is built until you do.

**Build** — Once you give the go-ahead, it builds exactly what was planned. The output is clean, readable, and yours to keep — a complete working script wrapped in a Jupyter notebook you can open anywhere.

**Train** — Click one button. It runs on your machine, in your environment. You watch it work in real time — metrics, charts, progress — as it trains. When it finishes, you download the result.

That is the whole process. No configuration. No expertise required. No waiting on someone else.

---

## Autonomous Research Mode

Beyond building a single model, Obsidian Networks includes an Autonomous Research Mode — a continuously running architecture discovery engine that explores an entire design space on your behalf.

You set a goal and a modality (vision, text, audio, time series, graph, multimodal, tabular, recommendation, or generative). Eight specialised AI agents then run in a coordinated loop:

**Researcher → Mathematician → Architect → Coder → Trainer → Evaluator → Validator → Critic**

Each generation produces a population of candidate architectures. The Critic scores every candidate on novelty, efficiency, soundness, and generalisation — then recurses on the top performers. The entire process runs autonomously. You watch a live feed of agent activity and, when it finishes, browse a ranked leaderboard of discovered architectures — each one downloadable as a production training script.

This is architecture search that used to require a dedicated research team. Now it runs overnight on your own hardware.

---

## What You Can Solve

**Prediction from data** — If you have a spreadsheet and a question — who will churn, what will sell, which patient is at risk, when will the part fail — upload it, describe what you want to know, and get a working model. No ML background needed.

**Forecasting over time** — If your data has a time dimension and you need to see what comes next, the platform handles the complexity of temporal modelling correctly. No data leakage. No manual windowing. Just describe the goal.

**Decision-making and control** — Describe a problem where something needs to learn to act — a trading strategy, a controller, an agent that optimises a process — and receive a complete environment and training loop built around it.

**Vision and video** — Describe what you need to see or detect. The platform asks the right questions, understands your constraints, and builds the architecture to match. No dataset upload required — just a clear description of the problem.

**Language and understanding** — From text classification to attention-based architectures for complex language tasks, built on current research rather than dated templates.

**Generation** — If the goal is to create rather than predict — synthetic data, novel images, learned representations — the platform builds the right generative architecture for it.

---

## Real-World Use Cases

**Early warning in a hospital ward.** Nurses document vitals and lab results every shift. Nobody has time to read all of it. Upload the EHR export, describe what deterioration looks like in your cohort, and get a model that flags the patients most likely to crash overnight — built on the sepsis prediction literature, trained on your data, not some generic benchmark.

**Predictive maintenance without the ML team.** Three years of vibration and temperature sensor logs from a production line. The failures are in there — they always were. Describe what a failure event looks like and the platform finds the temporal patterns that precede it, handles the lag windows correctly, and hands back something the maintenance team can actually use.

**Credit risk that doesn't stop at logistic regression.** There's always more signal in the data than a linear model can reach. Upload the application features, describe what default looks like, and get a deep tabular model trained on current research — with a full methodology writeup that explains every decision, ready to go in front of a risk committee.

**Climate forecasting on a small team's budget.** Thirty years of regional precipitation readings and a multi-step forecasting question that keeps getting pushed back because nobody has time to implement it properly. Describe the horizon and the spatial structure. The platform reads the literature, picks the right architecture, handles the windowing, and trains. What was sitting in a backlog for two years gets done in an afternoon.

**Gene expression classification with 20,000 features and 400 samples.** Most tools don't handle this well — they ignore the dimensionality problem or the class imbalance or both. Describe the tissue classes and the known imbalance. The platform finds the papers that specifically address this setup and builds accordingly.

**Recommendations that don't collapse on new users.** No interaction history, no embeddings, nothing to work from. Cold start is a solved problem in the research literature but a hard one to implement from scratch. Describe the constraint, describe the content, and get an architecture built around it rather than one that just pretends the problem doesn't exist.

**Finding a better execution policy without a six-week ablation study.** You have a working approach but suspect there's something better. Set a research goal in Autonomous Research Mode, point it at the RL domain, and let it run. Eight agents generate candidates, train them, score them on novelty and efficiency, and recurse on the top performers. By morning there's a ranked leaderboard. The answer to "is there something better?" used to require a dedicated research sprint. Now it runs while you sleep.

---

## It Fixes Its Own Mistakes

Code fails sometimes. That is not a problem here. When something does not compile or run correctly, the platform diagnoses what went wrong, rewrites what it needs to, and tries again — on its own, without you having to do anything. You will see it happen in the chat. By the time you look back, it has usually already fixed it.

---

## Choosing an AI Provider

The platform works with Anthropic (Claude), OpenAI (GPT-4o), or any local model via LM Studio. From testing, Claude gives the best results — it reads more thoroughly, reasons more carefully, and produces cleaner output with less fixing needed. GPT-4o is a solid second. Local models work well at larger sizes; smaller ones occasionally skip research steps or need a retry.

If privacy matters most and you want everything running locally with no API costs, use LM Studio with the largest model your hardware supports. If you want the best output with minimal friction, use Claude.

---

## Quick Start

### What you need

| Tool | Purpose |
|------|---------|
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | Runs the entire platform in containers |
| [Git](https://git-scm.com/downloads) | To clone the repository |
| An LLM API key | Anthropic, OpenAI, or a local LM Studio server |

Docker Desktop includes everything needed on Windows and macOS. On Linux, install Docker Engine and Docker Compose separately.

### Step 1 — Clone and configure

```bash
git clone https://github.com/sup3rus3r/obsidian-networks.git
cd obsidian-networks

cp .env.example .env
```

Open `.env` in any text editor and fill in two things:
- `AUTH_SECRET` — a random string used to secure sessions. Generate one by running: `openssl rand -base64 32`
- Your API key — `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `LMSTUDIO_BASE_URL` depending on which provider you're using

### Step 2 — Start the platform

```bash
docker compose up --build
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Local Development (without Docker)

If you prefer to run the stack directly on your machine — requires Linux or WSL2, Python 3.11+, Node.js 18+, and Redis installed locally:

```bash
git clone https://github.com/sup3rus3r/obsidian-networks.git
cd obsidian-networks

cd backend && pip install uv && uv sync && cd ..
cd frontend && npm install && cd ..

cp .env.example .env
npm run dev
```

This starts everything in one terminal with colour-coded output. Open [http://localhost:3000](http://localhost:3000).

---

## Environment Variables

All configuration lives in the `.env` file at the repo root.

| Variable | Default | What it does |
|----------|---------|--------------|
| `AUTH_SECRET` | — | **Required.** Secures session cookies. Generate with `openssl rand -base64 32` |
| `AI_PROVIDER` | `anthropic` | Which AI to use: `anthropic`, `openai`, or `lmstudio` |
| `AI_MODEL` | provider default | Override the specific model, e.g. `claude-sonnet-4-6` or `gpt-4o` |
| `ANTHROPIC_API_KEY` | — | Required when using Anthropic |
| `OPENAI_API_KEY` | — | Required when using OpenAI |
| `LMSTUDIO_BASE_URL` | `http://localhost:1234/v1` | Required when using LM Studio |
| `SESSION_TTL_HOURS` | `4` | How long before uploaded files and session data are automatically deleted |
| `MAX_FILE_SIZE_MB` | `500` | Maximum size of dataset uploads |
| `MAX_TRAINING_MINUTES` | `10` | How long a training job is allowed to run before being stopped |
| `MAX_MEMORY_GB` | `12` | Maximum memory the training worker can use |
| `MAX_EPOCHS` | `200` | Maximum number of training epochs any generated script can run |
| `NEXT_PUBLIC_UPLOAD_URL` | `http://localhost:8000` | The backend URL used for large file uploads. Change this to your server's public address when deploying |

---

## Sample Datasets

Two datasets are included so you can try the platform immediately after setup.

**`heart_failure_dataset.csv`** — 918 patients with 11 clinical features including age, chest pain type, blood pressure, cholesterol, and ECG results. The goal is to predict whether a patient has heart disease.

Try: *"Build a binary classifier to predict heart disease. Use a deep neural network with dropout regularisation."*

**`energy_consumption_sample.csv`** — Hourly energy consumption readings over time. The goal is to forecast future consumption.

Try: *"Forecast energy consumption for the next 24 hours using an LSTM."*

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Next.js 16, React 19, TypeScript, Tailwind CSS 4 |
| AI / Streaming | Vercel AI SDK 6, Anthropic / OpenAI / LM Studio |
| Backend API | FastAPI, Python 3.11 |
| Task Queue | Celery 5, Redis 7 |
| ML Runtime | TensorFlow 2.16+, Keras 3, NumPy, Pandas, scikit-learn, Gymnasium |
| Research | FAISS vector store, sentence-transformers, pypdf, arXiv API, Context7 |
| Research Mode | 8-agent autonomous loop, MongoDB, FAISS novelty index |
| Deployment | Docker, Docker Compose |

---

## Security

Generated scripts are checked for dangerous code before they ever run. Imports are restricted to a safe allowlist of known ML libraries. Calls to system commands, file execution, and anything that could affect your machine outside the sandbox are blocked.

Training runs happen inside an isolated subprocess with a hard time limit, a memory cap, and a file size limit. The worker container runs with additional Docker-level restrictions — blocked system calls, dropped Linux capabilities, and process limits — to prevent any generated code from escaping its sandbox.

Your uploaded files are stored in a private per-session directory that no other session can access, and everything is automatically deleted after the session expires.

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
