# Obsidian Networks — Research Mode: Master Implementation Doc

> Read this file at the start of every session. It is the single source of truth for the autonomous research mode build.

---

## What We Are Building

A second system that runs **alongside** the existing upload → research → plan → build pipeline.

**Autonomous Research Mode** runs agents in the background (idle time or user-triggered) to discover novel neural architectures across 8 modalities. Discovered architectures are stored in MongoDB and users can pull any of them directly into the existing build phase — skipping research and planning.

The existing platform is untouched except for:
1. `backend/main.py` — add research router import
2. `backend/routers/platform.py` — add `POST /candidate/{id}/compile` handler (pulls discovered arch into session)

---

## Existing Codebase (DO NOT BREAK)

```
backend/
  main.py                  — FastAPI app, lifespan, auth routes, includes platform_router
  routers/platform.py      — All /platform/* endpoints (session, upload, compile, notebook, vectorstore)
  tasks.py                 — Celery worker for model training (existing, unchanged)
  vectorstore.py           — Per-session FAISS index (existing, unchanged)
  sessions.py              — In-memory session store (existing, unchanged)
  config.py / auth.py / database.py / models.py / schemas.py / crypto_utils.py / rate_limiter.py
frontend/
  app/api/chat/route.ts    — AI chat endpoint with all tools (search_arxiv, ingest_arxiv_paper, fetch_tensorflow_docs, etc.)
```

**Key constraint:** The existing `celery_app` in `tasks.py` uses broker `redis://redis:6379/0`. Research tasks use the same broker but a separate Celery app instance in `tasks_research.py`.

---

## Complete File List To Create

### Backend

```
backend/schemas_research.py
backend/agents/__init__.py
backend/agents/core.py
backend/agents/mutations.py
backend/agents/synthetic_data.py
backend/agents/category_registry.py
backend/agents/safety_checker.py
backend/agents/gpu_manager.py
backend/agents/gpu_providers/__init__.py
backend/agents/gpu_providers/runpod.py
backend/agents/gpu_providers/lambda_labs.py
backend/agents/gpu_providers/local_docker.py
backend/agents/domains/__init__.py
backend/agents/domains/base_domain.py
backend/agents/domains/vision.py
backend/agents/domains/language.py
backend/agents/domains/audio.py
backend/agents/domains/timeseries.py
backend/agents/domains/graph.py
backend/agents/domains/multimodal.py
backend/agents/domains/tabular.py
backend/agents/domains/recommendation.py
backend/agents/domains/generative.py
backend/agents/researcher.py
backend/agents/mathematician.py
backend/agents/architect.py
backend/agents/coder.py
backend/agents/trainer.py
backend/agents/evaluator.py
backend/agents/validator.py
backend/agents/critic.py
backend/routers/research.py
backend/tasks_research.py
backend/init_research_db.py
```

### Frontend (later phase)

```
frontend/app/research/page.tsx              — Entry: category selector + start form
frontend/app/research/[session_id]/page.tsx — Progress page (SSE streaming)
frontend/app/research/[session_id]/results/page.tsx — Candidates results page
frontend/components/research/CategorySelector.tsx
frontend/components/research/DataSourceSelector.tsx
frontend/components/research/ConstraintsForm.tsx
frontend/components/research/ProgressTracker.tsx
frontend/components/research/CandidateCard.tsx
```

---

## Implementation Order

### Phase 1 — Foundation
1. `schemas_research.py`
2. `agents/__init__.py` (empty)
3. `agents/core.py`
4. `agents/safety_checker.py`

### Phase 2 — Data Layer
5. `agents/mutations.py`
6. `agents/synthetic_data.py`
7. `agents/category_registry.py`

### Phase 3 — Domain Handlers
8. `agents/domains/__init__.py` (empty)
9. `agents/domains/base_domain.py`
10. `agents/domains/vision.py`
11. `agents/domains/language.py`
12. `agents/domains/audio.py`
13. `agents/domains/timeseries.py`
14. `agents/domains/graph.py`
15. `agents/domains/multimodal.py`
16. `agents/domains/tabular.py`
17. `agents/domains/recommendation.py`
18. `agents/domains/generative.py`

### Phase 4 — GPU Layer
19. `agents/gpu_providers/__init__.py`
20. `agents/gpu_providers/runpod.py`
21. `agents/gpu_providers/lambda_labs.py`
22. `agents/gpu_providers/local_docker.py`
23. `agents/gpu_manager.py`

### Phase 5 — Agents (7 + validator)
24. `agents/researcher.py`
25. `agents/mathematician.py`
26. `agents/architect.py`
27. `agents/coder.py`
28. `agents/trainer.py`
29. `agents/evaluator.py`
30. `agents/validator.py`
31. `agents/critic.py`

### Phase 6 — API + Celery
32. `routers/research.py`
33. `tasks_research.py`
34. `init_research_db.py`
35. Wire into `main.py`
36. Update `pyproject.toml`

### Phase 7 — Frontend
37–43. React components + pages

---

## Data Models (schemas_research.py)

```python
# All Pydantic v2 models

class DataSourceParams(BaseModel):
    size: int | None = None
    resolution: int | None = None
    split: str | None = "train"          # "train" | "test" | "validation"
    subset_size: int | None = None
    language: str | None = None
    seed: int | None = 42

class DataSource(BaseModel):
    type: str                             # "synthetic" | "api" | "upload"
    generator: str | None = None         # e.g. "random_noise"
    api_name: str | None = None          # e.g. "huggingface_cifar10"
    upload_path: str | None = None
    params: DataSourceParams | None = None

class DatasetType(BaseModel):
    type: str
    description: str
    example_task: str
    domains: list[str]
    recommended_architectures: list[str]

class DataSourceInfo(BaseModel):
    id: str
    type: str
    name: str
    description: str
    generator: str | None = None
    api_name: str | None = None
    pros: list[str]
    cons: list[str]
    metadata: dict | None = None

class CategoryInfo(BaseModel):
    id: str
    name: str
    description: str
    dataset_types: list[DatasetType]
    data_sources: list[DataSourceInfo]

class ResearchModeWithCategoryRequest(BaseModel):
    category_id: str
    dataset_type: str | None = None
    data_source: DataSource
    base_model: str = "claude-sonnet-4-6"
    auto_select_domains: bool = True
    preferred_domains: list[str] | None = None
    enable_real_data_validation: bool = False
    real_data_source: DataSource | None = None
    real_data_split: str | None = "test"
    real_data_size: int | None = 1000
    max_depth: int = 5
    max_generations: int = 10
    population_size: int = 5
    max_time_mins: int = 30

class PrepareDataRequest(BaseModel):
    data_source: DataSource
    subset_size: int | None = None

class DataPreparationStatus(BaseModel):
    task_id: str
    status: str                           # "downloading" | "processing" | "completed" | "failed"
    progress: float                       # 0.0 – 1.0
    message: str
    data_path: str | None = None
    metadata: dict | None = None

class ValidationResult(BaseModel):
    enabled: bool
    real_data_source: str | None = None
    synthetic_metrics: dict
    real_metrics: dict | None = None
    loss_ratio: float | None = None       # real_loss / synthetic_loss
    generalization_score: float = 0.5    # 0–1
    overfitting_detected: bool = False
    status: str = "pending"

class CompositeScoreBreakdown(BaseModel):
    novelty_score: float
    efficiency_score: float
    soundness_score: float
    generalization_score: float
    composite_score: float                # 0.3×novelty + 0.2×eff + 0.2×sound + 0.3×gen

class CandidateResponse(BaseModel):
    candidate_id: str
    session_id: str
    generation: int
    depth: int
    architecture_name: str
    domain: str
    framework: str
    generated_code: str
    architecture_spec: dict
    composite_score_breakdown: CompositeScoreBreakdown
    validation: ValidationResult | None
    status: str
    created_at: str

class ResearchSessionResponse(BaseModel):
    research_session_id: str
    user_session_id: str
    category_id: str
    domains: list[str]
    status: str
    max_depth: int
    max_generations: int
    created_at: str

class ResearchProgressEvent(BaseModel):
    event: str                            # "agent_start" | "agent_done" | "generation_done" | "session_done" | "error"
    generation: int
    depth: int
    agent: str | None = None
    message: str
    data: dict | None = None
    timestamp: str
```

---

## Category Registry (category_registry.py)

```python
CATEGORY_TO_DOMAINS = {
    "text":                    ["language", "generative"],
    "vision":                  ["vision", "generative"],
    "audio":                   ["audio"],
    "timeseries":              ["timeseries"],
    "graph":                   ["graph"],
    "multimodal_text_image":   ["multimodal", "vision", "language"],
    "tabular":                 ["tabular", "probabilistic"],
    "recommendation":          ["recommendation", "graph"],
}

CATEGORY_TO_DEFAULT_ARCHITECTURES = {
    "text":                    ["transformer", "lstm"],
    "vision":                  ["cnn", "vit"],
    "audio":                   ["conformer", "cnn"],
    "timeseries":              ["lstm", "transformer_ts"],
    "graph":                   ["gcn", "gat"],
    "multimodal_text_image":   ["clip", "flamingo"],
    "tabular":                 ["mlp", "resnet_tabular"],
    "recommendation":          ["embedding_cf", "attention_rec"],
}

# Full DATASET_CATEGORIES dict with all 8 categories, their dataset_types,
# synthetic_generators, and public_apis — see spec doc for full list.
```

---

## Agent Context Dict (passed between all agents)

```python
context = {
    # Set at session start
    "research_session_id": str,
    "user_session_id": str,
    "generation": int,
    "depth": int,
    "domain": str,
    "category_id": str,
    "dataset_type": str,
    "base_model": str,              # Claude model ID
    "dataset_path": str,            # path to prepared synthetic/real data

    # Set by ResearcherAgent
    "research_papers": [
        {"title": str, "arxiv_id": str, "abstract": str, "url": str}
    ],
    "research_insights": str,       # LLM-extracted key insights

    # Set by MathematicianAgent
    "candidate_mechanisms": [
        {"name": str, "description": str, "sympy_expression": str}
    ],

    # Set by ArchitectAgent
    "architecture_proposals": [
        {
            "architecture_name": str,
            "base_template": str,
            "mutations": [str],
            "spec": dict,
            "rationale": str,
        }
    ],

    # Set by CoderAgent
    "generated_code": [
        {
            "architecture_name": str,
            "code": str,
            "framework": str,       # "tensorflow" | "pytorch"
            "param_count": int,
        }
    ],

    # Set by TrainerAgent
    "training_results": [
        {
            "architecture_name": str,
            "final_loss": float,
            "accuracy": float | None,
            "checkpoint_path": str,
            "training_time_s": float,
            "training_location": str,   # "runpod" | "local" | "cpu"
            "status": str,
        }
    ],

    # Set by EvaluatorAgent
    "evaluation_results": [
        {
            "architecture_name": str,
            "synthetic_metrics": dict,
            "memory_mb": float,
            "inference_time_ms": float,
            "param_count": int,
        }
    ],

    # Set by ValidatorAgent (optional)
    "validation_results": [
        {
            "architecture_name": str,
            "real_metrics": dict,
            "loss_ratio": float,
            "generalization_score": float,
            "overfitting_detected": bool,
        }
    ],

    # Set by CriticAgent
    "scored_candidates": [
        {
            "architecture_name": str,
            "composite_score": float,
            "breakdown": dict,
            "next_action": str,     # "recurse" | "archive" | "discard"
        }
    ],
}
```

---

## BaseAgent Interface (agents/core.py)

```python
class BaseAgent(ABC):
    def __init__(self, research_session_id: str, base_model: str, domain: str):
        self.research_session_id = research_session_id
        self.base_model = base_model
        self.domain = domain
        self.logger = logging.getLogger(f"agents.{self.__class__.__name__}")
        self.llm_cache: dict[str, str] = {}
        self.use_local_llm = os.getenv("USE_LOCAL_LLM", "false") == "true"

    @abstractmethod
    async def run(self, context: dict) -> dict:
        """Execute agent logic. Receives context, returns updated context."""
        pass

    async def call_llm(self, prompt: str, cache_key: str | None = None) -> str:
        """Route to local Ollama (cheap) or Claude API (complex reasoning)."""

    async def _call_claude(self, prompt: str) -> str:
        """Call Anthropic API with base_model."""

    async def _call_local(self, prompt: str) -> str:
        """Call Ollama (fallback to Claude if unavailable)."""

    def log_step(self, message: str, data: dict | None = None) -> None:
        """Structured log with session + agent context."""

    async def emit_progress(self, event: str, message: str, data: dict | None = None) -> None:
        """Push SSE event to Redis pub/sub channel for this research_session_id."""
```

---

## BaseDomain Interface (agents/domains/base_domain.py)

```python
class BaseDomain(ABC):
    name: str
    supported_architectures: list[str]
    base_templates: dict[str, dict]     # arch_name → spec template
    mutation_operators: list[str]
    metrics: list[str]

    @abstractmethod
    async def generate_mechanism(self, research_insights: str, llm_caller) -> list[dict]:
        """Return list of novel mechanisms derived from research."""

    @abstractmethod
    async def propose_mutations(self, base_arch: str, mechanisms: list[dict], llm_caller) -> list[dict]:
        """Return list of architecture mutation proposals."""

    @abstractmethod
    async def generate_code(self, arch_spec: dict, llm_caller) -> str:
        """Return executable Python training code for the architecture."""

    @abstractmethod
    def generate_synthetic_data(self, size: int, params: dict) -> tuple:
        """Return (X_train, X_test, y_train, y_test) or equivalent."""

    @abstractmethod
    async def evaluate(self, checkpoint_path: str, test_data: tuple) -> dict:
        """Return domain-specific evaluation metrics dict."""
```

---

## Domain Handlers Summary

| Domain | Base Templates | Key Metrics | Synthetic Data |
|--------|---------------|-------------|----------------|
| vision | cnn, vit | accuracy, top5_acc, memory_mb, inference_ms | tf.random.normal (N, H, W, 3) |
| language | transformer, lstm | loss, perplexity, memory_mb | random token sequences |
| audio | conformer, cnn | loss, cer, memory_mb | librosa sine waves |
| timeseries | lstm, transformer_ts | mse, mae, memory_mb | arima/random walk |
| graph | gcn, gat | node_acc, link_auc, memory_mb | networkx random graphs |
| multimodal | clip, flamingo | contrastive_loss, cross_modal_acc | paired image+text tensors |
| tabular | mlp, resnet_tabular | auc, rmse, memory_mb | sklearn make_classification |
| recommendation | embedding_cf, attention_rec | ndcg, hit_rate, memory_mb | random ratings matrix |
| generative | gan, vae, diffusion | fid_score, inception_score, loss | noise tensors |

---

## API Endpoints (routers/research.py)

```
PREFIX: /platform/research

GET  /categories                              → list all 8 categories
GET  /categories/{category_id}               → single category detail
GET  /categories/{category_id}/data-sources  → data sources for category

POST /start                                   → ResearchModeWithCategoryRequest → ResearchSessionResponse
     - Creates research session in MongoDB
     - Queues prepare_dataset_task (Celery)
     - Queues run_research_generation.delay(session_id, generation=0, depth=0)

GET  /progress/{research_session_id}          → SSE stream (text/event-stream)
     - Subscribes to Redis pub/sub channel research:{research_session_id}
     - Streams ResearchProgressEvent as JSON

GET  /sessions                                → list all research sessions
GET  /sessions/{research_session_id}          → session detail + status

GET  /candidates/{research_session_id}        → list candidates sorted by composite_score desc
GET  /candidate/{candidate_id}               → full candidate detail incl. generated_code

POST /candidate/{candidate_id}/compile        → pull into existing user session
     - Body: { "user_session_id": str }
     - Sets session.phase = "approved"
     - Sets session.plan_doc = architecture_spec as markdown
     - Returns { "user_session_id": str, "phase": "approved" }

POST /auto-start                              → admin: spawn idle research session
DELETE /sessions/{research_session_id}        → cancel + delete session
```

---

## Celery Tasks (tasks_research.py)

```python
# Separate celery_app instance, same Redis broker
research_celery_app = Celery("obsidian_research", broker=REDIS_URL, backend=REDIS_URL)

@research_celery_app.task
def prepare_dataset_task(research_session_id: str, data_source: dict) -> dict:
    """Download or generate dataset. Emits SSE progress. Returns data_path."""

@research_celery_app.task
def prepare_real_data_task(research_session_id: str, real_data_source: dict) -> dict:
    """Download real validation data. Returns real_data_path."""

@research_celery_app.task
def run_research_generation(research_session_id: str, generation: int, depth: int) -> dict:
    """
    Orchestrate 7-8 agent loop for one generation.
    1. Load session config from MongoDB
    2. Build context dict
    3. Run agents sequentially: researcher → mathematician → architect → coder → trainer → evaluator → [validator] → critic
    4. Emit SSE after each agent
    5. Save scored candidates to MongoDB
    6. If candidates_to_recurse and depth < MAX_DEPTH:
         run_research_generation.delay(session_id, generation+1, depth+1)
    7. Else: mark session complete
    """

@research_celery_app.task
def check_idle_and_spawn() -> None:
    """Check if system is idle. If so, spawn a research session automatically."""
```

---

## MongoDB Collections

### Collection: `research_sessions`
```json
{
  "_id": ObjectId,
  "research_session_id": "uuid",
  "user_session_id": "uuid | null",
  "category_id": "vision",
  "domains": ["vision"],
  "dataset_type": "image_classification",
  "data_source": {},
  "base_model": "claude-sonnet-4-6",
  "status": "running | completed | failed | cancelled",
  "max_depth": 5,
  "max_generations": 10,
  "population_size": 5,
  "max_time_mins": 30,
  "current_generation": 0,
  "current_depth": 0,
  "dataset_path": "/research_artifacts/session_id/data/",
  "real_data_path": null,
  "enable_real_data_validation": false,
  "created_at": "ISO datetime",
  "completed_at": null
}
```

### Collection: `research_candidates`
```json
{
  "_id": ObjectId,
  "candidate_id": "uuid",
  "research_session_id": "uuid",
  "generation": 0,
  "depth": 0,
  "architecture_name": "ViT_adaptive_attention",
  "domain": "vision",
  "framework": "tensorflow",
  "generated_code": "import tensorflow as tf...",
  "architecture_spec": {},
  "composite_score_breakdown": {
    "novelty_score": 0.82,
    "efficiency_score": 0.71,
    "soundness_score": 0.78,
    "generalization_score": 0.65,
    "composite_score": 0.745
  },
  "validation": {
    "enabled": false,
    "synthetic_metrics": {"loss": 0.45, "accuracy": 0.91},
    "real_metrics": null,
    "loss_ratio": null,
    "generalization_score": 0.5,
    "overfitting_detected": false,
    "status": "pending"
  },
  "next_action": "archive",
  "status": "scored",
  "created_at": "ISO datetime"
}
```

---

## GPU Provider Strategy

```
Priority order (cheapest first):
1. local_docker  — free, instant, limited to 1-2 parallel jobs
2. runpod        — $0.20/hr RTX4090, ~10s cold start
3. lambda_labs   — $0.60/hr A100, ~60s cold start
4. cpu_fallback  — always available, slow

Env vars:
USE_SERVERLESS_GPU=true
GPU_PROVIDER=runpod
GPU_BUDGET=10.0
RUNPOD_API_KEY=...
LAMBDA_API_KEY=...
USE_LOCAL_LLM=false
LOCAL_LLM_MODEL=mistral
OLLAMA_BASE_URL=http://localhost:11434
```

---

## Composite Scoring Formula

```
composite_score = 0.3×novelty + 0.2×efficiency + 0.2×soundness + 0.3×generalization

novelty_score:        FAISS embedding search vs all archived candidates (1 = most novel)
efficiency_score:     normalize(1/memory_mb, 1/param_count, 1/inference_ms)
soundness_score:      LLM judge rates architecture validity 0–1
generalization_score: max(0, 1 - (loss_ratio - 1) / 1.0) clamped [0,1]; default 0.5 if no validation

Thresholds:
  composite > 0.75 → "recurse" (spawn next generation with this as base)
  composite > 0.50 → "archive" (save to MongoDB for user browsing)
  composite ≤ 0.50 → "discard"
```

---

## New Dependencies (pyproject.toml additions)

```toml
"sympy>=1.12",
"scipy>=1.13.0",
"torch>=2.1.0",
"torchvision>=0.16.0",
"torchaudio>=2.1.0",
"timm>=0.9.12",
"albumentations>=1.4.0",
"scikit-image>=0.22.0",
"diffusers>=0.27.0",
"transformers>=4.36.0",
"pillow>=10.0.0",
"networkx>=3.2",
"librosa>=0.10.0",
"soundfile>=0.12.1",
"pymc>=5.10.0",
"faker>=21.0.0",
"tqdm>=4.66.0",
"arxiv>=2.0.0",
"faiss-cpu>=1.8.0",
"pymongo>=4.6.0",
"motor>=3.3.0",
"docker>=7.0.0",
"ollama>=0.1.7",
"torch-geometric>=2.5.0",
```

---

## Key Implementation Rules

1. **Never break the existing pipeline.** Research mode is additive only.
2. **All agents are async.** Use `asyncio` throughout.
3. **MongoDB via Motor** (async driver) — same pattern as existing `database_mongo.py`.
4. **SSE via Redis pub/sub** — agents emit to channel `research:{research_session_id}`, the `/progress/` endpoint subscribes and streams.
5. **Generated code safety** — always run `SafetyChecker.validate_code()` before executing any LLM-generated code.
6. **GPU training is optional** — if `USE_SERVERLESS_GPU=false` or all providers fail, fall back to CPU training locally.
7. **LLM routing** — use local Ollama for: extract, classify, summarize, parse, rate, score, compare. Use Claude API for: mathematician (derive mechanisms), architect (propose mutations), critic (soundness rating).
8. **arXiv ingestion** — researcher agent uses the `arxiv` Python library (not the XML API) for reliable paper fetching. Downloads PDFs to `/research_artifacts/{session_id}/papers/`.
9. **Synthetic data** — stored at `/research_artifacts/{session_id}/data/`. Each domain generates its own format.
10. **Checkpoints** — stored at `/research_artifacts/{session_id}/checkpoints/{candidate_id}/`.

---

## Progress Tracking (this doc)

- [ ] Phase 1: schemas_research.py + agents/core.py + safety_checker.py
- [ ] Phase 2: mutations.py + synthetic_data.py + category_registry.py
- [ ] Phase 3: All 9 domain handlers
- [ ] Phase 4: GPU providers + gpu_manager.py
- [ ] Phase 5: All 8 agents (researcher → critic)
- [ ] Phase 6: routers/research.py + tasks_research.py + init_research_db.py + wire main.py + update pyproject.toml
- [ ] Phase 7: Frontend components + pages
