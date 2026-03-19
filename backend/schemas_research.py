"""
Pydantic schemas for Obsidian Networks Autonomous Research Mode.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


# ── Data source models ────────────────────────────────────────────────────────

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


# ── Category models ───────────────────────────────────────────────────────────

class DatasetType(BaseModel):
    type: str
    description: str
    example_task: str
    domains: list[str]
    recommended_architectures: list[str]


class CategoryInfo(BaseModel):
    id: str
    name: str
    description: str
    dataset_types: list[DatasetType]
    data_sources: list[DataSourceInfo]


# ── Research session request ──────────────────────────────────────────────────

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
    max_depth: int = Field(default=5, ge=1, le=10)
    max_generations: int = Field(default=10, ge=1, le=50)
    population_size: int = Field(default=5, ge=1, le=20)
    max_time_mins: int = Field(default=30, ge=5, le=180)


# ── Data preparation ──────────────────────────────────────────────────────────

class PrepareDataRequest(BaseModel):
    data_source: DataSource
    subset_size: int | None = None


class DataPreparationStatus(BaseModel):
    task_id: str
    status: str                           # "downloading" | "processing" | "completed" | "failed"
    progress: float = 0.0                 # 0.0 – 1.0
    message: str
    data_path: str | None = None
    metadata: dict | None = None


# ── Scoring & validation ──────────────────────────────────────────────────────

class ValidationResult(BaseModel):
    enabled: bool = False
    real_data_source: str | None = None
    synthetic_metrics: dict = Field(default_factory=dict)
    real_metrics: dict | None = None
    loss_ratio: float | None = None       # real_loss / synthetic_loss
    generalization_score: float = 0.5    # 0–1; default 0.5 when no validation
    overfitting_detected: bool = False
    validation_time_s: float | None = None
    status: str = "pending"              # "pending" | "running" | "completed" | "failed"


class CompositeScoreBreakdown(BaseModel):
    novelty_score: float = 0.0
    efficiency_score: float = 0.0
    soundness_score: float = 0.0
    generalization_score: float = 0.5
    composite_score: float = 0.0         # 0.3×novelty + 0.2×eff + 0.2×sound + 0.3×gen


# ── Candidate ─────────────────────────────────────────────────────────────────

class CandidateResponse(BaseModel):
    candidate_id: str
    research_session_id: str
    generation: int
    depth: int
    architecture_name: str
    domain: str
    framework: str
    generated_code: str
    architecture_spec: dict
    composite_score_breakdown: CompositeScoreBreakdown
    validation: ValidationResult | None = None
    next_action: str = "archive"         # "recurse" | "archive" | "discard"
    status: str = "scored"
    created_at: str


# ── Research session ──────────────────────────────────────────────────────────

class ResearchSessionResponse(BaseModel):
    research_session_id: str
    user_session_id: str | None = None
    category_id: str
    domains: list[str]
    dataset_type: str | None = None
    base_model: str
    status: str                           # "preparing" | "running" | "completed" | "failed" | "cancelled"
    max_depth: int
    max_generations: int
    population_size: int
    current_generation: int = 0
    current_depth: int = 0
    created_at: str
    completed_at: str | None = None


# ── SSE progress events ───────────────────────────────────────────────────────

class ResearchProgressEvent(BaseModel):
    event: str                            # "agent_start" | "agent_done" | "generation_done" | "session_done" | "error"
    generation: int = 0
    depth: int = 0
    agent: str | None = None             # "researcher" | "mathematician" | "architect" | "coder" | "trainer" | "evaluator" | "validator" | "critic"
    message: str
    data: dict | None = None
    timestamp: str


# ── Compile candidate into user session ───────────────────────────────────────

class CompileCandidateRequest(BaseModel):
    user_session_id: str


class CompileCandidateResponse(BaseModel):
    user_session_id: str
    phase: str = "approved"
    architecture_name: str
    message: str
