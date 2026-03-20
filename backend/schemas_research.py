"""
Pydantic schemas for Obsidian Networks Research Labs.
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
    domain                    : str
    category                  : str
    task_description          : str
    population_size           : int = Field(default=3, ge=1, le=20)
    max_generations           : int = Field(default=3, ge=1, le=50)
    max_gen0_retries          : int = Field(default=3, ge=1, le=20)
    enable_real_data_validation: bool = False
    real_data_path            : str | None = None


# ── Data preparation ──────────────────────────────────────────────────────────

class PrepareDataRequest(BaseModel):
    source    : DataSource
    category  : str | None = None
    subset_size: int | None = None


class DataPreparationStatus(BaseModel):
    task_id   : str
    status    : str
    progress  : float = 0.0
    message   : str
    data_path : str | None = None
    created_at: str | None = None
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
    status             : str
    domain             : str
    category           : str
    created_at         : str


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
    architecture_name: str


class CompileCandidateResponse(BaseModel):
    architecture_name: str
    code             : str
    composite_score  : float
    filename         : str
