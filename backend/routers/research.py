"""
Research Mode router — all /platform/research/* endpoints.

Endpoints:
  POST   /platform/research/start           — start a research session
  GET    /platform/research/{id}/status     — poll session status
  GET    /platform/research/{id}/stream     — SSE stream of agent progress
  GET    /platform/research/{id}/candidates — list scored candidates
  GET    /platform/research/{id}/candidate/{arch} — get single candidate detail
  POST   /platform/research/{id}/compile    — compile a candidate to production code
  DELETE /platform/research/{id}            — cancel / delete session
  GET    /platform/research/categories      — list all dataset categories
  POST   /platform/research/prepare-data    — prepare / validate a dataset source
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import AsyncIterator

from fastapi import APIRouter, Cookie, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from schemas_research import (
    ResearchModeWithCategoryRequest,
    PrepareDataRequest,
    DataPreparationStatus,
    ResearchSessionResponse,
    CandidateResponse,
    CompileCandidateRequest,
    CompileCandidateResponse,
)
from sessions import get_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/platform/research", tags=["research"])

REDIS_URL     = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
ARTIFACTS_DIR = os.environ.get("RESEARCH_ARTIFACTS_DIR", "/research_artifacts")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_redis():
    import redis as _redis
    return _redis.from_url(REDIS_URL, decode_responses=True)


async def _get_mongo():
    from database_mongo import get_database
    return get_database()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _session_or_404(session_id: str):
    from sessions import get_session as _gs
    s = _gs(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return s


# ── Categories ────────────────────────────────────────────────────────────────

@router.get("/categories")
async def list_categories():
    """Return all supported dataset categories with metadata."""
    from agents.category_registry import get_all_categories
    cats = get_all_categories()
    return {"categories": cats}


# ── Data preparation ──────────────────────────────────────────────────────────

@router.post("/prepare-data", response_model=DataPreparationStatus)
async def prepare_data(
    req: PrepareDataRequest,
    obs_session_id: str | None = Cookie(default=None),
):
    """Validate and index a dataset source (URL, HuggingFace, upload path, or synthetic)."""

    from tasks_research import prepare_dataset_task
    task = prepare_dataset_task.delay(
        session_id  = obs_session_id,
        source      = req.source.model_dump(),
        category    = req.category,
    )

    return DataPreparationStatus(
        task_id    = task.id,
        status     = "queued",
        message    = "Dataset preparation queued",
        created_at = _now_iso(),
    )


# ── Start research session ────────────────────────────────────────────────────

@router.post("/start", response_model=ResearchSessionResponse)
async def start_research(
    req: ResearchModeWithCategoryRequest,
    obs_session_id: str | None = Cookie(default=None),
):
    """Launch an autonomous research session."""

    research_id = str(uuid.uuid4())

    # Persist initial session document to MongoDB
    try:
        db = await _get_mongo()
        await db["research_sessions"].insert_one({
            "_id"              : research_id,
            "session_id"       : obs_session_id,
            "status"           : "queued",
            "request"          : req.model_dump(),
            "created_at"       : _now_iso(),
            "generation"       : 0,
            "scored_candidates": [],
        })
    except Exception as e:
        logger.warning("MongoDB unavailable, continuing without persistence: %s", e)

    # Dispatch Celery task
    from tasks_research import run_research_generation
    run_research_generation.apply_async(
        kwargs = dict(
            research_session_id = research_id,
            context             = {
                "session_id"               : obs_session_id,
                "research_session_id"      : research_id,
                "domain"                   : req.domain,
                "category"                 : req.category,
                "task_description"         : req.task_description,
                "population_size"          : req.population_size,
                "max_generations"          : req.max_generations,
                "enable_real_data_validation": req.enable_real_data_validation,
                "real_data_path"           : req.real_data_path,
                "generation"               : 0,
                "depth"                    : 0,
            },
        ),
        queue = "research",
    )

    return ResearchSessionResponse(
        research_session_id = research_id,
        status              = "queued",
        domain              = req.domain,
        category            = req.category,
        created_at          = _now_iso(),
    )


# ── Session status ────────────────────────────────────────────────────────────

@router.get("/{research_id}/status")
async def get_research_status(research_id: str):
    """Poll the current status of a research session."""
    try:
        db  = await _get_mongo()
        doc = await db["research_sessions"].find_one({"_id": research_id})
        if not doc:
            raise HTTPException(status_code=404, detail="Research session not found")
        doc.pop("_id", None)
        return doc
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e}")


# ── SSE stream ────────────────────────────────────────────────────────────────

@router.get("/{research_id}/stream")
async def stream_research_progress(research_id: str, request: Request):
    """
    SSE endpoint — streams agent progress events published to Redis pub/sub.
    Channel: research:{research_id}
    """
    async def _event_generator() -> AsyncIterator[str]:
        import redis.asyncio as aioredis

        r = aioredis.from_url(REDIS_URL, decode_responses=True)
        channel = f"research:{research_id}"
        pubsub  = r.pubsub()
        await pubsub.subscribe(channel)

        try:
            # Send a connected ping
            yield f"event: connected\ndata: {json.dumps({'research_session_id': research_id})}\n\n"

            async for message in pubsub.listen():
                if await request.is_disconnected():
                    break
                if message["type"] != "message":
                    continue
                try:
                    data = json.loads(message["data"])
                except (json.JSONDecodeError, TypeError):
                    continue

                yield f"event: progress\ndata: {json.dumps(data)}\n\n"

                if data.get("event_type") in ("session_complete", "session_error"):
                    break

        finally:
            await pubsub.unsubscribe(channel)
            await r.aclose()

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control"    : "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Candidates ────────────────────────────────────────────────────────────────

@router.get("/{research_id}/candidates")
async def list_candidates(research_id: str):
    """Return all scored candidates for a research session, sorted by composite score."""
    try:
        db   = await _get_mongo()
        docs = await db["research_candidates"].find(
            {"research_session_id": research_id}
        ).sort("composite_score", -1).to_list(length=100)
        for d in docs:
            d.pop("_id", None)
        return {"research_session_id": research_id, "candidates": docs}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e}")


@router.get("/{research_id}/candidate/{arch_name}")
async def get_candidate(research_id: str, arch_name: str):
    """Return full detail for a single candidate including generated code."""
    try:
        db  = await _get_mongo()
        doc = await db["research_candidates"].find_one({
            "research_session_id": research_id,
            "architecture_name"  : arch_name,
        })
        if not doc:
            raise HTTPException(status_code=404, detail="Candidate not found")
        doc.pop("_id", None)
        return doc
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e}")


# ── Compile candidate ─────────────────────────────────────────────────────────

@router.post("/{research_id}/compile", response_model=CompileCandidateResponse)
async def compile_candidate(research_id: str, req: CompileCandidateRequest):
    """
    Export a candidate architecture as a production-ready training script.
    Returns the cleaned, annotated Python code for download.
    """
    try:
        db  = await _get_mongo()
        doc = await db["research_candidates"].find_one({
            "research_session_id": research_id,
            "architecture_name"  : req.architecture_name,
        })
        if not doc:
            raise HTTPException(status_code=404, detail="Candidate not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e}")

    raw_code = doc.get("code", "")
    if not raw_code:
        raise HTTPException(status_code=422, detail="No code available for this candidate")

    # Build a clean production script
    header = f"""# =============================================================================
# Architecture : {req.architecture_name}
# Research ID  : {research_id}
# Domain       : {doc.get('domain', 'unknown')}
# Composite    : {doc.get('composite_score', 0):.4f}
# Generated    : {_now_iso()}
# =============================================================================
"""
    production_code = header + raw_code

    return CompileCandidateResponse(
        architecture_name = req.architecture_name,
        code              = production_code,
        composite_score   = doc.get("composite_score", 0.0),
        filename          = f"{req.architecture_name.replace(' ', '_')}.py",
    )


# ── Continue after user decision ──────────────────────────────────────────────

@router.post("/{research_id}/continue")
async def continue_research(research_id: str):
    """
    Resume a session that is paused at 'awaiting_decision'.
    Resets the consecutive-failure counter and re-queues the next generation
    using the pending_context stored when the session paused.
    """
    try:
        db  = await _get_mongo()
        doc = await db["research_sessions"].find_one({"_id": research_id})
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e}")

    if not doc:
        raise HTTPException(status_code=404, detail="Research session not found")
    if doc.get("status") != "awaiting_decision":
        raise HTTPException(status_code=409, detail=f"Session is not awaiting a decision (status: {doc.get('status')})")

    pending_context = doc.get("pending_context")
    if not pending_context:
        raise HTTPException(status_code=422, detail="No pending context stored — cannot resume")

    # Reset the failure counter so the next stretch gets a clean slate
    pending_context["consecutive_improvement_attempts"] = 0

    try:
        await db["research_sessions"].update_one(
            {"_id": research_id},
            {"$set": {"status": "running", "pending_context": None, "updated_at": _now_iso()}},
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database update failed: {e}")

    from tasks_research import run_research_generation
    run_research_generation.apply_async(
        kwargs = dict(research_session_id=research_id, context=pending_context),
        queue  = "research",
    )

    try:
        r = _get_redis()
        r.publish(
            f"research:{research_id}",
            json.dumps({
                "event_type"         : "session_resumed",
                "research_session_id": research_id,
                "generation"         : pending_context.get("generation", 0),
                "message"            : "Research resumed — exploring with new mathematical mechanisms",
                "timestamp"          : _now_iso(),
            }),
        )
        r.close()
    except Exception:
        pass

    return {"message": "Research resumed", "research_session_id": research_id}


# ── Cancel session ────────────────────────────────────────────────────────────

@router.delete("/{research_id}")
async def cancel_research_session(research_id: str):
    """Cancel a running research session and mark it as cancelled in MongoDB."""
    try:
        r = _get_redis()
        # Set the cancel key the worker polls between agents
        r.set(f"research:cancel:{research_id}", "1", ex=3600)
        # Also publish to SSE so the frontend gets the cancelled event immediately
        r.publish(
            f"research:{research_id}",
            json.dumps({"event_type": "session_cancelled", "research_session_id": research_id,
                        "timestamp": _now_iso()}),
        )
        r.close()
    except Exception:
        pass

    try:
        db = await _get_mongo()
        await db["research_sessions"].update_one(
            {"_id": research_id},
            {"$set": {"status": "cancelled", "cancelled_at": _now_iso()}},
        )
    except Exception:
        pass

    return {"message": "Cancellation signal sent", "research_session_id": research_id}
