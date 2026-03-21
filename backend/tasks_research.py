"""
Celery tasks for Research Labs.

Tasks:
  prepare_dataset_task      — validate + index a dataset source
  prepare_real_data_task    — prepare real validation data (npz)
  run_research_generation   — main research loop (one generation)
  check_idle_and_spawn      — periodic task: resume idle sessions
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from celery import Celery

logger = logging.getLogger(__name__)

REDIS_URL     = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
ARTIFACTS_DIR = Path(os.environ.get("RESEARCH_ARTIFACTS_DIR", "/research_artifacts"))

# Separate Celery app so research tasks don't conflict with existing compilation tasks
research_celery_app = Celery(
    "research_tasks",
    broker  = REDIS_URL,
    backend = REDIS_URL,
)
research_celery_app.conf.update(
    task_serializer            = "json",
    result_serializer          = "json",
    accept_content             = ["json"],
    task_track_started         = True,
    task_acks_late             = True,
    worker_prefetch_multiplier = 1,
    task_default_queue         = "research",
    beat_schedule = {
        "check-idle-sessions": {
            "task"    : "tasks_research.check_idle_and_spawn",
            "schedule": 60.0,
            "options" : {"queue": "research"},
        },
    },
)


MAX_IMPROVEMENT_ATTEMPTS = 3   # consecutive non-recurse gens before pausing for user input
GEN0_GRADUATE_THRESHOLD  = 0.50  # composite score that earns gen 0 a promotion to gen 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _publish(research_id: str, event: dict):
    """Publish a progress event to Redis pub/sub."""
    try:
        import redis as _redis
        r = _redis.from_url(REDIS_URL, decode_responses=True)
        r.publish(f"research:{research_id}", json.dumps(event))
        r.close()
    except Exception as e:
        logger.warning("Redis publish failed: %s", e)


async def _update_mongo_session(research_id: str, updates: dict):
    """Update a research session document in MongoDB."""
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        mongo_url = os.environ.get("MONGO_URL", "mongodb://mongo:27017")
        db_name   = os.environ.get("MONGO_DB_NAME", "obsidian")
        client = AsyncIOMotorClient(mongo_url)
        db = client[db_name]
        await db["research_sessions"].update_one(
            {"_id": research_id},
            {"$set": updates},
        )
        client.close()
    except Exception as e:
        logger.warning("MongoDB update failed: %s", e)


async def _save_candidates(
    research_id: str,
    scored_candidates: list[dict],
    generated_code: list[dict],
    generation: int = 0,
    research_papers: list[dict] | None = None,
    mechanisms: list[dict] | None = None,
):
    """Upsert scored candidates into MongoDB."""
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        mongo_url = os.environ.get("MONGO_URL", "mongodb://mongo:27017")
        db_name   = os.environ.get("MONGO_DB_NAME", "obsidian")
        client = AsyncIOMotorClient(mongo_url)
        db = client[db_name]

        # Build full code/metadata lookup
        code_meta = {g["architecture_name"]: g for g in generated_code}

        for candidate in scored_candidates:
            arch = candidate["architecture_name"]
            meta = code_meta.get(arch, {})
            doc  = {
                "research_session_id": research_id,
                "architecture_name"  : arch,
                "generation"         : generation,
                "composite_score"    : candidate.get("composite_score", 0),
                "novelty_score"      : candidate.get("novelty_score", 0),
                "efficiency_score"   : candidate.get("efficiency_score", 0),
                "soundness_score"    : candidate.get("soundness_score", 0),
                "generalization_score": candidate.get("generalization_score", 0),
                "next_action"        : candidate.get("next_action", "discard"),
                "synthetic_metrics"  : candidate.get("synthetic_metrics", {}),
                "validation"         : candidate.get("validation"),
                "memory_mb"          : candidate.get("memory_mb", 0),
                "inference_time_ms"  : candidate.get("inference_time_ms", 0),
                "param_count"        : candidate.get("param_count", 0),
                "code"               : meta.get("code", ""),
                "base_template"      : meta.get("base_template", ""),
                "mutations"          : meta.get("mutations", []),
                "rationale"          : meta.get("rationale", ""),
                "research_papers"    : [
                    {"title": p.get("title", ""), "arxiv_id": p.get("arxiv_id", ""), "abstract": p.get("abstract", "")[:400]}
                    for p in (research_papers or [])
                ],
                "mechanisms"         : [
                    {"name": m.get("name", ""), "description": m.get("description", ""), "sympy_expression": m.get("sympy_expression", ""), "sympy_valid": m.get("sympy_valid", False)}
                    for m in (mechanisms or [])
                ],
                "updated_at"         : _now_iso(),
            }
            await db["research_candidates"].update_one(
                {"research_session_id": research_id, "architecture_name": arch},
                {"$set": doc},
                upsert=True,
            )
        client.close()
    except Exception as e:
        logger.warning("Failed to save candidates to MongoDB: %s", e)


# ── Dataset preparation ───────────────────────────────────────────────────────

@research_celery_app.task(name="tasks_research.prepare_dataset_task", bind=True)
def prepare_dataset_task(self, session_id: str, source: dict, category: str):
    """Validate a dataset source and return summary info."""
    import asyncio

    async def _run():
        source_type = source.get("type", "synthetic")

        if source_type == "synthetic":
            from agents.synthetic_data import get_synthetic_data
            data = get_synthetic_data(category, size=100, params={})
            return {
                "status"     : "ready",
                "source_type": "synthetic",
                "category"   : category,
                "sample_size": 100,
                "message"    : "Synthetic data ready",
            }

        if source_type == "upload":
            path = source.get("path", "")
            if not Path(path).exists():
                return {"status": "error", "message": f"File not found: {path}"}
            return {"status": "ready", "source_type": "upload", "path": path}

        if source_type == "huggingface":
            return {
                "status"     : "ready",
                "source_type": "huggingface",
                "dataset_id" : source.get("dataset_id", ""),
                "message"    : "HuggingFace dataset will be streamed at training time",
            }

        return {"status": "error", "message": f"Unknown source type: {source_type}"}

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run())
    finally:
        loop.close()


@research_celery_app.task(name="tasks_research.prepare_real_data_task", bind=True)
def prepare_real_data_task(self, session_id: str, real_data_path: str):
    """Validate that the real data npz file is readable."""
    path = Path(real_data_path)
    if not path.exists():
        return {"status": "error", "message": f"Real data not found: {real_data_path}"}
    if path.suffix != ".npz":
        return {"status": "error", "message": "Only .npz format supported for real data"}

    try:
        import numpy as np
        data = np.load(path)
        keys = list(data.keys())
        return {
            "status" : "ready",
            "path"   : str(path),
            "keys"   : keys,
            "message": "Real data validated successfully",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── Main research generation task ─────────────────────────────────────────────

@research_celery_app.task(name="tasks_research.run_research_generation", bind=True, max_retries=0)
def run_research_generation(self, research_session_id: str, context: dict):
    """
    Execute one full generation of the research loop:
      Researcher → Mathematician → Architect → Coder → Trainer → Evaluator → Validator → Critic

    On completion:
      - Saves candidates to MongoDB
      - If any candidates scored > RECURSE_THRESHOLD and generation < max_generations → recurse
      - Publishes session_complete or session_error to Redis
    """
    import asyncio

    async def _run():
        research_id   = research_session_id
        generation    = context.get("generation", 0)
        max_gen       = context.get("max_generations", 3)
        session_id    = context.get("session_id", "")

        logger.info("Research generation %d starting for session %s", generation, research_id)

        _publish(research_id, {
            "event_type"          : "generation_start",
            "research_session_id" : research_id,
            "generation"          : generation,
            "timestamp"           : _now_iso(),
        })

        # Build agents with shared config
        agent_kwargs = {
            "research_session_id" : research_id,
            "base_model"          : context.get("base_model") or os.environ.get("AI_MODEL") or "claude-sonnet-4-6",
            "domain"              : context.get("domain", "vision"),
        }

        try:
            from agents.researcher     import ResearcherAgent
            from agents.mathematician  import MathematicianAgent
            from agents.architect      import ArchitectAgent
            from agents.coder          import CoderAgent
            from agents.code_validator import CodeValidatorAgent
            from agents.trainer        import TrainerAgent
            from agents.evaluator      import EvaluatorAgent
            from agents.validator      import ValidatorAgent
            from agents.critic         import CriticAgent

            pipeline = [
                ResearcherAgent(**agent_kwargs),
                MathematicianAgent(**agent_kwargs),
                ArchitectAgent(**agent_kwargs),
                CoderAgent(**agent_kwargs),
                CodeValidatorAgent(**agent_kwargs),
                TrainerAgent(**agent_kwargs),
                EvaluatorAgent(**agent_kwargs),
                ValidatorAgent(**agent_kwargs),
                CriticAgent(**agent_kwargs),
            ]

            ctx = dict(context)
            for agent in pipeline:
                # Check for cancellation signal via Redis key
                try:
                    import redis as _redis
                    r = _redis.from_url(REDIS_URL, decode_responses=True)
                    cancelled = r.get(f"research:cancel:{research_id}")
                    r.close()
                    if cancelled:
                        logger.info("Research session %s cancelled", research_id)
                        return
                except Exception:
                    pass

                ctx = await agent.run(ctx)

            # Save results
            scored     = ctx.get("scored_candidates", [])
            gen_code   = ctx.get("generated_code", [])
            to_recurse = ctx.get("candidates_to_recurse", [])

            # Build failure patterns: discarded candidates the next generation should avoid
            recurse_names  = {c["architecture_name"] for c in to_recurse}
            code_meta      = {g["architecture_name"]: g for g in gen_code}
            new_failures   = [
                {
                    "architecture_name": c["architecture_name"],
                    "composite_score"  : round(c.get("composite_score", 0), 3),
                    "mutations"        : code_meta.get(c["architecture_name"], {}).get("mutations", []),
                    "failure_reason"   : c.get("next_action", "discard"),
                }
                for c in scored
                if c["architecture_name"] not in recurse_names
            ]
            # Accumulate across generations (cap at 20 to avoid prompt bloat)
            prior_failures = context.get("failed_patterns", [])
            all_failures   = (prior_failures + new_failures)[-20:]

            await _save_candidates(
                research_id, scored, gen_code,
                generation=generation,
                research_papers=ctx.get("research_papers", []),
                mechanisms=ctx.get("candidate_mechanisms", []),
            )
            await _update_mongo_session(research_id, {
                "status"           : "running",
                "generation"       : generation,
                "top_candidates"   : [s["architecture_name"] for s in scored[:3]],
                "updated_at"       : _now_iso(),
            })

            # ── Decide what to do next ──────────────────────────────────────────
            best      = scored[0] if scored else None
            best_code = code_meta.get(best["architecture_name"], {}) if best else {}
            best_score = best["composite_score"] if best else 0.0

            next_gen         = generation + 1
            max_gen0_retries = context.get("max_gen0_retries", 3)
            gen0_attempt     = context.get("gen0_attempt", 0)

            # ── Gen 0 retry logic ────────────────────────────────────────────────
            # If we're still in generation 0, no candidate met the graduation threshold,
            # AND we still have retries left → re-run gen 0 with fresh research + mutations
            # (failure patterns accumulate so we don't repeat the same approaches).
            gen0_graduated = bool(to_recurse) or best_score >= GEN0_GRADUATE_THRESHOLD
            still_in_gen0  = generation == 0
            has_gen0_retry = gen0_attempt + 1 < max_gen0_retries

            if still_in_gen0 and not gen0_graduated and has_gen0_retry:
                new_attempt = gen0_attempt + 1
                logger.info(
                    "Gen 0 attempt %d/%d: best score %.3f < %.2f — retrying with fresh research",
                    new_attempt, max_gen0_retries, best_score, GEN0_GRADUATE_THRESHOLD,
                )
                _publish(research_id, {
                    "event_type"          : "gen0_retry",
                    "research_session_id" : research_id,
                    "generation"          : 0,
                    "gen0_attempt"        : new_attempt,
                    "max_gen0_retries"    : max_gen0_retries,
                    "best_score"          : round(best_score, 3),
                    "message"             : (
                        f"Gen 0 attempt {new_attempt}/{max_gen0_retries}: best score "
                        f"{round(best_score * 100)}% below {round(GEN0_GRADUATE_THRESHOLD * 100)}% threshold "
                        f"— retrying with fresh research"
                    ),
                    "timestamp"           : _now_iso(),
                })
                retry_context = dict(context)
                retry_context["generation"]    = 0          # stay in gen 0
                retry_context["gen0_attempt"]  = new_attempt
                retry_context["failed_patterns"] = all_failures
                # Pass the best code forward so the Coder improves it with the new mechanisms
                # rather than generating from scratch — same "edit not regenerate" principle as gen 1+
                if best:
                    retry_context["previous_winner_code"]  = best_code.get("code", "")
                    retry_context["previous_winner_score"] = round(best_score, 3)
                # Clear pipeline outputs so research/math agents run fresh (new papers, new mechanisms)
                # but keep previous_winner_code so Coder edits rather than rewrites
                for key in ("research_papers", "research_insights",
                            "research_paper_sets", "research_insight_sets",
                            "candidate_mechanisms", "mechanism_sets",
                            "architecture_proposals", "generated_code", "scored_candidates",
                            "candidates_to_recurse"):
                    retry_context.pop(key, None)
                run_research_generation.apply_async(
                    kwargs = dict(research_session_id=research_id, context=retry_context),
                    queue  = "research",
                )
                return

            # ── Move to next generation ──────────────────────────────────────────
            if next_gen < max_gen:
                next_context = dict(context)
                next_context["generation"]      = next_gen
                next_context["depth"]           = context.get("depth", 0) + 1
                next_context["failed_patterns"] = all_failures
                next_context["gen0_attempt"]    = 0          # reset for next gen

                if to_recurse:
                    # Winners found — refine them; reset the failure counter
                    logger.info("Recursing into generation %d with %d candidates", next_gen, len(to_recurse))
                    next_context["previous_winner_arch"]               = ctx.get("previous_winner_arch")
                    next_context["previous_winner_base_arch"]          = ctx.get("previous_winner_base_arch")
                    next_context["candidates_to_recurse"]              = to_recurse
                    next_context["consecutive_improvement_attempts"]   = 0
                    # Pass the best winner's code for the editor to build on
                    if best:
                        next_context["previous_winner_code"]  = best_code.get("code", "")
                        next_context["previous_winner_score"] = round(best_score, 3)
                else:
                    # No winners above recurse threshold — improve the best candidate's
                    # actual code with new mathematical mechanisms rather than regenerating
                    # from a template. This is how real research works: you iterate on
                    # what you have, not throw it away and start over.
                    consecutive = context.get("consecutive_improvement_attempts", 0) + 1
                    logger.info(
                        "Generation %d: no recurse candidates (attempt %d/%d) — improving best: %s (score %.3f)",
                        generation, consecutive, MAX_IMPROVEMENT_ATTEMPTS,
                        best["architecture_name"] if best else "none",
                        best_score,
                    )
                    if best:
                        next_context["previous_winner_arch"]      = best["architecture_name"]
                        next_context["previous_winner_base_arch"] = best_code.get("base_template") or best["architecture_name"]
                        next_context["candidates_to_recurse"]     = scored[:2]
                        # Pass the actual code so the Coder edits it instead of regenerating
                        next_context["previous_winner_code"]      = best_code.get("code", "")
                        next_context["previous_winner_score"]     = round(best_score, 3)
                    else:
                        next_context.pop("previous_winner_arch",       None)
                        next_context.pop("previous_winner_base_arch",  None)
                        next_context.pop("candidates_to_recurse",      None)
                        next_context.pop("previous_winner_code",       None)
                        next_context.pop("previous_winner_score",      None)
                    next_context["consecutive_improvement_attempts"] = consecutive

                    # After MAX attempts without a winner, pause and ask the user
                    if consecutive >= MAX_IMPROVEMENT_ATTEMPTS:
                        logger.info(
                            "Session %s: %d consecutive failed generations — awaiting user decision",
                            research_id, consecutive,
                        )
                        await _update_mongo_session(research_id, {
                            "status"          : "awaiting_decision",
                            "pending_context" : next_context,
                            "updated_at"      : _now_iso(),
                        })
                        _publish(research_id, {
                            "event_type"              : "awaiting_user_decision",
                            "research_session_id"     : research_id,
                            "generation"              : generation,
                            "consecutive_failures"    : consecutive,
                            "best_score"              : round(best_score, 3),
                            "message"                 : (
                                f"Tried {consecutive} times to improve without finding strong candidates "
                                f"(best score: {round(best_score * 100)}%). "
                                f"Continue exploring or stop?"
                            ),
                            "timestamp"               : _now_iso(),
                        })
                        return   # pause — wait for user to call /continue

                _publish(research_id, {
                    "event_type"          : "generation_complete",
                    "research_session_id" : research_id,
                    "generation"          : generation,
                    "recurse"             : True,
                    "mode"                : "recurse" if to_recurse else "improve_best",
                    "timestamp"           : _now_iso(),
                })
                run_research_generation.apply_async(
                    kwargs = dict(research_session_id=research_id, context=next_context),
                    queue  = "research",
                )
            else:
                # Session complete — max generations reached
                top = scored[0] if scored else {}
                await _update_mongo_session(research_id, {
                    "status"         : "completed",
                    "completed_at"   : _now_iso(),
                    "best_candidate" : top.get("architecture_name"),
                    "best_score"     : top.get("composite_score", 0),
                })
                _publish(research_id, {
                    "event_type"          : "session_complete",
                    "research_session_id" : research_id,
                    "generation"          : generation,
                    "best_candidate"      : top.get("architecture_name"),
                    "best_score"          : top.get("composite_score", 0),
                    "timestamp"           : _now_iso(),
                })

        except Exception as e:
            logger.exception("Research session %s failed: %s", research_id, e)
            await _update_mongo_session(research_id, {
                "status"    : "error",
                "error"     : str(e),
                "failed_at" : _now_iso(),
            })
            _publish(research_id, {
                "event_type"          : "session_error",
                "research_session_id" : research_id,
                "error"               : str(e),
                "timestamp"           : _now_iso(),
            })

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_run())
    finally:
        loop.close()


# ── Periodic idle-check ───────────────────────────────────────────────────────

@research_celery_app.task(name="tasks_research.check_idle_and_spawn")
def check_idle_and_spawn():
    """
    Periodic beat task: find research sessions stuck in 'queued' or 'running'
    state for more than 10 minutes and attempt to requeue them.
    """
    import asyncio

    async def _run():
        try:
            from database_mongo import get_database
            db = get_database()

            cutoff = _now_iso()   # use ISO; MongoDB stores as string here
            stale  = await db["research_sessions"].find({
                "status": {"$in": ["queued", "running"]},
            }).to_list(length=20)

            for doc in stale:
                research_id = doc["_id"]
                updated_at  = doc.get("updated_at", doc.get("created_at", ""))
                # Simple staleness: if we can find no recent Redis activity, requeue
                try:
                    import redis as _redis
                    r = _redis.from_url(REDIS_URL, decode_responses=True)
                    heartbeat = r.get(f"research:heartbeat:{research_id}")
                    r.close()
                    if heartbeat:
                        continue   # still alive
                except Exception:
                    pass

                # Requeue if still queued
                if doc.get("status") == "queued":
                    request = doc.get("request", {})
                    if request:
                        logger.info("Re-queuing stale research session %s", research_id)
                        run_research_generation.apply_async(
                            kwargs = dict(research_session_id=research_id, context={
                                "session_id"                : doc.get("session_id", ""),
                                "research_session_id"       : research_id,
                                "domain"                    : request.get("domain", "vision"),
                                "category"                  : request.get("category", "vision"),
                                "task_description"          : request.get("task_description", ""),
                                "population_size"           : request.get("population_size", 3),
                                "max_generations"           : request.get("max_generations", 3),
                                "enable_real_data_validation": request.get("enable_real_data_validation", False),
                                "generation"                : 0,
                                "depth"                     : 0,
                            }),
                            queue = "research",
                        )

        except Exception as e:
            logger.warning("check_idle_and_spawn failed: %s", e)

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_run())
    finally:
        loop.close()
