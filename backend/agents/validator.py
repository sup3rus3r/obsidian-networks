"""
ValidatorAgent — optional agent that evaluates architectures on real data.

Only runs if context["enable_real_data_validation"] is True and real_data_path exists.
Computes generalization_score = max(0, 1 - (real_loss/synthetic_loss - 1))
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from .core import BaseAgent

logger = logging.getLogger(__name__)


class ValidatorAgent(BaseAgent):

    async def run(self, context: dict) -> dict:
        generation = context.get("generation", 0)
        depth      = context.get("depth", 0)
        enabled    = context.get("enable_real_data_validation", False)
        real_path  = context.get("real_data_path")

        if not enabled or not real_path:
            # Skip — set default generalization scores
            context["validation_results"] = []
            return context

        await self.emit_progress("agent_start", "Validator testing on real data...", generation, depth)

        # Load soundness skill — provides domain-specific loss_ratio thresholds
        # used to calibrate overfitting detection per domain.
        self._soundness_skill = self.load_skill(filename="soundness.md")

        domain           = context.get("domain", "vision")
        eval_results     = context.get("evaluation_results", [])
        real_data_source = context.get("real_data_source", {})

        from agents.domains import get_domain
        domain_handler = get_domain(domain)

        # Load real data
        try:
            real_data = await asyncio.get_event_loop().run_in_executor(
                None, self._load_real_data, real_path
            )
        except Exception as e:
            self.log_step("Failed to load real data", {"error": str(e)})
            context["validation_results"] = []
            return context

        tasks = [
            self._validate_single(domain_handler, er, real_data)
            for er in eval_results
            if er.get("status") == "evaluated"
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        validation_results = []
        for r in results:
            if isinstance(r, Exception):
                self.log_step("Validation error", {"error": str(r)})
            elif r is not None:
                validation_results.append(r)

        await self.emit_progress(
            "agent_done",
            f"Validator completed {len(validation_results)} real-data evaluations",
            generation, depth,
        )

        context["validation_results"] = validation_results
        return context

    async def _validate_single(self, domain_handler, eval_result: dict, real_data) -> dict | None:
        arch_name        = eval_result["architecture_name"]
        synthetic_metrics = eval_result.get("synthetic_metrics", {})
        synthetic_loss   = synthetic_metrics.get("loss", 1.0)

        # Find checkpoint
        checkpoint_path = self._find_checkpoint(arch_name)
        if not checkpoint_path:
            return None

        try:
            real_metrics = await domain_handler.evaluate(checkpoint_path, real_data)
            real_loss    = real_metrics.get("loss", synthetic_loss)

            loss_ratio          = real_loss / max(synthetic_loss, 1e-6)
            generalization_score = max(0.0, 1.0 - (loss_ratio - 1.0) / 1.0)
            generalization_score = min(1.0, generalization_score)
            overfitting          = loss_ratio > self._overfitting_threshold()

            return {
                "architecture_name"   : arch_name,
                "real_metrics"        : real_metrics,
                "synthetic_metrics"   : synthetic_metrics,
                "loss_ratio"          : round(loss_ratio, 4),
                "generalization_score": round(generalization_score, 4),
                "overfitting_detected": overfitting,
                "status"              : "completed",
            }
        except Exception as e:
            return {
                "architecture_name"   : arch_name,
                "real_metrics"        : {},
                "synthetic_metrics"   : synthetic_metrics,
                "loss_ratio"          : 1.0,
                "generalization_score": 0.5,
                "overfitting_detected": False,
                "status"              : "failed",
                "error"               : str(e),
            }

    def _find_checkpoint(self, arch_name: str) -> str | None:
        """Look for the trained checkpoint in the artifacts directory."""
        import os
        base = Path(os.environ.get("RESEARCH_ARTIFACTS_DIR", "/research_artifacts"))
        checkpoint = base / self.research_session_id / "checkpoints" / arch_name / "model.keras"
        return str(checkpoint) if checkpoint.exists() else None

    def _overfitting_threshold(self) -> float:
        """Return the domain-specific loss_ratio threshold above which overfitting is flagged.
        Thresholds are informed by the soundness skill; see validator/soundness.md.
        """
        thresholds = {
            "vision"        : 1.5,
            "tabular"       : 1.5,
            "timeseries"    : 2.0,
            "language"      : 2.0,
            "graph"         : 2.0,
            "recommendation": 2.0,
            "generative"    : 2.5,
            "audio"         : 2.0,
            "multimodal"    : 2.0,
        }
        return thresholds.get(self.domain, 1.5)

    def _load_real_data(self, real_data_path: str):
        """Load real data from path — domain-agnostic best-effort loader."""
        import numpy as np
        path = Path(real_data_path)
        if path.suffix == ".npz":
            data = np.load(path)
            return (data.get("X_train"), data.get("X_test"), data.get("y_train"), data.get("y_test"))
        raise ValueError(f"Unsupported real data format: {path.suffix}")
