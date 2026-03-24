"""
EvaluatorAgent — evaluates trained checkpoints on synthetic test data.

Computes domain-specific metrics: accuracy, loss, memory_mb, inference_time_ms, param_count.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path

from .core import BaseAgent

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(os.environ.get("RESEARCH_ARTIFACTS_DIR", "/research_artifacts"))


class EvaluatorAgent(BaseAgent):

    async def run(self, context: dict) -> dict:
        generation       = context.get("generation", 0)
        depth            = context.get("depth", 0)
        domain           = context.get("domain", "vision")
        training_results = context.get("training_results", [])

        await self.emit_progress("agent_start", f"Evaluator assessing {len(training_results)} checkpoints...", generation, depth)
        self.log_step("Starting evaluation", {"domain": domain})

        # Load metrics skill — provides per-domain random-chance thresholds
        # used to flag models that failed to learn as "training_failed".
        self._metrics_skill = self.load_skill(filename="metrics.md")
        self._domain = domain

        from agents.domains import get_domain
        domain_handler = get_domain(domain)

        # Generate test data once for all candidates
        test_data = await asyncio.get_event_loop().run_in_executor(
            None, domain_handler.generate_synthetic_data, 200
        )

        tasks = [
            self._evaluate_single(domain_handler, r, test_data)
            for r in training_results
            if r.get("status") == "completed" and r.get("checkpoint_path")
        ]
        eval_results = await asyncio.gather(*tasks, return_exceptions=True)

        evaluation_results = []
        for r in eval_results:
            if isinstance(r, Exception):
                self.log_step("Evaluation failed", {"error": str(r)})
            elif r is not None:
                evaluation_results.append(r)

        # Include failed training runs with zero metrics
        evaluated_names = {e["architecture_name"] for e in evaluation_results}
        for tr in training_results:
            if tr["architecture_name"] not in evaluated_names:
                evaluation_results.append({
                    "architecture_name": tr["architecture_name"],
                    "synthetic_metrics": {"loss": 999.0},
                    "memory_mb"        : 0.0,
                    "inference_time_ms": 9999.0,
                    "param_count"      : 0,
                    "status"           : tr.get("status", "failed"),
                })

        await self.emit_progress(
            "agent_done",
            f"Evaluator completed {len(evaluation_results)} evaluations",
            generation, depth,
        )

        context["evaluation_results"] = evaluation_results
        return context

    async def _evaluate_single(self, domain_handler, training_result: dict, test_data) -> dict | None:
        arch_name       = training_result["architecture_name"]
        checkpoint_path = training_result.get("checkpoint_path", "")

        if not checkpoint_path or not Path(checkpoint_path).exists():
            return {
                "architecture_name": arch_name,
                "synthetic_metrics": {"loss": 999.0},
                "memory_mb"        : 0.0,
                "inference_time_ms": 9999.0,
                "param_count"      : training_result.get("param_count", 0),
                "training_time_s"  : training_result.get("training_time_s", 9999.0),
                "status"           : "no_checkpoint",
            }

        try:
            # Domain-specific evaluation
            metrics = await domain_handler.evaluate(checkpoint_path, test_data)

            # Memory + inference time measurement
            memory_mb, inference_ms = await asyncio.get_event_loop().run_in_executor(
                None, self._measure_resources, checkpoint_path, test_data
            )

            status = "training_failed" if self._is_random_chance(metrics) else "evaluated"

            return {
                "architecture_name": arch_name,
                "synthetic_metrics": metrics,
                "memory_mb"        : memory_mb,
                "inference_time_ms": inference_ms,
                "param_count"      : training_result.get("param_count", 0),
                "training_time_s"  : training_result.get("training_time_s", 0),
                "status"           : status,
            }
        except Exception as e:
            self.log_step(f"Evaluation error for {arch_name}", {"error": str(e)})
            return {
                "architecture_name": arch_name,
                "synthetic_metrics": {"loss": 999.0, "error": str(e)},
                "memory_mb"        : 0.0,
                "inference_time_ms": 9999.0,
                "param_count"      : 0,
                "status"           : "eval_error",
            }

    def _is_random_chance(self, metrics: dict) -> bool:
        """Return True if metrics indicate the model did not learn (random-chance performance).

        Thresholds are informed by the evaluator/metrics.md skill; see that file for
        the per-domain interpretation table.
        """
        domain = getattr(self, "_domain", "vision")
        loss   = metrics.get("loss", 0.0)
        acc    = metrics.get("accuracy")
        mse    = metrics.get("mse", metrics.get("loss", 0.0))

        if loss >= 999.0:
            return True

        if domain == "vision":
            return acc is not None and acc < 0.12
        if domain == "language":
            return acc is not None and acc < 0.20
        if domain == "timeseries":
            return mse > 1.0
        if domain == "graph":
            return acc is not None and acc < 0.20
        if domain == "audio":
            return acc is not None and acc < 0.20
        if domain == "tabular":
            return acc is not None and acc < 0.33
        if domain == "generative":
            return loss > 10.0
        if domain == "recommendation":
            return loss > 0.6
        if domain == "multimodal":
            return acc is not None and acc < 0.12
        return False

    def _measure_resources(self, checkpoint_path: str, test_data) -> tuple[float, float]:
        """Measure memory usage and inference time for a checkpoint."""
        try:
            import os, time
            import tensorflow as tf
            import numpy as np
            import tracemalloc

            tracemalloc.start()
            model = tf.keras.models.load_model(checkpoint_path)

            # Build a small input batch
            if isinstance(test_data, tuple):
                sample = test_data[0][:10]   # X_test slice
            elif isinstance(test_data, dict):
                sample = next(iter(test_data.get("test", {}).values()))[:10]
            else:
                return 0.0, 0.0

            if hasattr(sample, "astype"):
                sample = sample.astype(np.float32)

            # Warm up
            model.predict(sample, verbose=0)

            # Time inference
            start = time.perf_counter()
            for _ in range(5):
                model.predict(sample, verbose=0)
            elapsed_ms = (time.perf_counter() - start) / 5 * 1000

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            return round(peak / 1024 / 1024, 2), round(elapsed_ms, 2)
        except Exception:
            return 0.0, 0.0
