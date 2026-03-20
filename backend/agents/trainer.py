"""
TrainerAgent — trains generated architectures on synthetic data.

Supports:
- Local CPU/GPU training (default)
- Serverless GPU via GPUManager (if USE_SERVERLESS_GPU=true)
"""
from __future__ import annotations

import asyncio
import importlib
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

from .core import BaseAgent

logger = logging.getLogger(__name__)

ARTIFACTS_DIR    = Path(os.environ.get("RESEARCH_ARTIFACTS_DIR", "/research_artifacts"))
USE_SERVERLESS   = os.environ.get("USE_SERVERLESS_GPU", "false").lower() == "true"
TOY_EPOCHS       = int(os.environ.get("RESEARCH_TOY_EPOCHS", "5"))
TOY_BATCH_SIZE   = int(os.environ.get("RESEARCH_TOY_BATCH_SIZE", "32"))


class TrainerAgent(BaseAgent):

    async def run(self, context: dict) -> dict:
        generation     = context.get("generation", 0)
        depth          = context.get("depth", 0)
        domain         = context.get("domain", "vision")
        generated_code = context.get("generated_code", [])
        dataset_path   = context.get("dataset_path")

        await self.emit_progress("agent_start", f"Trainer running {len(generated_code)} training jobs...", generation, depth)
        self.log_step("Starting training", {"n_candidates": len(generated_code), "gpu": USE_SERVERLESS})

        if USE_SERVERLESS:
            training_results = await self._train_on_gpu(generated_code, dataset_path)
        else:
            training_results = await self._train_local(generated_code, domain, generation, depth)

        await self.emit_progress(
            "agent_done",
            f"Trainer completed {len(training_results)} training runs",
            generation, depth,
            {"results": [{
                "name"  : r["architecture_name"],
                "loss"  : round(r.get("final_loss", 999), 4),
                "status": r.get("status"),
            } for r in training_results]},
        )

        context["training_results"] = training_results
        return context

    async def _train_local(self, candidates: list[dict], domain: str, generation: int, depth: int) -> list[dict]:
        """Train each candidate sequentially on local CPU/GPU."""
        results = []
        session_dir = ARTIFACTS_DIR / self.research_session_id
        checkpoints_dir = session_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        for candidate in candidates:
            arch_name    = candidate["architecture_name"]
            code         = candidate["code"]
            checkpoint   = checkpoints_dir / arch_name / "model.keras"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)

            await self.emit_progress(
                "agent_start", f"Training {arch_name}...",
                generation, depth, {"architecture": arch_name}
            )

            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._run_training_script,
                code, str(checkpoint), arch_name, param_count,
            )
            results.append(result)

        return results

    def _run_training_script(self, code: str, checkpoint_path: str, arch_name: str, param_count: int = 0) -> dict:
        """Execute training code in a subprocess-like isolated exec with timeout."""
        import subprocess, sys, tempfile, json as _json

        # Inject checkpoint path and epoch override into the code
        patched = self._patch_code(code, checkpoint_path)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(patched)
            script_path = f.name

        start = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=300,  # 5-minute hard limit
            )
            elapsed = time.time() - start

            if proc.returncode == 0:
                # Try to parse final loss from stdout
                loss = self._parse_loss(proc.stdout)
                return {
                    "architecture_name" : arch_name,
                    "final_loss"        : loss,
                    "accuracy"          : None,
                    "checkpoint_path"   : checkpoint_path,
                    "training_time_s"   : elapsed,
                    "param_count"       : param_count,
                    "training_location" : "local",
                    "status"            : "completed",
                    "stdout"            : proc.stdout[-500:],
                }
            else:
                return {
                    "architecture_name" : arch_name,
                    "final_loss"        : 999.0,
                    "checkpoint_path"   : checkpoint_path,
                    "training_time_s"   : elapsed,
                    "param_count"       : param_count,
                    "training_location" : "local",
                    "status"            : "failed",
                    "error"             : proc.stderr[-500:],
                }
        except subprocess.TimeoutExpired:
            return {
                "architecture_name": arch_name,
                "final_loss"       : 999.0,
                "training_time_s"  : 300.0,
                "param_count"      : param_count,
                "training_location": "local",
                "status"           : "timeout",
                "error"            : "Training exceeded 5-minute limit",
            }
        except Exception as e:
            return {
                "architecture_name": arch_name,
                "final_loss"       : 999.0,
                "param_count"      : param_count,
                "training_location": "local",
                "status"           : "error",
                "error"            : str(e),
            }
        finally:
            try:
                Path(script_path).unlink()
            except Exception:
                pass

    def _patch_code(self, code: str, checkpoint_path: str) -> str:
        """Inject checkpoint path and cap epochs for toy training."""
        import re
        # Override output path
        output_dir = str(Path(checkpoint_path).parent)
        code = f"import os; os.makedirs('{output_dir}', exist_ok=True)\n" + code

        # Replace ALL output/*.keras and output/*.h5 save paths — not just the
        # literal 'output/model.keras' — so any model name the LLM chose gets
        # redirected to the canonical checkpoint path the evaluator expects.
        code = re.sub(r"'output/[^']*\.keras'", f"'{checkpoint_path}'", code)
        code = re.sub(r'"output/[^"]*\.keras"', f'"{checkpoint_path}"', code)
        code = re.sub(r"'output/[^']*\.h5'",    f"'{checkpoint_path}'", code)
        code = re.sub(r'"output/[^"]*\.h5"',    f'"{checkpoint_path}"', code)

        # Cap epochs
        code = re.sub(r'epochs\s*=\s*\d+', f'epochs={TOY_EPOCHS}', code)

        # Silence TF logs
        prefix = "import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'\n"
        return prefix + code

    def _parse_loss(self, stdout: str) -> float:
        """Extract final loss value from training output."""
        import re
        # Look for patterns like "loss: 0.4523" or "val_loss: 0.3"
        matches = re.findall(r'(?:val_)?loss[:\s]+([0-9]+\.[0-9]+)', stdout)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                pass
        return 0.5  # default if not parseable

    async def _train_on_gpu(self, candidates: list[dict], dataset_path: str | None) -> list[dict]:
        """Offload training to serverless GPU providers."""
        from agents.gpu_manager import GPUManager
        manager  = GPUManager()
        job_list = await manager.submit_training_batch(candidates, dataset_path or "")
        results  = await manager.wait_for_jobs(job_list, timeout_mins=10)

        training_results = []
        for candidate, job in zip(candidates, job_list):
            job_id = job.get("job_id")
            r      = results.get(job_id, {"status": "failed"})
            training_results.append({
                "architecture_name" : candidate["architecture_name"],
                "final_loss"        : r.get("metrics", {}).get("loss", 999.0),
                "accuracy"          : r.get("metrics", {}).get("accuracy"),
                "checkpoint_path"   : r.get("checkpoint_url", ""),
                "training_time_s"   : r.get("metrics", {}).get("training_time_s", 0),
                "training_location" : job.get("provider", "unknown"),
                "status"            : r.get("status", "failed"),
            })
        return training_results
