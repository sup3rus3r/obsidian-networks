"""
GPUManager — orchestrates training across GPU providers.

Priority order:
  1. local_docker  (free, if Docker + GPU available)
  2. runpod        (if RUNPOD_API_KEY set)
  3. lambda_labs   (if LAMBDA_API_KEY set)
  4. cpu_fallback  (always available — runs training locally without container)

Each provider must implement:
  submit_training_job(candidate_id, code, data_path, framework, timeout_mins) → {job_id, ...}
  check_job_status(job_id) → {status: running|completed|failed, ...}
  get_job_result(job_id) → {metrics: {loss, accuracy, ...}, ...}
  cancel_job(job_id) → bool
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

RUNPOD_API_KEY  = os.environ.get("RUNPOD_API_KEY", "")
LAMBDA_API_KEY  = os.environ.get("LAMBDA_API_KEY", "")
POLL_INTERVAL_S = int(os.environ.get("GPU_POLL_INTERVAL_S", "15"))


class GPUManager:
    """Selects the best available provider and manages job lifecycle."""

    def __init__(self):
        self._provider = self._select_provider()
        logger.info("GPUManager: selected provider=%s", self._provider.name)

    def _select_provider(self):
        # 1. Local Docker
        try:
            from agents.gpu_providers.local_docker import LocalDockerProvider
            p = LocalDockerProvider()
            p._get_docker()   # will raise if docker not available
            return p
        except Exception:
            pass

        # 2. RunPod
        if RUNPOD_API_KEY:
            from agents.gpu_providers.runpod import RunPodProvider
            return RunPodProvider(RUNPOD_API_KEY)

        # 3. Lambda Labs
        if LAMBDA_API_KEY:
            from agents.gpu_providers.lambda_labs import LambdaLabsProvider
            return LambdaLabsProvider(LAMBDA_API_KEY)

        # 4. CPU fallback
        return _CPUFallbackProvider()

    @property
    def provider_name(self) -> str:
        return self._provider.name

    async def submit_training_batch(
        self,
        candidates: list[dict],
        dataset_path: str,
        framework: str = "tensorflow",
        timeout_mins: int = 10,
    ) -> list[dict]:
        """Submit all candidates and return list of job descriptors."""
        jobs = []
        for candidate in candidates:
            arch_name = candidate["architecture_name"]
            code      = candidate.get("code", "")
            try:
                job = await self._provider.submit_training_job(
                    candidate_id  = arch_name,
                    code          = code,
                    data_path     = dataset_path,
                    framework     = framework,
                    timeout_mins  = timeout_mins,
                )
                job["architecture_name"] = arch_name
                jobs.append(job)
            except NotImplementedError:
                # Provider not fully deployed — fallback to CPU
                logger.warning("Provider %s raised NotImplementedError, falling back to CPU", self._provider.name)
                fallback = _CPUFallbackProvider()
                job = await fallback.submit_training_job(
                    candidate_id = arch_name,
                    code         = code,
                    data_path    = dataset_path,
                    framework    = framework,
                    timeout_mins = timeout_mins,
                )
                job["architecture_name"] = arch_name
                jobs.append(job)
            except Exception as e:
                logger.error("Failed to submit job for %s: %s", arch_name, e)
                jobs.append({
                    "architecture_name": arch_name,
                    "job_id"           : f"failed_{arch_name}",
                    "status"           : "failed",
                    "error"            : str(e),
                    "provider"         : self._provider.name,
                })
        return jobs

    async def wait_for_jobs(
        self,
        job_list: list[dict],
        timeout_mins: int = 15,
    ) -> dict[str, dict]:
        """Poll all jobs until completion or timeout. Returns {job_id: result}."""
        timeout_s  = timeout_mins * 60
        elapsed    = 0
        pending    = {j["job_id"]: j for j in job_list if j.get("status") not in ("failed", "no_submit")}
        results    = {}

        # Pre-fill failed
        for j in job_list:
            if j.get("status") in ("failed", "no_submit"):
                results[j["job_id"]] = {"status": "failed", "metrics": {}}

        while pending and elapsed < timeout_s:
            await asyncio.sleep(POLL_INTERVAL_S)
            elapsed += POLL_INTERVAL_S

            done_ids = []
            for job_id, job_info in pending.items():
                try:
                    status = await self._provider.check_job_status(job_id)
                    if status.get("status") in ("completed", "failed"):
                        done_ids.append(job_id)
                        if status.get("status") == "completed":
                            result = await self._provider.get_job_result(job_id)
                            results[job_id] = result
                        else:
                            results[job_id] = {"status": "failed", "metrics": {}}
                except Exception as e:
                    logger.error("Error polling job %s: %s", job_id, e)

            for job_id in done_ids:
                del pending[job_id]

        # Timeout any remaining
        for job_id in pending:
            results[job_id] = {"status": "timeout", "metrics": {}}
            try:
                await self._provider.cancel_job(job_id)
            except Exception:
                pass

        return results


class _CPUFallbackProvider:
    """
    Runs training inline using subprocess (same as TrainerAgent._run_training_script).
    Used when no GPU provider is available or a GPU provider stub raises NotImplementedError.
    """
    name             = "cpu_fallback"
    pricing_per_hour = 0.0
    min_latency_s    = 0

    async def submit_training_job(
        self,
        candidate_id: str,
        code: str,
        data_path: str,
        framework: str = "tensorflow",
        timeout_mins: int = 5,
    ) -> dict:
        # Run in executor so it doesn't block the event loop
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            self._run_sync,
            candidate_id, code, timeout_mins,
        )
        return result

    def _run_sync(self, candidate_id: str, code: str, timeout_mins: int) -> dict:
        import subprocess, sys, tempfile, time
        from pathlib import Path

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            prefix = "import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'\n"
            f.write(prefix + code)
            script = f.name

        start = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, script],
                capture_output=True,
                text=True,
                timeout=timeout_mins * 60,
            )
            elapsed = time.time() - start
            if proc.returncode == 0:
                return {
                    "job_id"  : f"cpu_{candidate_id}",
                    "status"  : "completed",
                    "metrics" : {"loss": self._parse_loss(proc.stdout)},
                    "provider": self.name,
                    "elapsed" : elapsed,
                }
            return {
                "job_id"  : f"cpu_{candidate_id}",
                "status"  : "failed",
                "error"   : proc.stderr[-300:],
                "provider": self.name,
            }
        except subprocess.TimeoutExpired:
            return {
                "job_id"  : f"cpu_{candidate_id}",
                "status"  : "timeout",
                "provider": self.name,
            }
        finally:
            try:
                Path(script).unlink()
            except Exception:
                pass

    async def check_job_status(self, job_id: str) -> dict:
        # CPU fallback is synchronous — always returns completed by the time we poll
        return {"job_id": job_id, "status": "completed"}

    async def get_job_result(self, job_id: str) -> dict:
        return {"job_id": job_id, "status": "completed", "metrics": {}}

    async def cancel_job(self, job_id: str) -> bool:
        return False

    def _parse_loss(self, stdout: str) -> float:
        import re
        matches = re.findall(r'(?:val_)?loss[:\s]+([0-9]+\.[0-9]+)', stdout)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                pass
        return 0.5
