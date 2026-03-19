"""
Local Docker GPU provider — runs training jobs in an isolated container on the host machine.
Requires: docker, nvidia-docker2 (or --gpus all support), and a pre-built training image.

Image env var: TRAINING_DOCKER_IMAGE (default: obsidian-trainer:latest)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

DOCKER_IMAGE    = os.environ.get("TRAINING_DOCKER_IMAGE", "obsidian-trainer:latest")
ARTIFACTS_DIR   = Path(os.environ.get("RESEARCH_ARTIFACTS_DIR", "/research_artifacts"))
CONTAINER_PREFIX = "obs_trainer_"


class LocalDockerProvider:
    name             = "local_docker"
    pricing_per_hour = 0.0    # free — uses host GPU
    min_latency_s    = 5

    def __init__(self):
        self._docker = None

    def _get_docker(self):
        if self._docker is None:
            import docker
            self._docker = docker.from_env()
        return self._docker

    async def submit_training_job(
        self,
        candidate_id: str,
        code: str,
        data_path: str,
        framework: str = "tensorflow",
        timeout_mins: int = 5,
    ) -> dict:
        """Write code to a temp file and run it in a Docker container."""
        job_id     = f"{CONTAINER_PREFIX}{candidate_id}_{uuid.uuid4().hex[:8]}"
        output_dir = ARTIFACTS_DIR / "docker_jobs" / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write training script
        script_path = output_dir / "train.py"
        script_path.write_text(code, encoding="utf-8")

        # Write job metadata
        meta = {"candidate_id": candidate_id, "framework": framework, "timeout_mins": timeout_mins}
        (output_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

        try:
            client  = self._get_docker()
            volumes = {
                str(output_dir): {"bind": "/job", "mode": "rw"},
            }
            if data_path and Path(data_path).exists():
                volumes[data_path] = {"bind": "/data", "mode": "ro"}

            # Launch detached container
            container = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.containers.run(
                    DOCKER_IMAGE,
                    command=f"python /job/train.py",
                    volumes=volumes,
                    detach=True,
                    name=job_id,
                    device_requests=[
                        {"Driver": "nvidia", "Count": -1, "Capabilities": [["gpu"]]}
                    ] if self._has_gpu(client) else [],
                    mem_limit="8g",
                    cpu_period=100000,
                    cpu_quota=400000,   # 4 CPUs
                    environment={"TF_CPP_MIN_LOG_LEVEL": "3"},
                ),
            )

            return {
                "job_id"      : job_id,
                "container_id": container.id,
                "output_dir"  : str(output_dir),
                "status"      : "running",
                "provider"    : self.name,
            }

        except Exception as e:
            logger.error("Docker job submission failed: %s", e)
            return {
                "job_id"  : job_id,
                "status"  : "failed",
                "error"   : str(e),
                "provider": self.name,
            }

    async def check_job_status(self, job_id: str) -> dict:
        try:
            client    = self._get_docker()
            container = await asyncio.get_event_loop().run_in_executor(
                None, lambda: client.containers.get(job_id)
            )
            container.reload()
            state = container.status   # running | exited | dead

            if state == "exited":
                exit_code = container.attrs["State"]["ExitCode"]
                return {
                    "job_id"   : job_id,
                    "status"   : "completed" if exit_code == 0 else "failed",
                    "exit_code": exit_code,
                }
            return {"job_id": job_id, "status": "running"}

        except Exception as e:
            return {"job_id": job_id, "status": "unknown", "error": str(e)}

    async def get_job_result(self, job_id: str) -> dict:
        """Read training metrics from the job's output directory."""
        output_dir  = ARTIFACTS_DIR / "docker_jobs" / job_id
        result_file = output_dir / "result.json"

        if result_file.exists():
            try:
                data = json.loads(result_file.read_text())
                return {"job_id": job_id, "status": "completed", "metrics": data}
            except Exception:
                pass

        # Fallback: parse stdout logs
        logs_file = output_dir / "stdout.log"
        if logs_file.exists():
            return {
                "job_id" : job_id,
                "status" : "completed",
                "metrics": {"loss": self._parse_loss(logs_file.read_text())},
            }

        return {"job_id": job_id, "status": "no_result"}

    async def cancel_job(self, job_id: str) -> bool:
        try:
            client    = self._get_docker()
            container = await asyncio.get_event_loop().run_in_executor(
                None, lambda: client.containers.get(job_id)
            )
            await asyncio.get_event_loop().run_in_executor(None, container.stop)
            return True
        except Exception:
            return False

    def _has_gpu(self, client) -> bool:
        try:
            info = client.info()
            runtimes = info.get("Runtimes", {})
            return "nvidia" in runtimes
        except Exception:
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
