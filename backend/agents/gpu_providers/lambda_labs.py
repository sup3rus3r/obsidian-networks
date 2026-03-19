"""
Lambda Labs GPU provider.
Docs: https://cloud.lambdalabs.com/api/v1
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

LAMBDA_API_BASE = "https://cloud.lambdalabs.com/api/v1"


class LambdaLabsProvider:
    name             = "lambda_labs"
    pricing_per_hour = 0.60   # A100 on-demand
    min_latency_s    = 30

    def __init__(self, api_key: str, instance_type: str = "gpu_1x_a100"):
        self.api_key       = api_key
        self.instance_type = instance_type
        self._headers      = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type" : "application/json",
        }

    async def submit_training_job(
        self,
        candidate_id: str,
        code: str,
        data_path: str,
        framework: str = "tensorflow",
        timeout_mins: int = 5,
    ) -> dict:
        if not self.api_key:
            raise ValueError("LAMBDA_API_KEY not set")

        # Lambda Labs does not have a native serverless job API.
        # Real implementation: launch an instance via /instances, SSH the code, run it.
        # This stub documents the intended flow.
        raise NotImplementedError(
            "Lambda Labs provider requires instance launch + SSH execution. "
            "Set LAMBDA_API_KEY env var and deploy a pre-configured instance template."
        )

    async def list_instance_types(self) -> list[dict]:
        """Return available GPU instance types."""
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{LAMBDA_API_BASE}/instance-types",
                headers=self._headers,
            )
            resp.raise_for_status()
            data = resp.json()
            return list(data.get("data", {}).values())

    async def check_job_status(self, job_id: str) -> dict:
        raise NotImplementedError("Lambda Labs status check not implemented")

    async def get_job_result(self, job_id: str) -> dict:
        raise NotImplementedError("Lambda Labs result retrieval not implemented")

    async def cancel_job(self, job_id: str) -> bool:
        return False
