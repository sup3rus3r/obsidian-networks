"""
RunPod serverless GPU provider.
Docs: https://docs.runpod.io/reference/run-sync
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

RUNPOD_API_BASE = "https://api.runpod.io/graphql"


class RunPodProvider:
    name              = "runpod"
    pricing_per_hour  = 0.20   # RTX 4090 community
    min_latency_s     = 10

    def __init__(self, api_key: str, gpu_type: str = "RTX4090"):
        self.api_key  = api_key
        self.gpu_type = gpu_type

    async def submit_training_job(
        self,
        candidate_id: str,
        code: str,
        data_path: str,
        framework: str = "tensorflow",
        timeout_mins: int = 5,
    ) -> dict:
        if not self.api_key:
            raise ValueError("RUNPOD_API_KEY not set")

        # For RunPod serverless: we'd submit to a pre-built worker endpoint
        # This is the stub — real implementation requires a RunPod worker image
        # that accepts {"candidate_id", "code", "framework"} and returns metrics
        raise NotImplementedError(
            "RunPod provider requires a deployed RunPod worker endpoint. "
            "Set RUNPOD_ENDPOINT_ID env var and deploy the worker image."
        )

    async def check_job_status(self, job_id: str) -> dict:
        raise NotImplementedError("RunPod status check not implemented")

    async def get_job_result(self, job_id: str) -> dict:
        raise NotImplementedError("RunPod result retrieval not implemented")

    async def cancel_job(self, job_id: str) -> bool:
        return False
