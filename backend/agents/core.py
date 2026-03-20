"""
BaseAgent — shared infrastructure for all autonomous research agents.

Provides:
- LLM routing (Ollama for simple tasks, Claude API for complex reasoning)
- Prompt caching
- SSE progress emission via Redis pub/sub
- Structured logging
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone

import httpx
from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)

REDIS_URL     = os.environ.get("REDIS_URL", "redis://redis:6379/0")
OLLAMA_URL    = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
LOCAL_MODEL   = os.environ.get("LOCAL_LLM_MODEL", "mistral")
USE_LOCAL_LLM = os.environ.get("USE_LOCAL_LLM", "false").lower() == "true"

# Tasks that can safely run on a smaller/local model
_LOCAL_TASK_KEYWORDS = {"extract", "classify", "summarize", "parse", "rate", "score", "compare", "list"}


class BaseAgent(ABC):
    """Abstract base class for all research mode agents."""

    def __init__(self, research_session_id: str, base_model: str, domain: str):
        self.research_session_id = research_session_id
        self.base_model          = base_model
        self.domain              = domain
        self.logger              = logging.getLogger(f"agents.{self.__class__.__name__}")
        self._llm_cache: dict[str, str] = {}
        self._anthropic: AsyncAnthropic | None = None

    # ── Abstract interface ─────────────────────────────────────────────────────

    @abstractmethod
    async def run(self, context: dict) -> dict:
        """Execute agent logic. Receives full context dict, returns updated context."""
        pass

    # ── LLM routing ───────────────────────────────────────────────────────────

    async def call_llm(
        self,
        prompt: str,
        cache_key: str | None = None,
        force_claude: bool = False,
        max_tokens: int = 2000,
        system: str | None = None,
    ) -> str:
        """
        Route prompt to local Ollama or Claude API.

        - If cache_key provided and result is cached: return immediately.
        - If USE_LOCAL_LLM=true and prompt is a simple task: use Ollama.
        - Otherwise: use Claude API.
        - If Ollama fails: fall back to Claude API.
        """
        if cache_key and cache_key in self._llm_cache:
            return self._llm_cache[cache_key]

        use_local = (
            USE_LOCAL_LLM
            and not force_claude
            and self._is_local_task(prompt)
        )

        result: str
        if use_local:
            try:
                result = await self._call_local(prompt)
            except Exception as e:
                self.logger.warning("Ollama failed (%s), falling back to Claude.", e)
                result = await self._call_claude(prompt, max_tokens=max_tokens, system=system)
        else:
            result = await self._call_claude(prompt, max_tokens=max_tokens, system=system)

        if cache_key:
            self._llm_cache[cache_key] = result

        return result

    def _is_local_task(self, prompt: str) -> bool:
        lower = prompt.lower()
        return any(kw in lower for kw in _LOCAL_TASK_KEYWORDS)

    async def _call_claude(self, prompt: str, max_tokens: int = 2000, system: str | None = None) -> str:
        if self._anthropic is None:
            self._anthropic = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        model = self.base_model or os.environ.get("AI_MODEL") or "claude-sonnet-4-6"

        # Retry on 529 overloaded with exponential backoff (max 4 attempts)
        import asyncio as _asyncio
        from anthropic import APIStatusError as _APIStatusError
        delays = [5, 15, 30]
        for attempt, delay in enumerate(delays + [None]):
            try:
                kwargs: dict = dict(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                if system:
                    # cache_control on system prompt — Anthropic caches the KV state
                    # after this block for up to 5 min, saving tokens on repeated calls.
                    kwargs["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
                response = await self._anthropic.messages.create(**kwargs)
                return response.content[0].text
            except _APIStatusError as e:
                if e.status_code != 529 or delay is None:
                    raise
                self.logger.warning(
                    "Anthropic API overloaded (attempt %d/%d), retrying in %ds…",
                    attempt + 1, len(delays) + 1, delay,
                )
                await _asyncio.sleep(delay)

    async def _call_local(self, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": LOCAL_MODEL, "prompt": prompt, "stream": False},
            )
            resp.raise_for_status()
            return resp.json()["response"]

    # ── SSE progress emission ─────────────────────────────────────────────────

    async def emit_progress(
        self,
        event: str,
        message: str,
        generation: int = 0,
        depth: int = 0,
        data: dict | None = None,
    ) -> None:
        """
        Publish a progress event to Redis pub/sub channel.
        Channel: research:{research_session_id}
        The /platform/research/progress/{id} SSE endpoint subscribes to this.
        """
        payload = {
            "event_type"          : event,
            "research_session_id" : self.research_session_id,
            "generation"          : generation,
            "depth"               : depth,
            "agent"               : self.__class__.__name__.replace("Agent", "").lower(),
            "message"             : message,
            "data"                : data or {},
            "timestamp"           : datetime.now(timezone.utc).isoformat(),
        }
        try:
            import redis.asyncio as aioredis
            r = aioredis.from_url(REDIS_URL, decode_responses=True)
            await r.publish(
                f"research:{self.research_session_id}",
                json.dumps(payload),
            )
            await r.aclose()
        except Exception as e:
            self.logger.warning("SSE emit failed: %s", e)

    # ── Structured logging ────────────────────────────────────────────────────

    def log_step(self, message: str, data: dict | None = None) -> None:
        self.logger.info(
            "[%s][%s] %s %s",
            self.research_session_id[:8],
            self.__class__.__name__,
            message,
            json.dumps(data) if data else "",
        )

    # ── Shared arXiv helper ───────────────────────────────────────────────────

    async def fetch_arxiv_papers(self, query: str, max_results: int = 5) -> list[dict]:
        """
        Search arXiv using the arxiv Python library (v2 API).
        Returns list of dicts with: title, arxiv_id, abstract, url, pdf_url, authors.
        """
        import arxiv

        def _search() -> list[dict]:
            client  = arxiv.Client()
            results = list(client.results(
                arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
            ))
            papers = []
            for r in results:
                papers.append({
                    "title"    : r.title,
                    "arxiv_id" : r.get_short_id().replace("v" + r.get_short_id().split("v")[-1], "") if "v" in r.get_short_id() else r.get_short_id(),
                    "abstract" : r.summary[:2000],
                    "url"      : str(r.entry_id),
                    "pdf_url"  : r.pdf_url,
                    "authors"  : [str(a) for a in r.authors[:3]],
                })
            return papers

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _search)

    # ── Shared Context7 helper ────────────────────────────────────────────────

    async def fetch_tf_docs(self, topic: str, tokens: int = 6000) -> str:
        """
        Fetch Keras/TF API docs from Context7.
        Returns combined text from keras.io + tensorflow/docs.
        """
        async with httpx.AsyncClient(timeout=20.0) as client:
            keras_task = client.get(
                f"https://context7.com/api/v1/websites/keras_io",
                params={"tokens": tokens, "topic": topic},
            )
            tf_task = client.get(
                f"https://context7.com/api/v1/tensorflow/docs",
                params={"tokens": tokens // 2, "topic": topic},
            )
            keras_res, tf_res = await asyncio.gather(keras_task, tf_task, return_exceptions=True)

        parts = []
        if not isinstance(keras_res, Exception) and keras_res.status_code == 200:
            parts.append(keras_res.text)
        if not isinstance(tf_res, Exception) and tf_res.status_code == 200:
            parts.append(tf_res.text)

        return "\n\n---\n\n".join(parts)
