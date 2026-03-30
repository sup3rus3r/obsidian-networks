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
from pathlib import Path

import httpx
from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)

REDIS_URL     = os.environ.get("REDIS_URL", "redis://redis:6379/0")
OLLAMA_URL    = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
LOCAL_MODEL   = os.environ.get("LOCAL_LLM_MODEL", "mistral")
USE_LOCAL_LLM = os.environ.get("USE_LOCAL_LLM", "false").lower() == "true"

# Tasks that can safely run on a smaller/local model
_LOCAL_TASK_KEYWORDS = {"extract", "classify", "summarize", "parse", "rate", "score", "compare", "list"}

_SKILLS_DIR = Path(__file__).parent / "skills"


class BaseAgent(ABC):
    """Abstract base class for all Research Labs agents."""

    def __init__(self, research_session_id: str, base_model: str, domain: str):
        self.research_session_id = research_session_id
        self.base_model          = base_model
        self.domain              = domain
        self.logger              = logging.getLogger(f"agents.{self.__class__.__name__}")
        self._llm_cache: dict[str, str]   = {}
        self._skill_cache: dict[str, str] = {}

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
                self.logger.warning("Ollama failed (%s), falling back to primary LLM.", e)
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
        provider = os.environ.get("AI_PROVIDER", "anthropic").lower()
        model = self.base_model or os.environ.get("AI_MODEL")

        if provider in ("lmstudio", "openai"):
            return await self._call_openai_compatible(prompt, max_tokens=max_tokens, system=system, model=model)

        if provider == "gemini":
            return await self._call_gemini(prompt, max_tokens=max_tokens, system=system, model=model)

        model = model or "claude-sonnet-4-6"

        # Retry on 529 overloaded with exponential backoff (max 4 attempts)
        import asyncio as _asyncio
        from anthropic import APIStatusError as _APIStatusError
        delays = [5, 15, 30]
        # Use client as a context manager so httpx connections close cleanly even
        # when the Celery event loop shuts down mid-task (avoids RuntimeError: Event loop is closed)
        async with AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY")) as client:
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
                    response = await client.messages.create(**kwargs)
                    return response.content[0].text
                except _APIStatusError as e:
                    if e.status_code != 529 or delay is None:
                        raise
                    self.logger.warning(
                        "Anthropic API overloaded (attempt %d/%d), retrying in %ds…",
                        attempt + 1, len(delays) + 1, delay,
                    )
                    await _asyncio.sleep(delay)

    async def _call_openai_compatible(
        self, prompt: str, max_tokens: int = 2000, system: str | None = None, model: str | None = None
    ) -> str:
        from openai import AsyncOpenAI

        provider = os.environ.get("AI_PROVIDER", "lmstudio").lower()
        if provider == "lmstudio":
            base_url = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
            api_key  = os.environ.get("LMSTUDIO_API_KEY", "lm-studio")
            model    = model or "local-model"
        else:
            base_url = None
            api_key  = os.environ.get("OPENAI_API_KEY")
            model    = model or "gpt-4o"

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async with AsyncOpenAI(base_url=base_url, api_key=api_key) as client:
            response = await client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
            )
            return response.choices[0].message.content

    async def _call_gemini(
        self, prompt: str, max_tokens: int = 2000, system: str | None = None, model: str | None = None
    ) -> str:
        from google import genai
        from google.genai import types

        api_key = os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY")
        model   = model or "gemini-2.5-flash"

        client = genai.Client(api_key=api_key)

        config = types.GenerateContentConfig(max_output_tokens=max_tokens)
        if system:
            config.system_instruction = system

        contents = prompt
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            ),
        )
        return response.text

    async def _call_local(self, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": LOCAL_MODEL, "prompt": prompt, "stream": False},
            )
            resp.raise_for_status()
            return resp.json()["response"]

    # ── Skills loading ────────────────────────────────────────────────────────

    def load_skill(self, domain: str | None = None, filename: str | None = None) -> str | None:
        """
        Load a skill file for this agent and return the markdown body (frontmatter stripped).

        Resolution order:
        - domain   -> skills/<agent>/<domain>.md
        - filename -> skills/<agent>/<filename>

        Returns None if the file does not exist — agent runs without skill, no crash.
        Skill bodies are cached in memory for the agent's lifetime.
        """
        agent_folder = self.__class__.__name__.replace("Agent", "").lower()

        if domain:
            skill_path = _SKILLS_DIR / agent_folder / f"{domain}.md"
        elif filename:
            skill_path = _SKILLS_DIR / agent_folder / filename
        else:
            return None

        cache_key = str(skill_path)
        if cache_key in self._skill_cache:
            return self._skill_cache[cache_key]

        if not skill_path.exists():
            self.logger.debug("No skill file at %s — running without skill.", skill_path)
            return None

        raw  = skill_path.read_text(encoding="utf-8")
        body = self._strip_skill_frontmatter(raw)
        self._skill_cache[cache_key] = body
        self.logger.debug("Loaded skill from %s (%d chars).", skill_path, len(body))
        return body

    @staticmethod
    def _strip_skill_frontmatter(text: str) -> str:
        """Remove YAML frontmatter (--- ... ---) from the top of a skill file."""
        if not text.startswith("---"):
            return text
        end = text.find("---", 3)
        if end == -1:
            return text
        return text[end + 3:].lstrip("\n")

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
            import datetime
            current_year = datetime.datetime.now().year
            date_filter  = f"submittedDate:[{current_year - 2}0101 TO {current_year}1231]"
            dated_query  = f"{query} {date_filter}"
            client  = arxiv.Client(num_retries=1, delay_seconds=1)
            results = list(client.results(
                arxiv.Search(query=dated_query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
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

        self.log_step("arXiv search starting", {"query": query, "max_results": max_results, "date_filter": f"{__import__('datetime').datetime.now().year - 2}-{__import__('datetime').datetime.now().year}"})
        loop = asyncio.get_event_loop()
        papers = await loop.run_in_executor(None, _search)
        self.log_step(f"arXiv search done", {"query": query, "found": len(papers)})
        return papers

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
