"""
AgentCodeEditor — uses Claude tool_use to make surgical edits to existing Python code.

Instead of generating training scripts from scratch, this class gives Claude a set of
code-editing tools (read, str_replace, validate, finish) and lets it iteratively improve
an existing candidate. This is how a real researcher would work: read the existing code,
understand its structure, then apply targeted changes to implement new mechanisms.

Tools available to Claude during the editing loop:
  read_code      — view the current state of the code
  str_replace     — replace an exact string with a new one (like str_replace_editor)
  validate_syntax — run ast.parse and report any syntax errors
  finish          — commit the final code and exit the loop
"""
from __future__ import annotations

import ast
import json
import logging
import os
from typing import Any

from anthropic import AsyncAnthropic

from .domains.base_domain import TF_CODE_SYSTEM

logger = logging.getLogger(__name__)

EDITOR_SYSTEM = (
    TF_CODE_SYSTEM
    + """

EDITING RULES — you are pushing this architecture into unexplored territory:
- Use read_code first to understand the full current state of the code.
- Make TARGETED, SURGICAL edits via str_replace. Do not rewrite entire functions unless necessary.
- Each str_replace must use an exact substring that appears in the current code (including whitespace).
- After edits, use validate_syntax to confirm the code is still valid Python.
- If validate_syntax reports an error, fix it before calling finish.
- Call finish when all changes are in place and the code is valid.
- Maximum 12 tool-use turns before the loop exits automatically.

NOVELTY RULES (critical):
- Your goal is NOT to implement what existing papers describe. The mechanisms provided are hypotheses — \
open questions derived from reading the literature, not solutions from it.
- Do NOT reproduce standard patterns (vanilla attention, plain residuals, stock batch norm). \
If the existing code already uses these, REPLACE them with something genuinely new.
- Combine the provided mathematical ideas in ways that no published paper has tried. \
The value is in the untested combination, not in faithful reproduction.
- A working implementation of an untested idea is worth far more than a clean \
reimplementation of a known one.
"""
)


class AgentCodeEditor:
    """
    Runs a Claude tool-use loop to improve existing TensorFlow training code.

    Usage:
        editor = AgentCodeEditor(previous_code, model, api_key)
        improved_code = await editor.edit(user_prompt)
    """

    MAX_TURNS = 12

    def __init__(self, initial_code: str, model: str, api_key: str):
        self._code    = initial_code
        self._model   = model
        self._api_key = api_key

    @property
    def code(self) -> str:
        return self._code

    # ── Tool definitions ───────────────────────────────────────────────────────

    def _tools(self) -> list[dict]:
        return [
            {
                "name": "read_code",
                "description": "Read the current state of the Python script being improved.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "str_replace",
                "description": (
                    "Replace an exact substring in the code with a new string. "
                    "old_str must match the code exactly (whitespace included). "
                    "Use this for targeted, minimal edits — one logical change per call."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "old_str": {
                            "type": "string",
                            "description": "The exact string to find in the current code",
                        },
                        "new_str": {
                            "type": "string",
                            "description": "The replacement string",
                        },
                    },
                    "required": ["old_str", "new_str"],
                },
            },
            {
                "name": "validate_syntax",
                "description": "Run ast.parse on the current code. Returns OK or a SyntaxError description.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "finish",
                "description": "Signal that all improvements are complete. The current code will be returned as the final result.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
        ]

    # ── Tool execution ─────────────────────────────────────────────────────────

    def _handle_tool(self, name: str, inp: dict) -> tuple[str, bool]:
        """Returns (result_text, should_finish)."""
        if name == "read_code":
            return self._code, False

        elif name == "str_replace":
            old_str = inp.get("old_str", "")
            new_str = inp.get("new_str", "")
            if old_str not in self._code:
                # Give Claude a helpful hint so it can fix the mismatch
                snippet = self._code[:200].replace("\n", "\\n")
                return (
                    f"ERROR: The exact string was not found in the code. "
                    f"Use read_code to see the current state, then try again. "
                    f"Code starts with: {snippet}",
                    False,
                )
            self._code = self._code.replace(old_str, new_str, 1)
            return "OK: edit applied successfully", False

        elif name == "validate_syntax":
            try:
                ast.parse(self._code)
                return "OK: syntax is valid", False
            except SyntaxError as e:
                return f"SyntaxError at line {e.lineno}: {e.msg}", False

        elif name == "finish":
            return "Editing complete", True

        else:
            return f"Unknown tool: {name}", False

    def _tools_openai(self) -> list[dict]:
        """Convert tool definitions to OpenAI function-calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in self._tools()
        ]

    # ── Agentic edit loop ──────────────────────────────────────────────────────

    async def edit(self, user_prompt: str) -> str:
        """
        Run the agentic editing loop. Returns the final (improved) code.
        Routes to Anthropic or OpenAI-compatible backend based on AI_PROVIDER env var.
        """
        provider = os.environ.get("AI_PROVIDER", "anthropic").lower()
        if provider in ("lmstudio", "openai"):
            return await self._edit_openai(user_prompt)
        return await self._edit_anthropic(user_prompt)

    async def _edit_anthropic(self, user_prompt: str) -> str:
        messages: list[dict] = [{"role": "user", "content": user_prompt}]

        async with AsyncAnthropic(api_key=self._api_key) as client:
            for turn in range(self.MAX_TURNS):
                response = await client.messages.create(
                    model=self._model,
                    max_tokens=4000,
                    system=[{"type": "text", "text": EDITOR_SYSTEM, "cache_control": {"type": "ephemeral"}}],
                    tools=self._tools(),
                    messages=messages,
                )

                tool_uses = [b for b in response.content if b.type == "tool_use"]

                if not tool_uses:
                    logger.info("AgentCodeEditor: no tool calls on turn %d, exiting loop", turn + 1)
                    break

                messages.append({"role": "assistant", "content": response.content})

                tool_results = []
                finished = False
                for tu in tool_uses:
                    result_text, is_finish = self._handle_tool(tu.name, getattr(tu, "input", {}))
                    logger.debug("AgentCodeEditor tool=%s finished=%s result=%s", tu.name, is_finish, result_text[:80])
                    tool_results.append({
                        "type"       : "tool_result",
                        "tool_use_id": tu.id,
                        "content"    : result_text,
                    })
                    if is_finish:
                        finished = True

                messages.append({"role": "user", "content": tool_results})

                if finished:
                    logger.info("AgentCodeEditor: finished after %d turns", turn + 1)
                    break

        return self._code

    async def _edit_openai(self, user_prompt: str) -> str:
        from openai import AsyncOpenAI

        provider = os.environ.get("AI_PROVIDER", "lmstudio").lower()
        if provider == "lmstudio":
            base_url = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
            api_key  = os.environ.get("LMSTUDIO_API_KEY", "lm-studio")
        else:
            base_url = None
            api_key  = os.environ.get("OPENAI_API_KEY", "")

        messages: list[dict] = [
            {"role": "system", "content": EDITOR_SYSTEM},
            {"role": "user",   "content": user_prompt},
        ]

        async with AsyncOpenAI(base_url=base_url, api_key=api_key) as client:
            for turn in range(self.MAX_TURNS):
                response = await client.chat.completions.create(
                    model=self._model,
                    max_tokens=4000,
                    tools=self._tools_openai(),
                    tool_choice="auto",
                    messages=messages,
                )

                msg = response.choices[0].message
                tool_calls = msg.tool_calls or []

                if not tool_calls:
                    logger.info("AgentCodeEditor: no tool calls on turn %d, exiting loop", turn + 1)
                    break

                # Append assistant message with tool_calls
                messages.append({
                    "role"      : "assistant",
                    "content"   : msg.content or "",
                    "tool_calls": [
                        {
                            "id"      : tc.id,
                            "type"    : "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in tool_calls
                    ],
                })

                finished = False
                for tc in tool_calls:
                    try:
                        inp = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        inp = {}
                    result_text, is_finish = self._handle_tool(tc.function.name, inp)
                    logger.debug("AgentCodeEditor tool=%s finished=%s result=%s", tc.function.name, is_finish, result_text[:80])
                    messages.append({
                        "role"        : "tool",
                        "tool_call_id": tc.id,
                        "content"     : result_text,
                    })
                    if is_finish:
                        finished = True

                if finished:
                    logger.info("AgentCodeEditor: finished after %d turns", turn + 1)
                    break

        return self._code
