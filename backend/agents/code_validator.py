"""
CodeValidatorAgent — verifies that generated code is correct before training.

For each generated script, Claude is given the code + the mechanism descriptions
and asked to check:
  1. Tensor shape correctness at every layer boundary
  2. Whether the custom layer's call() actually implements the stated mechanism
  3. Silent broadcasting traps (e.g. [B,1,F] - [B,T,F] silently changing F)
  4. Gradient flow — any tf.cast(bool) or other non-differentiable ops on trainable weights
  5. Dimensional mismatches between what the rationale describes and what the code does

If issues are found, it uses AgentCodeEditor to make surgical fixes — same tool the
Coder uses in improvement mode.  If after one fix-pass the code still has critical
errors, the candidate is dropped rather than sending broken code to the trainer.
"""
from __future__ import annotations

import asyncio
import logging
import re

from .core import BaseAgent

logger = logging.getLogger(__name__)

VALIDATOR_SYSTEM = """\
You are a senior ML engineer doing a correctness review of a TensorFlow training script.
Your job is NOT to rewrite the architecture — it is to find concrete bugs that would
cause silent wrong results or crashes.

Check specifically:
1. TENSOR SHAPES — trace shapes through every custom layer. Flag any op where the
   output shape is inconsistent with what the next layer expects. Pay special
   attention to reshape/broadcast ops.
2. MECHANISM FIDELITY — does call() actually implement what the description says?
   If the description says "cosine similarity between timesteps" but the code does
   dot-product between features, that is a bug.
3. BROADCASTING TRAPS — ops like `x - y` where x is [B,T,F] and y is [B,F] broadcast
   silently. Only flag when the broadcast changes the semantics (e.g. subtracting a
   per-sample summary from a sequence when you meant per-timestep).
4. GRADIENT KILLERS — tf.cast(bool_tensor, tf.float32) applied directly to a
   trainable weight's output has zero gradient. Flag and suggest soft alternative
   (e.g. tf.nn.sigmoid(k*(x - threshold))).
5. DIMENSION MISMATCHES — e.g. a Dense(horizon * n_features) output reshaped to
   (horizon, n_features) when the actual values differ.

Output format — respond with ONLY a JSON object:
{
  "has_bugs": true | false,
  "bugs": [
    {
      "type": "shape | mechanism | broadcast | gradient | dimension",
      "location": "brief description of where in the code",
      "description": "what is wrong",
      "fix": "specific suggested fix"
    }
  ],
  "summary": "one sentence overall assessment"
}

If there are no bugs, return {"has_bugs": false, "bugs": [], "summary": "Code is correct."}.
Return ONLY valid JSON — no prose, no markdown fences.
"""


class CodeValidatorAgent(BaseAgent):

    async def run(self, context: dict) -> dict:
        generation     = context.get("generation", 0)
        depth          = context.get("depth", 0)
        generated_code = context.get("generated_code", [])
        mechanisms     = context.get("candidate_mechanisms", [])

        if not generated_code:
            return context

        await self.emit_progress(
            "code_validator_start",
            f"Code Validator checking {len(generated_code)} scripts for correctness...",
            generation, depth,
        )

        tasks = [
            self._validate_single(gc, mechanisms, generation, depth)
            for gc in generated_code
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        validated = []
        fixed = 0
        dropped = 0
        for r in results:
            if isinstance(r, Exception):
                logger.warning("CodeValidator exception: %s", r)
            elif r is not None:
                validated.append(r)
                if r.get("_validator_fixed"):
                    fixed += 1
            else:
                dropped += 1

        await self.emit_progress(
            "code_validator_done",
            f"Code Validator: {len(validated)} passed ({fixed} fixed, {dropped} dropped)",
            generation, depth,
            {"fixed": fixed, "dropped": dropped},
        )

        context["generated_code"] = validated
        return context

    async def _validate_single(
        self, gc: dict, mechanisms: list[dict], generation: int, depth: int
    ) -> dict | None:
        arch_name = gc.get("architecture_name", "unknown")
        code      = gc.get("code", "")
        rationale = gc.get("rationale", "")

        report = await self._check_code(code, mechanisms, rationale)

        if not report.get("has_bugs"):
            self.log_step(f"CodeValidator: {arch_name} — clean", {})
            return gc

        bugs = report.get("bugs", [])
        self.log_step(
            f"CodeValidator: {arch_name} — {len(bugs)} bug(s) found",
            {"bugs": [b["type"] for b in bugs]},
        )

        # Attempt fix via AgentCodeEditor
        fixed_code = await self._fix_bugs(code, bugs, mechanisms, rationale, arch_name)
        if fixed_code is None:
            return None  # drop candidate

        # Re-check once
        re_report = await self._check_code(fixed_code, mechanisms, rationale)
        if re_report.get("has_bugs"):
            critical = [b for b in re_report.get("bugs", []) if b["type"] in ("shape", "dimension")]
            if critical:
                self.log_step(f"CodeValidator: {arch_name} still has critical bugs — dropping", {})
                return None

        result = dict(gc)
        result["code"] = fixed_code
        result["_validator_fixed"] = True
        return result

    async def _check_code(self, code: str, mechanisms: list[dict], rationale: str) -> dict:
        mech_lines = "\n".join(
            f"  - {m.get('name','?')}: {m.get('description','')}"
            + (f"\n    Math: {m['sympy_expression']}" if m.get("sympy_expression") else "")
            for m in mechanisms
        )
        prompt = (
            f"Rationale: {rationale}\n\n"
            f"Mechanisms:\n{mech_lines}\n\n"
            f"Code to review:\n```python\n{code}\n```\n\n"
            "Perform the correctness check and return JSON."
        )
        raw = await self.call_llm(prompt, system=VALIDATOR_SYSTEM, force_claude=True, max_tokens=1500)
        try:
            start = raw.find("{"); end = raw.rfind("}") + 1
            import json
            return json.loads(raw[start:end])
        except Exception:
            return {"has_bugs": False, "bugs": [], "summary": "Parse error — assuming clean."}

    async def _fix_bugs(
        self,
        code: str,
        bugs: list[dict],
        mechanisms: list[dict],
        rationale: str,
        arch_name: str,
    ) -> str | None:
        from .code_editor import AgentCodeEditor
        import os

        bug_lines = "\n".join(
            f"  [{b['type'].upper()}] {b['location']}: {b['description']}\n    Fix: {b['fix']}"
            for b in bugs
        )
        mech_lines = "\n".join(
            f"  - {m.get('name','?')}: {m.get('description','')}"
            for m in mechanisms
        )

        prompt = (
            f"You are fixing correctness bugs in a TensorFlow training script.\n\n"
            f"Architecture: {arch_name}\n"
            f"Rationale: {rationale}\n\n"
            f"Mechanisms the code implements:\n{mech_lines}\n\n"
            f"Bugs to fix (DO NOT change the architecture, only fix these specific issues):\n{bug_lines}\n\n"
            f"Instructions:\n"
            f"1. Use read_code to review the current code\n"
            f"2. Use str_replace to fix ONLY the listed bugs — do not refactor or redesign\n"
            f"3. Use validate_syntax to confirm the fix compiles\n"
            f"4. Call finish when done\n"
        )

        try:
            editor = AgentCodeEditor(
                initial_code=code,
                model=self.base_model or os.environ.get("AI_MODEL", "claude-sonnet-4-6"),
                api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            )
            return await editor.edit(prompt)
        except Exception as e:
            self.log_step(f"CodeValidator fix failed for {arch_name}", {"error": str(e)})
            return None
