"""
CoderAgent — generates executable training code for each architecture proposal.

Steps:
1. For each architecture proposal, call domain.generate_code()
2. Run safety validation on generated code
3. Count parameters via AST analysis
4. Return generated_code list in context
"""
from __future__ import annotations

import asyncio
import logging
import os

from .core import BaseAgent
from .safety_checker import validate_code

logger = logging.getLogger(__name__)


class CoderAgent(BaseAgent):

    async def run(self, context: dict) -> dict:
        generation           = context.get("generation", 0)
        depth                = context.get("depth", 0)
        domain               = context.get("domain", "vision")
        proposals            = context.get("architecture_proposals", [])
        previous_winner_code  = context.get("previous_winner_code", "")
        previous_winner_score = context.get("previous_winner_score", 0.0)

        await self.emit_progress("agent_start", f"Coder generating code for {len(proposals)} architectures...", generation, depth)
        self.log_step("Generating code", {"domain": domain, "n_proposals": len(proposals), "improvement_mode": bool(previous_winner_code)})

        from agents.domains import get_domain
        domain_handler = get_domain(domain)

        # Load domain skill; wrap call_llm so skill is injected into all code generation calls
        domain_skill = self.load_skill(domain=domain)

        async def skill_llm(prompt, system=None, **kwargs):
            enhanced = (system or "") + (f"\n\n---\n\n{domain_skill}" if domain_skill else "")
            return await self.call_llm(prompt, system=enhanced if enhanced.strip() else None, **kwargs)

        self._skill_llm = skill_llm if domain_skill else self.call_llm

        shared_mechanisms = context.get("candidate_mechanisms", [])

        # Generate code for all proposals in parallel
        # Each proposal may carry its own mechanism set (from per-slot research);
        # fall back to the shared mechanisms if not present.
        tasks = [
            self._generate_single(
                domain_handler, p,
                p.get("mechanisms") or shared_mechanisms,
                generation, depth,
                previous_winner_code=previous_winner_code,
                previous_winner_score=previous_winner_score,
            )
            for p in proposals
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        generated_code = []
        for r in results:
            if isinstance(r, Exception):
                self.log_step("Code generation failed", {"error": str(r)})
            elif r is not None:
                generated_code.append(r)

        await self.emit_progress(
            "agent_done",
            f"Coder produced {len(generated_code)} valid scripts",
            generation, depth,
            {"architectures": [g["architecture_name"] for g in generated_code]},
        )

        context["generated_code"] = generated_code
        return context

    async def _generate_single(
        self,
        domain_handler,
        proposal: dict,
        mechanisms: list[dict],
        generation: int,
        depth: int,
        previous_winner_code: str = "",
        previous_winner_score: float = 0.0,
    ) -> dict | None:
        arch_name = proposal.get("architecture_name", "unknown")
        arch_spec = proposal.get("spec", {})
        rationale = proposal.get("rationale", "")

        try:
            if previous_winner_code:
                # Improvement mode: use the agentic code editor to make surgical edits
                # to the previous winner's code rather than generating from scratch.
                code = await self._improve_code(
                    previous_winner_code,
                    previous_winner_score,
                    mechanisms,
                    arch_name,
                    rationale,
                )
            else:
                code = await domain_handler.generate_code(
                    arch_spec,
                    llm_caller=self._skill_llm,
                    mechanisms=mechanisms,
                    rationale=rationale,
                )

            # If mechanisms were specified, verify the code actually implements custom layers.
            # If it's just standard Keras layers, the novel mechanism was silently dropped —
            # force a targeted retry before the safety check.
            if mechanisms and not self._has_custom_layers(code):
                self.log_step(f"No custom layers in {arch_name} — forcing mechanism implementation", {})
                code = await self._force_mechanism_implementation(code, mechanisms, rationale)

            # Safety check
            is_safe, violations = validate_code(code)
            if not is_safe:
                self.log_step(f"Safety check failed for {arch_name}", {"violations": violations})
                # Attempt to fix by regenerating once
                code = await self._fix_code(code, violations, domain_handler, arch_spec)
                is_safe, violations = validate_code(code)
                if not is_safe:
                    return None

            param_count = self._estimate_params(code)

            return {
                "architecture_name": arch_name,
                "code"             : code,
                "framework"        : "tensorflow",
                "param_count"      : param_count,
                "spec"             : arch_spec,
                "base_template"    : proposal.get("base_template", ""),
                "mutations"        : proposal.get("mutations", []),
                "rationale"        : proposal.get("rationale", ""),
            }

        except Exception as e:
            self.log_step(f"Code generation exception for {arch_name}", {"error": str(e)})
            return None

    async def _improve_code(
        self,
        previous_code: str,
        previous_score: float,
        mechanisms: list[dict],
        arch_name: str,
        rationale: str,
    ) -> str:
        """
        Use the agentic AgentCodeEditor to improve the previous winner's code.
        Claude receives read/edit/validate tools so it can make surgical changes
        rather than regenerating the entire script from scratch.
        """
        from .code_editor import AgentCodeEditor

        mech_lines = "\n".join(
            f"  - {m.get('name', '?')}: {m.get('description', '')}"
            + (f"\n    Math: {m['sympy_expression']}" if m.get("sympy_expression") else "")
            for m in mechanisms
        )

        prompt = (
            f"You are pushing an existing architecture into unexplored territory.\n\n"
            f"Current architecture: {arch_name}\n"
            f"Previous score: {previous_score:.1%} — discover something genuinely better, not just a tweak\n"
            f"Research hypothesis: {rationale}\n\n"
            f"Novel mathematical hypotheses to explore (OPEN QUESTIONS, not recipes):\n{mech_lines}\n\n"
            f"Instructions:\n"
            f"1. Use read_code to understand the existing code\n"
            f"2. Identify what to REPLACE or fundamentally change — don't just append to existing patterns\n"
            f"3. Use str_replace to make targeted edits — do NOT rewrite the whole script\n"
            f"4. Implement these ideas in ways no published paper has tried — combine, invert, reframe them\n"
            f"   A working untested idea beats a clean copy of a known one\n"
            f"5. Use validate_syntax to confirm the code is valid\n"
            f"6. Call finish when done\n\n"
            f"The script must remain self-contained with synthetic data and save to output/model.keras."
        )

        editor = AgentCodeEditor(
            initial_code=previous_code,
            model=self.base_model or os.environ.get("AI_MODEL", "claude-sonnet-4-6"),
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        )
        return await editor.edit(prompt)

    async def _fix_code(self, code: str, violations: list[str], domain_handler, arch_spec: dict) -> str:
        """Ask LLM to fix safety violations."""
        violation_text = "\n".join(f"- {v}" for v in violations)
        prompt = f"""
The following Python code has safety violations. Fix them:

Violations:
{violation_text}

Code:
```python
{code[:3000]}
```

Return ONLY the fixed Python code. Remove any forbidden imports or patterns.
"""
        fixed = await self.call_llm(prompt, force_claude=True, max_tokens=3000)
        if "```python" in fixed: fixed = fixed.split("```python")[1].split("```")[0]
        elif "```" in fixed: fixed = fixed.split("```")[1].split("```")[0]
        return fixed.strip()

    def _has_custom_layers(self, code: str) -> bool:
        """Return True if the code defines at least one custom tf.keras.layers.Layer subclass."""
        import re
        return bool(re.search(r'class\s+\w+\s*\(\s*tf\.keras\.layers\.Layer', code))

    async def _force_mechanism_implementation(
        self,
        code: str,
        mechanisms: list[dict],
        rationale: str,
    ) -> str:
        """
        The generated code has no custom layers — the LLM fell back to standard Keras.
        Force a targeted rewrite that MUST implement each mechanism as a custom layer.
        """
        mech_lines = "\n".join(
            f"  - {m.get('name', '?')}: {m.get('description', '')}"
            + (f"\n    Math: {m['sympy_expression']}" if m.get("sympy_expression") else "")
            for m in mechanisms
        )
        prompt = (
            f"The following script was generated but does NOT implement the required novel mechanisms "
            f"— it only uses standard Keras layers. This is incorrect.\n\n"
            f"Mechanisms that MUST be implemented as tf.keras.layers.Layer subclasses:\n{mech_lines}\n\n"
            f"Rationale: {rationale}\n\n"
            f"Current (incorrect) script:\n```python\n{code[:3000]}\n```\n\n"
            f"Rewrite the script so that:\n"
            f"1. Each mechanism above appears as a Python class inheriting tf.keras.layers.Layer\n"
            f"2. The class overrides call() with the actual mathematical operation described\n"
            f"3. The model uses these custom layers as core components\n"
            f"4. All other TF coding rules are preserved (synthetic data, output/model.keras, etc.)\n\n"
            f"Return ONLY the complete Python script. No markdown fences."
        )
        fixed = await self.call_llm(prompt, force_claude=True, max_tokens=4000)
        if "```python" in fixed: fixed = fixed.split("```python")[1].split("```")[0]
        elif "```" in fixed:     fixed = fixed.split("```")[1].split("```")[0]
        return fixed.strip()

    def _estimate_params(self, code: str) -> int:
        """Rough heuristic: count Dense/Conv layer sizes mentioned in code."""
        import re
        # Look for units=N or filters=N
        units   = sum(int(m) for m in re.findall(r'units\s*=\s*(\d+)', code))
        filters = sum(int(m) for m in re.findall(r'filters\s*=\s*(\d+)', code))
        return (units + filters) * 100  # rough multiplier
