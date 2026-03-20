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

from .core import BaseAgent
from .safety_checker import validate_code

logger = logging.getLogger(__name__)


class CoderAgent(BaseAgent):

    async def run(self, context: dict) -> dict:
        generation = context.get("generation", 0)
        depth      = context.get("depth", 0)
        domain     = context.get("domain", "vision")
        proposals  = context.get("architecture_proposals", [])

        await self.emit_progress("agent_start", f"Coder generating code for {len(proposals)} architectures...", generation, depth)
        self.log_step("Generating code", {"domain": domain, "n_proposals": len(proposals)})

        from agents.domains import get_domain
        domain_handler = get_domain(domain)

        # Generate code for all proposals in parallel
        tasks = [
            self._generate_single(domain_handler, p, generation, depth)
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

    async def _generate_single(self, domain_handler, proposal: dict, generation: int, depth: int) -> dict | None:
        arch_name = proposal.get("architecture_name", "unknown")
        arch_spec = proposal.get("spec", {})

        try:
            code = await domain_handler.generate_code(arch_spec, llm_caller=self.call_llm)

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
            }

        except Exception as e:
            self.log_step(f"Code generation exception for {arch_name}", {"error": str(e)})
            return None

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

    def _estimate_params(self, code: str) -> int:
        """Rough heuristic: count Dense/Conv layer sizes mentioned in code."""
        import re
        # Look for units=N or filters=N
        units   = sum(int(m) for m in re.findall(r'units\s*=\s*(\d+)', code))
        filters = sum(int(m) for m in re.findall(r'filters\s*=\s*(\d+)', code))
        return (units + filters) * 100  # rough multiplier
