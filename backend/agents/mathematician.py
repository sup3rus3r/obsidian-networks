"""
MathematicianAgent — derives novel mathematical mechanisms from research insights.

Steps:
1. Call LLM to extract 3 novel mechanisms from research insights
2. Generate SymPy expressions for each mechanism
3. Validate SymPy expressions are parseable
4. Return updated context with candidate_mechanisms
"""
from __future__ import annotations

import json
import logging

from .core import BaseAgent

logger = logging.getLogger(__name__)


class MathematicianAgent(BaseAgent):

    async def run(self, context: dict) -> dict:
        generation       = context.get("generation", 0)
        depth            = context.get("depth", 0)
        domain           = context.get("domain", "vision")
        research_insights = context.get("research_insights", "")

        await self.emit_progress("agent_start", "Mathematician deriving mechanisms...", generation, depth)
        self.log_step("Deriving mechanisms", {"domain": domain})

        # Get domain handler to provide domain-specific mechanism generation
        from agents.domains import get_domain
        domain_handler = get_domain(domain)

        mechanisms = await domain_handler.generate_mechanism(
            research_insights,
            llm_caller=self.call_llm,
        )

        # Validate SymPy expressions
        validated = []
        for m in mechanisms:
            expr_str = m.get("sympy_expression", "")
            is_valid = self._validate_sympy(expr_str)
            m["sympy_valid"] = is_valid
            validated.append(m)
            self.log_step(f"Mechanism: {m.get('name')}", {"valid": is_valid, "expr": expr_str[:80]})

        await self.emit_progress(
            "agent_done",
            f"Mathematician derived {len(validated)} mechanisms",
            generation, depth,
            {"mechanisms": [m["name"] for m in validated]},
        )

        context["candidate_mechanisms"] = validated
        return context

    def _validate_sympy(self, expr_str: str) -> bool:
        """Attempt to parse the expression with sympy. Return True if parseable."""
        if not expr_str:
            return False
        try:
            import sympy
            # Replace common notation that sympy might not accept
            clean = (
                expr_str
                .replace("^", "**")
                .replace("∑", "sum")
                .replace("∏", "prod")
            )
            sympy.sympify(clean)
            return True
        except Exception:
            # Not parseable, but still keep the mechanism — it's descriptive
            return False
