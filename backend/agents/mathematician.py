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
        generation        = context.get("generation", 0)
        depth             = context.get("depth", 0)
        domain            = context.get("domain", "vision")
        task_description  = context.get("task_description", "")
        research_insights = context.get("research_insights", "")

        await self.emit_progress("agent_start", "Mathematician deriving mechanisms...", generation, depth)
        self.log_step("Deriving mechanisms", {"domain": domain})

        # Query the session vectorstore for targeted content from the actual papers.
        # This gives the mechanism derivation real methods/equations from the papers
        # rather than the compressed abstract-level summary.
        grounded_content = await self._query_vectorstore_for_mechanisms(domain, task_description)

        # Combine vectorstore content with the abstract-level summary as fallback.
        # Vectorstore content takes precedence (actual paper text); summary provides
        # high-level framing if the store is empty or sparsely populated.
        if grounded_content:
            full_insights = (
                f"EXTRACTED PAPER CONTENT (methods, equations, results):\n{grounded_content}"
                + (f"\n\nRESEARCH SUMMARY:\n{research_insights}" if research_insights else "")
            )
        else:
            full_insights = research_insights

        # Get domain handler to provide domain-specific mechanism generation
        from agents.domains import get_domain
        domain_handler = get_domain(domain)

        mechanisms = await domain_handler.generate_mechanism(
            full_insights,
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

    async def _query_vectorstore_for_mechanisms(self, domain: str, task_description: str) -> str:
        """Query the session FAISS store for content relevant to mathematical mechanisms."""
        try:
            import os
            from pathlib import Path
            from vectorstore import query

            session_dir = Path(os.environ.get("RESEARCH_ARTIFACTS_DIR", "/research_artifacts")) / self.research_session_id

            goal_hint = task_description[:80] if task_description else domain
            queries = [
                f"mathematical mechanism equation formula {domain}",
                f"novel architecture method {goal_hint}",
                f"attention mechanism computation {domain}",
                f"loss function training objective {domain}",
            ]

            seen_sources: set[str] = set()
            chunks: list[str] = []
            for q in queries:
                results = query(session_dir, q, k=4)
                for r in results:
                    src = r.get("source", "")
                    # Deduplicate by (source, chunk_i) to avoid repeating the same passage
                    key = f"{src}:{r.get('chunk_i', 0)}"
                    if key not in seen_sources and r.get("score", 0) > 0.3:
                        seen_sources.add(key)
                        title = r.get("title", "")
                        header = f"[{title}]" if title else f"[{src}]"
                        chunks.append(f"{header}\n{r['text']}")

            return "\n\n---\n\n".join(chunks[:12])   # cap at 12 chunks to avoid bloat
        except Exception as e:
            self.log_step("Vectorstore query failed — using summary fallback", {"error": str(e)})
            return ""

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
