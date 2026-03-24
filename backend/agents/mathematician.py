"""
MathematicianAgent — derives novel mathematical mechanisms from research insights.

Steps:
1. For each per-slot insight set (from ResearcherAgent), independently derive mechanisms
2. Generate SymPy expressions for each mechanism
3. Validate SymPy expressions are parseable
4. Return mechanism_sets (one per candidate slot) + backward-compat candidate_mechanisms
"""
from __future__ import annotations

import asyncio

import json
import logging

from .core import BaseAgent

logger = logging.getLogger(__name__)


class MathematicianAgent(BaseAgent):

    async def run(self, context: dict) -> dict:
        generation       = context.get("generation", 0)
        depth            = context.get("depth", 0)
        domain           = context.get("domain", "vision")
        task_description = context.get("task_description", "")

        # Per-slot insight sets from ResearcherAgent — one set per candidate
        insight_sets: list[str] = context.get("research_insight_sets", [])
        if not insight_sets:
            insight_sets = [context.get("research_insights", "")]

        await self.emit_progress("agent_start", "Mathematician deriving mechanisms...", generation, depth)
        self.log_step("Deriving mechanisms", {"domain": domain, "n_slots": len(insight_sets)})

        from agents.domains import get_domain
        domain_handler   = get_domain(domain)
        grounded_content = await self._query_vectorstore_for_mechanisms(domain, task_description)

        # Load mathematician skill and novelty feedback; wrap call_llm to inject both
        math_skill       = self.load_skill(filename="novel_mechanisms.md")
        novelty_feedback = context.get("novelty_feedback", "")
        skill_suffix = ""
        if math_skill:
            skill_suffix += f"\n\n---\n\n{math_skill}"
        if novelty_feedback:
            skill_suffix += f"\n\n---\n\n{novelty_feedback}"

        async def skill_llm(prompt, system=None, **kwargs):
            enhanced = (system or "") + skill_suffix
            return await self.call_llm(prompt, system=enhanced if enhanced.strip() else None, **kwargs)

        llm_caller = skill_llm if skill_suffix.strip() else self.call_llm

        # Derive one independent mechanism set per candidate slot in parallel
        tasks = [
            self._derive_for_slot(insights, grounded_content, domain_handler, llm_caller)
            for insights in insight_sets
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        mechanism_sets: list[list[dict]] = []
        all_names: list[str] = []
        for r in results:
            if isinstance(r, Exception):
                self.log_step("Mechanism derivation failed for slot", {"error": str(r)})
                mechanism_sets.append([])
            else:
                mechanism_sets.append(r)
                all_names.extend(m["name"] for m in r)

        await self.emit_progress(
            "agent_done",
            f"Mathematician derived {len(all_names)} mechanisms across {len(mechanism_sets)} slots",
            generation, depth,
            {"mechanisms": all_names},
        )

        # Backward-compat single key + per-slot sets for ArchitectAgent
        context["candidate_mechanisms"] = mechanism_sets[0] if mechanism_sets else []
        context["mechanism_sets"]       = mechanism_sets
        return context

    async def _derive_for_slot(
        self,
        research_insights: str,
        grounded_content: str,
        domain_handler,
        llm_caller=None,
    ) -> list[dict]:
        if llm_caller is None:
            llm_caller = self.call_llm

        full_insights = (
            f"EXTRACTED PAPER CONTENT (methods, equations, results):\n{grounded_content}"
            + (f"\n\nRESEARCH SUMMARY:\n{research_insights}" if research_insights else "")
        ) if grounded_content else research_insights

        mechanisms = await domain_handler.generate_mechanism(full_insights, llm_caller=llm_caller)

        if not mechanisms:
            # Model returned an empty array (parsed cleanly but no mechanisms).
            # Retry once with an explicit non-empty instruction appended to the insights.
            self.log_step("Empty mechanism list — retrying once", {})
            retry_insights = (
                full_insights
                + "\n\nCRITICAL: You returned an empty array. That is not acceptable. "
                "You MUST return at least 2 mechanisms. Tier 2 (known mechanisms in new "
                "contexts) is fine — return those rather than nothing."
            )
            mechanisms = await domain_handler.generate_mechanism(retry_insights, llm_caller=llm_caller)

        for m in mechanisms:
            m["sympy_valid"] = self._validate_sympy(m.get("sympy_expression", ""))
            self.log_step(f"Mechanism: {m.get('name')}", {"expr": m.get("sympy_expression", "")[:80]})
        return mechanisms

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
