"""
ArchitectAgent — proposes architecture mutations based on mechanisms.

Steps:
1. Load domain handler
2. Select base architecture(s) from domain templates
3. Call domain.propose_mutations() for each base arch
4. Return architecture_proposals in context
"""
from __future__ import annotations

import logging
import random

from .core import BaseAgent

logger = logging.getLogger(__name__)


class ArchitectAgent(BaseAgent):

    async def run(self, context: dict) -> dict:
        generation  = context.get("generation", 0)
        depth       = context.get("depth", 0)
        domain      = context.get("domain", "vision")
        category_id = context.get("category_id", domain)
        mechanisms  = context.get("candidate_mechanisms", [])

        await self.emit_progress("agent_start", "Architect proposing mutations...", generation, depth)
        self.log_step("Proposing mutations", {"domain": domain, "n_mechanisms": len(mechanisms)})

        from agents.domains import get_domain
        from agents.category_registry import get_default_architectures

        domain_handler  = get_domain(domain)
        base_archs      = domain_handler.list_architectures()

        # Load domain skill and novelty feedback; wrap call_llm to inject both
        domain_skill    = self.load_skill(domain=domain)
        novelty_feedback = context.get("novelty_feedback", "")
        skill_suffix = ""
        if domain_skill:
            skill_suffix += f"\n\n---\n\n{domain_skill}"
        if novelty_feedback:
            skill_suffix += f"\n\n---\n\n{novelty_feedback}"

        async def skill_llm(prompt, system=None, **kwargs):
            enhanced = (system or "") + skill_suffix
            return await self.call_llm(prompt, system=enhanced if enhanced.strip() else None, **kwargs)

        llm_caller = skill_llm if skill_suffix.strip() else self.call_llm
        failed_patterns = context.get("failed_patterns", [])

        # If recursing (depth > 0), seed with the winning base template from previous generation.
        # Use previous_winner_base_arch (e.g. "lstm") not previous_winner_arch ("lstm_mutant")
        # so that get_base_template() can look it up in domain.base_templates.
        prev_winner_base = context.get("previous_winner_base_arch") or context.get("previous_winner_arch")
        if prev_winner_base and depth > 0:
            # Only prepend if it's a known base template; avoid duplicates
            if prev_winner_base not in base_archs:
                base_archs = [prev_winner_base] + base_archs
            else:
                # Already in list — move it to front so it's prioritised
                base_archs = [prev_winner_base] + [a for a in base_archs if a != prev_winner_base]

        # Per-candidate mechanism sets from MathematicianAgent
        mechanism_sets: list[list[dict]] = context.get("mechanism_sets", [])

        population_size = context.get("population_size", 5)
        all_proposals   = []

        explored_summary = self._build_explored_summary(context)

        for base_arch in base_archs[:2]:  # max 2 base archs per generation
            proposals = await domain_handler.propose_mutations(
                base_arch,
                mechanisms,
                llm_caller=llm_caller,
                failed_patterns=failed_patterns or None,
                explored_summary=explored_summary or None,
            )
            all_proposals.extend(proposals)

        if len(all_proposals) > population_size:
            all_proposals = all_proposals[:population_size]

        # Attach each proposal's own mechanism set so Coder uses per-candidate papers
        for i, proposal in enumerate(all_proposals):
            if mechanism_sets:
                proposal["mechanisms"] = mechanism_sets[i % len(mechanism_sets)]

        await self.emit_progress(
            "agent_done",
            f"Architect proposed {len(all_proposals)} architectures",
            generation, depth,
            {"architectures": [p["architecture_name"] for p in all_proposals]},
        )

        context["architecture_proposals"] = all_proposals
        return context

    def _build_explored_summary(self, context: dict) -> str:
        """Summarise previously scored architectures so the LLM avoids re-exploring them.

        Reads 'scored_candidates' (CriticAgent) and 'generated_code' (CoderAgent) from
        context to produce a compact list of explored names, base templates, mutation
        patterns, and composite/novelty scores.
        """
        scored: list[dict] = context.get("scored_candidates", [])
        if not scored:
            return ""

        # Build a lookup from generated_code so we can surface mutation details
        code_lookup: dict[str, dict] = {
            g["architecture_name"]: g
            for g in context.get("generated_code", [])
        }

        lines = []
        for c in scored[:20]:  # cap to avoid bloating the prompt
            name  = c.get("architecture_name", "?")
            score = c.get("composite_score", 0)
            nov   = c.get("novelty_score", 0)
            code_entry = code_lookup.get(name, {})
            base  = code_entry.get("base_template", "?")
            muts  = code_entry.get("mutations", [])
            mut_str = ", ".join(muts) if muts else "unknown"
            lines.append(
                f"  - {name} (base={base}, mutations=[{mut_str}], "
                f"composite={score:.2f}, novelty={nov:.2f})"
            )

        return "\n".join(lines)
