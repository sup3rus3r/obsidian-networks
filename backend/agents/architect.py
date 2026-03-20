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

        all_proposals = []
        for base_arch in base_archs[:2]:  # max 2 base archs per generation
            proposals = await domain_handler.propose_mutations(
                base_arch,
                mechanisms,
                llm_caller=self.call_llm,
                failed_patterns=failed_patterns or None,
            )
            all_proposals.extend(proposals)

        # Limit to population_size
        population_size = context.get("population_size", 5)
        if len(all_proposals) > population_size:
            all_proposals = all_proposals[:population_size]

        await self.emit_progress(
            "agent_done",
            f"Architect proposed {len(all_proposals)} architectures",
            generation, depth,
            {"architectures": [p["architecture_name"] for p in all_proposals]},
        )

        context["architecture_proposals"] = all_proposals
        return context
