"""
CriticAgent — scores candidates and decides whether to recurse, archive, or discard.

Scoring formula:
  composite = 0.3×novelty + 0.2×efficiency + 0.2×soundness + 0.3×generalization

Thresholds:
  composite > 0.75 → recurse
  composite > 0.50 → archive
  composite ≤ 0.50 → discard
"""
from __future__ import annotations

import asyncio
import json
import logging

from .core import BaseAgent

logger = logging.getLogger(__name__)

RECURSE_THRESHOLD = 0.40   # top candidates seed next generation
ARCHIVE_THRESHOLD = 0.25   # anything scoreable is worth showing


class CriticAgent(BaseAgent):

    async def run(self, context: dict) -> dict:
        generation        = context.get("generation", 0)
        depth             = context.get("depth", 0)
        eval_results      = context.get("evaluation_results", [])
        validation_results = context.get("validation_results", [])
        generated_code    = context.get("generated_code", [])

        await self.emit_progress("agent_start", f"Critic scoring {len(eval_results)} candidates...", generation, depth)
        self.log_step("Scoring candidates", {"n_candidates": len(eval_results)})

        # Build validation lookup
        val_lookup = {v["architecture_name"]: v for v in validation_results}

        # Build code lookup for soundness assessment
        code_lookup = {g["architecture_name"]: g for g in generated_code}

        tasks = [
            self._score_candidate(er, val_lookup, code_lookup, generation, depth)
            for er in eval_results
        ]
        scored = await asyncio.gather(*tasks, return_exceptions=True)

        scored_candidates = [s for s in scored if s and not isinstance(s, Exception)]

        # Sort by composite score descending
        scored_candidates.sort(key=lambda x: x["composite_score"], reverse=True)

        # Determine actions
        to_recurse = [s for s in scored_candidates if s["next_action"] == "recurse"]
        to_archive = [s for s in scored_candidates if s["next_action"] == "archive"]

        await self.emit_progress(
            "agent_done",
            f"Critic: {len(to_recurse)} recurse, {len(to_archive)} archive, {len(scored_candidates)-len(to_recurse)-len(to_archive)} discard",
            generation, depth,
            {
                "top_score"   : round(scored_candidates[0]["composite_score"], 3) if scored_candidates else 0,
                "to_recurse"  : [s["architecture_name"] for s in to_recurse],
                "to_archive"  : [s["architecture_name"] for s in to_archive],
            },
        )

        context["scored_candidates"] = scored_candidates
        context["candidates_to_recurse"] = to_recurse
        if scored_candidates:
            winner_name = scored_candidates[0]["architecture_name"]
            context["previous_winner_arch"] = winner_name
            # Store the base template name (e.g. "lstm") so the next-generation
            # Architect can call get_base_template() without crashing on "lstm_mutant".
            winner_code = code_lookup.get(winner_name, {})
            context["previous_winner_base_arch"] = winner_code.get("base_template") or winner_name
        else:
            context["previous_winner_arch"] = None
            context["previous_winner_base_arch"] = None

        # Update FAISS novelty index with the specs of all scored candidates so
        # future generations can compute real embedding distances instead of defaulting to 0.8.
        await self._update_novelty_index(scored_candidates, code_lookup)

        # Generate novelty feedback for the next generation's Mathematician and Architect.
        # Written into context so subsequent agents can read it and steer away from
        # overexplored territory toward genuinely novel directions.
        novelty_feedback = self._generate_novelty_feedback(
            scored_candidates, code_lookup, generation
        )
        if novelty_feedback:
            context["novelty_feedback"] = novelty_feedback
            self.log_step("Novelty feedback generated", {"length": len(novelty_feedback)})

        return context

    async def _score_candidate(
        self,
        eval_result: dict,
        val_lookup: dict,
        code_lookup: dict,
        generation: int,
        depth: int,
    ) -> dict | None:
        arch_name         = eval_result["architecture_name"]
        synthetic_metrics = eval_result.get("synthetic_metrics", {})
        memory_mb         = eval_result.get("memory_mb", 0)
        inference_ms      = eval_result.get("inference_time_ms", 9999)
        param_count       = eval_result.get("param_count", 0)
        training_time_s   = eval_result.get("training_time_s", 9999)

        # 1. Novelty score (FAISS embedding distance from archived candidates)
        novelty_score = await self._compute_novelty(arch_name, code_lookup.get(arch_name, {}).get("spec", {}))

        # 2. Efficiency score (normalized: lower memory/inference = better)
        efficiency_score = self._compute_efficiency(memory_mb, inference_ms, param_count, training_time_s)

        # 3. Soundness score (LLM judge)
        code = code_lookup.get(arch_name, {}).get("code", "")
        soundness_score = await self._compute_soundness(arch_name, code, synthetic_metrics)

        # 4. Generalization score (from validator or default 0.5)
        val = val_lookup.get(arch_name)
        generalization_score = val["generalization_score"] if val else 0.5

        # Composite
        composite = (
            0.3 * novelty_score +
            0.2 * efficiency_score +
            0.2 * soundness_score +
            0.3 * generalization_score
        )
        composite = round(min(1.0, max(0.0, composite)), 4)

        if composite > RECURSE_THRESHOLD:
            action = "recurse"
        elif composite > ARCHIVE_THRESHOLD:
            action = "archive"
        else:
            action = "discard"

        return {
            "architecture_name"          : arch_name,
            "composite_score"            : composite,
            "novelty_score"              : round(novelty_score, 4),
            "efficiency_score"           : round(efficiency_score, 4),
            "soundness_score"            : round(soundness_score, 4),
            "generalization_score"       : round(generalization_score, 4),
            "next_action"                : action,
            "synthetic_metrics"          : synthetic_metrics,
            "validation"                 : val_lookup.get(arch_name),
            "memory_mb"                  : memory_mb,
            "inference_time_ms"          : inference_ms,
            "param_count"                : param_count,
        }

    async def _update_novelty_index(self, scored_candidates: list[dict], code_lookup: dict) -> None:
        """Add the specs of all scored candidates to the FAISS novelty index.

        This ensures future generations compute real embedding distances rather
        than always defaulting to 0.8 (the 'no index yet' fallback).
        """
        try:
            import faiss
            import numpy as np
            import os
            from pathlib import Path
            from sentence_transformers import SentenceTransformer

            index_path = Path(os.environ.get("RESEARCH_ARTIFACTS_DIR", "/research_artifacts")) / "novelty_index.faiss"
            index_path.parent.mkdir(parents=True, exist_ok=True)

            model = SentenceTransformer("all-MiniLM-L6-v2")

            texts = []
            for c in scored_candidates:
                spec = code_lookup.get(c["architecture_name"], {}).get("spec", {})
                texts.append(json.dumps(spec) if spec else c["architecture_name"])

            if not texts:
                return

            vecs = model.encode(texts, normalize_embeddings=True).astype("float32")
            dim  = vecs.shape[1]

            if index_path.exists():
                index = faiss.read_index(str(index_path))
            else:
                index = faiss.IndexFlatL2(dim)

            index.add(vecs)
            faiss.write_index(index, str(index_path))
            logger.info("Novelty index updated: %d total vectors", index.ntotal)

        except Exception as e:
            logger.warning("Failed to update novelty index: %s", e)

    async def _compute_novelty(self, arch_name: str, spec: dict) -> float:
        """
        Compute novelty by comparing spec embedding against archived candidates via FAISS.
        Falls back to random novelty if FAISS index not initialised yet.
        """
        try:
            import faiss
            import numpy as np
            from sentence_transformers import SentenceTransformer
            import os
            from pathlib import Path

            index_path = Path(os.environ.get("RESEARCH_ARTIFACTS_DIR", "/research_artifacts")) / "novelty_index.faiss"
            if not index_path.exists():
                # No index yet — first generation is always novel
                return 0.8

            model     = SentenceTransformer("all-MiniLM-L6-v2")
            spec_text = json.dumps(spec)
            vec       = model.encode([spec_text], normalize_embeddings=True).astype("float32")

            index = faiss.read_index(str(index_path))
            if index.ntotal == 0:
                return 0.8

            distances, _ = index.search(vec, min(5, index.ntotal))
            # Average L2 distance → novelty (higher distance = more novel)
            avg_dist = float(distances[0].mean())
            # Normalize: distance 0 = duplicate (0.0), distance 2+ = very novel (1.0)
            novelty  = min(1.0, avg_dist / 2.0)
            return novelty

        except Exception:
            import random
            return round(random.uniform(0.5, 0.9), 4)

    def _compute_efficiency(self, memory_mb: float, inference_ms: float, param_count: int, training_time_s: float) -> float:
        """Normalize efficiency metrics to [0,1]. Lower resource use = higher score."""
        # Reference values for normalization
        MAX_MEMORY   = 4096.0   # MB
        MAX_INFERENCE = 1000.0  # ms
        MAX_PARAMS    = 100_000_000
        MAX_TRAIN     = 3600.0  # seconds

        mem_score   = max(0, 1 - memory_mb / MAX_MEMORY)         if memory_mb > 0 else 0.5
        inf_score   = max(0, 1 - inference_ms / MAX_INFERENCE)    if inference_ms > 0 else 0.5
        param_score = max(0, 1 - param_count / MAX_PARAMS)        if param_count > 0 else 0.5
        train_score = max(0, 1 - training_time_s / MAX_TRAIN)     if training_time_s > 0 else 0.5

        return round((mem_score + inf_score + param_score + train_score) / 4.0, 4)

    def _generate_novelty_feedback(
        self,
        scored_candidates: list[dict],
        code_lookup: dict,
        generation: int,
    ) -> str:
        """
        Build a novelty feedback string for the next generation's agents.

        Identifies low-novelty candidates, explains why they scored low, and
        suggests unexplored directions. Written to context["novelty_feedback"]
        for consumption by MathematicianAgent and ArchitectAgent.

        Only generates feedback when at least one candidate has novelty_score < 0.45.
        """
        LOW_NOVELTY_THRESHOLD = 0.45

        low_novelty = [
            c for c in scored_candidates
            if c.get("novelty_score", 1.0) < LOW_NOVELTY_THRESHOLD
        ]
        if not low_novelty:
            return ""

        # Count mutation frequency across ALL candidates to identify overexplored ops
        mutation_counts: dict[str, int] = {}
        for c in scored_candidates:
            mutations = code_lookup.get(c["architecture_name"], {}).get("mutations", [])
            for m in mutations:
                mutation_counts[m] = mutation_counts.get(m, 0) + 1

        # The mutations that dominated this generation
        overexplored = [
            m for m, count in sorted(mutation_counts.items(), key=lambda x: -x[1])
            if count >= max(1, len(scored_candidates) // 2)  # used in ≥ half the candidates
        ]

        lines = [f"NOVELTY FEEDBACK (generation {generation}):"]

        for c in low_novelty[:3]:  # cap at 3 to avoid bloating the prompt
            name    = c["architecture_name"]
            score   = c["novelty_score"]
            code_entry = code_lookup.get(name, {})
            mutations  = code_entry.get("mutations", [])
            base       = code_entry.get("base_template", "unknown")

            reason = "mutations are too similar to previously archived candidates"
            if all(m in ("layer_insertion", "width_change", "depth_change") for m in mutations):
                reason = "only standard structural operators were used (layer_insertion, width_change, depth_change) — these produce near-zero novelty"
            elif not mutations:
                reason = "no mutations recorded — the architecture may be a minimal variant of the base template"

            lines.append(
                f"- Candidate '{name}' (base={base}): novelty={score:.2f}. "
                f"Reason: {reason}. Mutations used: {mutations}."
            )

        if overexplored:
            lines.append(
                f"- Overexplored mutation operators this generation: {overexplored}. "
                f"Do NOT use these as the primary operator next generation."
            )

        lines.append("- Suggested directions for next generation:")
        lines.append("  1. Use 'free_form' to propose a completely custom layer not expressible by standard operators.")
        lines.append("  2. Use 'architecture_crossover' with one of: state_space_model, neural_ode, hyperbolic_geometry, fourier_neural_operator, capsule_network, reservoir_computing.")
        lines.append("  3. Derive mechanisms that cross paper boundaries — combine a mathematical principle from one paper with the structure of another that the original authors did not consider.")
        lines.append("  4. Target the Mathematician to look for Tier 1 mechanisms (see Mathematician skill) — reject any Tier 3 (renamed standard operations).")

        return "\n".join(lines)

    async def _compute_soundness(self, arch_name: str, code: str, metrics: dict) -> float:
        """LLM judge rates the architecture's theoretical soundness 0–1."""
        if not code:
            return 0.5

        loss = metrics.get("loss", 999)
        acc  = metrics.get("accuracy")
        acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else "N/A"

        # Load scoring skill as system prompt for the soundness LLM judge
        scoring_skill = self.load_skill(filename="scoring.md")

        try:
            prompt = (
                f"Rate the soundness of this neural architecture on a scale from 0.0 to 1.0.\n\n"
                f"Architecture: {arch_name}\n"
                f"Final training loss: {float(loss):.4f}\n"
                f"Final accuracy: {acc_str}\n\n"
                f"Code (first 800 chars):\n{code[:800]}\n\n"
                f"Use the five-criterion rubric in your scoring skill to evaluate each criterion "
                f"(theoretical coherence, convergence evidence, custom layer quality, "
                f"shape consistency, domain appropriateness). Sum the criterion scores.\n\n"
                f"Reply with ONLY a single float between 0.0 and 1.0. Example: 0.72"
            )
            raw   = await self.call_llm(prompt, force_claude=True, max_tokens=10, system=scoring_skill)
            score = float(raw.strip().split()[0])
            return min(1.0, max(0.0, score))
        except Exception:
            # Use loss-based heuristic as fallback
            if loss < 1.0:
                return 0.8
            elif loss < 5.0:
                return 0.5
            else:
                return 0.2
