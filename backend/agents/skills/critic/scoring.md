---
name: critic-scoring
description: Detailed scoring rubric for all four axes (novelty, efficiency, soundness, generalisation) used by the Critic to evaluate candidates. Load when scoring any candidate.
agent: critic
version: 1.0
domains: [all]
---

# Critic Skill — Scoring Rubric

## Purpose

The Critic scores candidates on four axes: novelty (0.3 weight), efficiency (0.2),
soundness (0.2), generalisation (0.3). Soundness is the only LLM-judged axis — the
others are computed from metrics. This skill provides the detailed rubric that makes
the soundness score meaningful and consistent across generations.

The composite formula is:
```
composite = 0.3 × novelty + 0.2 × efficiency + 0.2 × soundness + 0.3 × generalisation
```

Thresholds: composite > 0.40 → recurse; > 0.25 → archive; ≤ 0.25 → discard.

## Soundness Scoring Rubric (LLM-judged, 0.0–1.0)

When rating soundness, evaluate the architecture on these five criteria.
Each criterion is worth up to 0.2 points of the soundness score.

### Criterion 1: Theoretical Coherence (0–0.2)

Does the architecture make conceptual sense? Is each component connected to a sound
machine learning principle?

| Score | Description |
|-------|-------------|
| 0.2   | Every component has a clear theoretical motivation. Novel components have an explicit hypothesis being tested. |
| 0.15  | Most components are well-motivated. One component is questionable but not harmful. |
| 0.1   | Architecture works but some components appear arbitrary (e.g. random layer count changes without motivation). |
| 0.05  | Multiple components contradict each other or the domain (e.g. bidirectional LSTM for forecasting). |
| 0.0   | Fundamental architectural error (e.g. causal mask on encoder, Conv2D on 1D tabular data). |

### Criterion 2: Training Convergence Evidence (0–0.2)

Does the training loss indicate the model actually learned something?

| Score | Condition |
|-------|-----------|
| 0.2   | Loss decreased substantially from epoch 1 to final epoch AND is below random-chance baseline. |
| 0.15  | Loss decreased but final value is higher than expected for the domain. |
| 0.1   | Loss decreased slightly but model is underfit. |
| 0.05  | Loss barely changed — model likely did not learn. |
| 0.0   | Loss = 999.0 (training crashed) or loss is at random-chance level. |

### Criterion 3: Custom Layer Implementation Quality (0–0.2)

If the architecture includes custom tf.keras.layers.Layer subclasses (required for
high-novelty architectures), are they implemented correctly?

| Score | Condition |
|-------|-----------|
| 0.2   | Custom layers present, inherit from tf.keras.layers.Layer, implement call() with the described mathematical operation. |
| 0.15  | Custom layers present but implementation is a thin wrapper around standard layers rather than implementing the stated mechanism. |
| 0.1   | Standard Keras layers only — no novel implementation (low novelty). |
| 0.05  | Custom layer attempts novel operation but has implementation errors. |
| 0.0   | No custom layers despite mechanism-based mutation, OR custom layers that would fail at runtime. |

### Criterion 4: Dimension and Shape Consistency (0–0.2)

Are tensor shapes consistent throughout the model?

| Score | Condition |
|-------|-----------|
| 0.2   | No shape mismatches identifiable from code inspection. |
| 0.15  | Shape handling correct but could break with different input sizes. |
| 0.1   | One potential shape issue (e.g. hardcoded batch size, missing reshape). |
| 0.05  | Likely shape mismatch that training somehow survived (e.g. broadcasting masked an error). |
| 0.0   | Obvious shape error that should have caused a crash — metrics may be from a fallback. |

### Criterion 5: Domain Appropriateness (0–0.2)

Is the architecture appropriate for its stated domain?

| Score | Condition |
|-------|-----------|
| 0.2   | Architecture fully respects domain constraints (causal for timeseries, symmetric adjacency for graph, etc.). |
| 0.15  | Mostly appropriate with one minor domain mismatch. |
| 0.1   | Architecture is generic (could be any domain) — misses domain-specific structure. |
| 0.05  | Architecture has a domain constraint violation that would cause invalid results in production. |
| 0.0   | Fundamental domain mismatch (e.g. Conv2D on tabular, bidirectional on forecasting). |

## Novelty Score Interpretation

The novelty score is computed from FAISS embedding distance against previous candidates.
When providing feedback on low novelty:

- **Score < 0.3** — Architecture is very similar to previous candidates. Mutations are
  incremental (width/depth changes). Next generation must use `free_form` or
  `architecture_crossover` operators.
- **Score 0.3–0.6** — Moderate novelty. Some new elements but core structure is familiar.
  Encourage combining with mechanisms from a different paper angle.
- **Score > 0.6** — Genuinely novel. Core structural idea is distinct from the archive.

## Efficiency Score Interpretation

- **Score < 0.3** — Model is too large or too slow. param_count > 10M or inference > 500ms.
  Next generation should reduce model size or use parameter-efficient alternatives.
- **Score 0.3–0.6** — Acceptable. Room for improvement.
- **Score > 0.7** — Efficient. Good balance of capability and resource cost.

## Novelty Feedback for Next Generation

When composite score is below archive threshold OR novelty score is below 0.4,
generate a `novelty_feedback` string in this format:

```
NOVELTY FEEDBACK (generation N):
- Candidate <name> scored <score> on novelty. Reason: <why it's low>.
- The mutations used (<list>) are too similar to previously explored candidates.
- Suggested directions for next generation:
  1. <specific free_form or crossover suggestion>
  2. <alternative mathematical mechanism to try>
  3. <domain-specific unexplored direction>
```

This feedback must be placed in `context["novelty_feedback"]` and is consumed by the
Architect and Mathematician in the next generation to guide their proposals.
