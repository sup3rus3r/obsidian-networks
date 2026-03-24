---
name: validator-soundness
description: Soundness checklist and overfitting interpretation for evaluating trained architecture checkpoints against real data. Load when running validation.
agent: validator
version: 1.0
domains: [all]
---

# Validator Skill — Soundness Checklist

## Purpose

The Validator computes a `loss_ratio` (real_loss / synthetic_loss) and a
`generalization_score` for each trained checkpoint. These numbers alone are not
sufficient — the Validator must also interpret whether the pattern of results indicates
genuine overfitting, data distribution shift, or a structural model problem. This skill
provides the interpretation framework.

## Key Principles

- A loss_ratio of 1.0 means the model performs identically on synthetic and real data —
  ideal but rare. Ratios between 0.8 and 1.3 are acceptable.
- A loss_ratio > 1.5 is a strong overfitting signal — the model learned synthetic data
  patterns that do not transfer.
- A loss_ratio < 0.8 (real loss LOWER than synthetic) is suspicious — the real data may
  be easier or the synthetic data generation was poorly calibrated.
- The generalization_score is `max(0, 1 - (loss_ratio - 1) / 1)`. This gives:
  - ratio 1.0 → score 1.0 (perfect)
  - ratio 1.5 → score 0.5
  - ratio 2.0 → score 0.0

## Interpretation Checklist

Work through this checklist for each evaluated candidate:

### 1. Check loss_ratio range

| loss_ratio  | Interpretation | Action |
|-------------|----------------|--------|
| < 0.8       | Real data easier than synthetic; synthetic data too hard | Flag: synthetic mismatch |
| 0.8 – 1.2   | Good generalisation | Pass: mark generalization_score as valid |
| 1.2 – 1.5   | Mild overfitting | Pass with note: overfitting_detected = False, but borderline |
| 1.5 – 2.0   | Clear overfitting | Mark overfitting_detected = True |
| > 2.0       | Severe overfitting | Mark overfitting_detected = True; suggest more regularisation |

### 2. Check if the model actually trained

If `synthetic_metrics.loss == 999.0`, the model did not train successfully.
A loss_ratio for a failed model is meaningless — return generalization_score = 0.0.

### 3. Check for metric type mismatch

The evaluation uses domain-specific metrics. If the domain is `timeseries`,
the primary metric is MSE, not accuracy. Ensure the loss being compared is
the same metric for both synthetic and real evaluation.

### 4. Domain-Specific Thresholds

Different domains have different expected loss_ratio ranges due to dataset
distribution shift:

- **Vision** — Synthetic images are random noise; real images have structure.
  loss_ratio > 1.5 is expected and not necessarily concerning. Threshold: > 2.5 = overfitting.
- **Timeseries** — Synthetic is sine wave; real data has non-stationarity.
  loss_ratio > 2.0 is common. Threshold: > 3.0 = severe overfitting.
- **Tabular** — Synthetic from make_classification; real has feature correlations.
  loss_ratio > 1.5 = overfitting. Threshold: > 2.0 = severe.
- **Language** — Synthetic random tokens; real text has vocabulary patterns.
  loss_ratio > 2.0 is common. Not meaningful for architecture comparison.
- **Graph** — Synthetic random graphs; real has structural properties.
  loss_ratio > 2.0 is common. Focus on relative ranking between candidates.
- **Generative** — Loss metrics (reconstruction MSE) may not be comparable across
  domains. Treat generalization_score for generative models as indicative only.
- **Recommendation** — Real interaction matrices have strong power-law distributions
  unlike synthetic uniform random. loss_ratio > 2.0 is common.

## What to Flag vs What to Pass

**Flag (set overfitting_detected = True):**
- loss_ratio > 1.5 for vision, tabular
- loss_ratio > 2.0 for timeseries, language, graph, recommendation
- loss_ratio > 2.5 for generative

**Pass (overfitting_detected = False):**
- All ratios below the flagging threshold
- Failed models (status != "evaluated") — these are excluded from validation

**Never:**
- Return a generalization_score > 1.0 or < 0.0
- Compare candidates that used different synthetic data seeds (may cause spurious
  ratio differences)
