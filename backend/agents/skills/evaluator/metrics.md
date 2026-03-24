---
name: evaluator-metrics
description: Domain-specific metric thresholds and interpretation for evaluating trained checkpoints. Load when evaluating results for any domain.
agent: evaluator
version: 1.0
domains: [all]
---

# Evaluator Skill — Metric Interpretation

## Purpose

The Evaluator runs domain-specific metrics and returns raw numbers. Raw numbers are only
meaningful in context. A loss of 0.5 is excellent for vision classification but poor for
timeseries MSE. This skill provides per-domain thresholds and interpretation so the
Evaluator can correctly flag failed, weak, and strong results before passing them to
the Critic.

## Key Principles

- A model that fails to train (loss stays at initialisation value, accuracy at chance)
  must be flagged with `status = "training_failed"`, not `status = "evaluated"`.
- Metrics that indicate random-chance performance (accuracy ≈ 1/n_classes, MSE ≈ variance
  of target) indicate the model did not learn anything. Flag these.
- Domain primary metrics differ: use the correct one for quality assessment.
- Memory and inference time should be compared relatively (within a generation), not
  against absolute thresholds that would disqualify all small models.

## Per-Domain Metric Thresholds

### Vision (primary: accuracy, secondary: loss)

| accuracy | Interpretation |
|----------|----------------|
| < 0.12   | Random chance for 10-class (1/10). Model did not learn. |
| 0.12–0.30 | Weak but learning. Architecturally valid but underfit. |
| 0.30–0.60 | Moderate. Typical for novel architectures on 5-epoch synthetic runs. |
| > 0.60   | Strong. Candidate worth recursing on. |

Loss: `< 1.0` = good; `> 3.0` = likely not converged; `= 2.302...` = random (log(10)).

### Language (primary: accuracy for classification, secondary: loss)

| accuracy | Interpretation |
|----------|----------------|
| < 0.20   | Random for 5-class. Did not learn. |
| 0.20–0.40 | Weak but valid. |
| 0.40–0.65 | Moderate. Expected for transformer/LSTM on synthetic text. |
| > 0.65   | Strong. |

### Timeseries (primary: mse, secondary: mae)

| mse | Interpretation |
|-----|----------------|
| > 1.0 | Worse than predicting mean. Model did not learn. |
| 0.3–1.0 | Weak. Capturing some signal. |
| 0.05–0.3 | Moderate. |
| < 0.05 | Strong on synthetic sine data. |

Note: synthetic timeseries is sine + noise (variance ≈ 0.5). MSE > 0.5 = model predicts
constant and ignores temporal structure.

### Graph (primary: accuracy for node classification, secondary: loss)

| accuracy | Interpretation |
|----------|----------------|
| < 0.20   | Random for 5-class on random graph. Expected baseline. |
| 0.20–0.40 | Weak but valid. Graph structure helping. |
| 0.40–0.65 | Moderate. |
| > 0.65   | Strong on synthetic random graph. |

Note: random graphs have no real community structure. High accuracy on random graphs
means the model learned node features, not graph structure — which is fine for evaluation.

### Audio (primary: accuracy, secondary: loss)

| accuracy | Interpretation |
|----------|----------------|
| < 0.20   | Random for 5-class. Did not learn. |
| 0.20–0.45 | Weak. |
| 0.45–0.70 | Moderate. |
| > 0.70   | Strong on synthetic spectrogram data. |

### Tabular (primary: accuracy for classification, mse for regression)

| accuracy | Interpretation |
|----------|----------------|
| < 0.33   | Random for 3-class. Did not learn. |
| 0.33–0.55 | Weak. |
| 0.55–0.75 | Moderate. Expected for tabular deep learning on make_classification. |
| > 0.75   | Strong. Note: sklearn baseline is often 0.8+ on make_classification. |

### Generative (primary: reconstruction loss/mse)

| loss | Interpretation |
|------|----------------|
| > 10.0 | Reconstruction failed. Model did not learn. |
| 1.0–10.0 | Weak but converging. |
| 0.1–1.0 | Moderate. Expected for VAE on random 64-dim vectors. |
| < 0.1 | Strong reconstruction. |

Note: GAN training is measured differently — g_loss and d_loss should converge to
a rough balance (d_loss ≈ 0.7, g_loss ≈ 1.0 at equilibrium).

### Recommendation (primary: loss via BPR, secondary: ranking metrics if available)

BPR loss should decrease below 0.5 for the model to have learned meaningful rankings.
A BPR loss > 0.6 at convergence indicates the model is not differentiating positive
from negative items.

### Multimodal (primary: accuracy, secondary: modality-specific metrics)

Same thresholds as vision for classification accuracy. Additionally:
- If alignment loss is included, it should decrease during training (not stay flat).
- If one modality is masked and accuracy drops < 5%, modality collapse is occurring.

## Flagging Failed Models

Flag `status = "training_failed"` if:
- Primary metric is at random-chance level (see thresholds above)
- Loss did not decrease from epoch 1 to final epoch (model never converged)
- `synthetic_metrics["loss"] >= 999.0` (training crashed)

Do NOT discard failed models — pass them forward with `status = "training_failed"`
so the Critic can give them a low soundness score and include them in the failure
pattern log.

## Efficiency Metrics

- `memory_mb < 500` — Efficient for small research runs
- `memory_mb > 2000` — Large; will constrain GPU runs
- `inference_time_ms < 50` — Fast; suitable for production
- `inference_time_ms > 500` — Slow; efficiency score will be low
- `param_count < 1M` — Small model; generalises well on limited synthetic data
- `param_count > 10M` — Large; likely to overfit on synthetic data with < 2000 samples
