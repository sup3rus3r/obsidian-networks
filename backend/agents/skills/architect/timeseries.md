---
name: architect-timeseries
description: Mutation strategy for timeseries domain architectures (LSTMs, TCNs, forecasting models). Load when proposing mutations for timeseries tasks.
agent: architect
version: 1.0
domains: [timeseries]
---

# Architect Skill — Timeseries Domain

## Purpose

Timeseries architectures must respect causality — the model may only use past timesteps
to predict future ones. This is the single most important constraint. Beyond causality,
the challenge is learning multi-scale temporal patterns: short-term fluctuations and
long-term trends simultaneously. Mutations that collapse temporal structure (global
pooling too early, ignoring sequence order) or introduce data leakage (using future
values during training) produce invalid models. This skill guides the Architect toward
causally valid, structurally sound mutation proposals.

## Key Principles

- Never allow future timestep information to flow backward. Bidirectional LSTMs and
  non-causal convolutions are forbidden for forecasting tasks.
- Multi-scale temporal modelling is the highest-value mutation direction. Dilated
  causal convolutions, hierarchical LSTMs, and wavelet decomposition are all underexplored.
- The output shape must match the forecast horizon exactly. If predicting 24 steps ahead,
  the output layer must produce exactly 24 values, not 1 repeated 24 times.
- Normalization should be applied per-sequence (instance normalization), not batch-wide,
  to handle non-stationary time series correctly.
- Attention over time (temporal self-attention) should be causal (masked) for forecasting
  or uncausal (full) for classification of sequences.
- Prefer `free_form` and `architecture_crossover` over standard layer-stacking mutations.

## Domain-Specific Guidance

### Approved Base Templates

- **LSTM** — Default for most timeseries tasks. Mutate gating mechanisms, memory cell
  structure, or add attention over hidden states.
- **TCN (Temporal Convolutional Network)** — Use when proposing dilated convolution
  mutations. TCN with exponentially growing dilation has very large receptive fields
  with few parameters.

### Known-Good Mutation Combinations

- Dilated Causal Convolution (free_form) — Stack 1D causal convolutions with dilation
  rates [1, 2, 4, 8, 16]. This gives a receptive field of 32 timesteps with only 5 layers.
  Well-studied but rarely implemented correctly with proper causal padding.
- Temporal Attention + LSTM (free_form) — After the LSTM produces hidden states for
  all timesteps, apply a learned attention weighting over the hidden states before
  aggregation. Tests whether variable-importance over time improves forecasting.
- `architecture_crossover` with `neural_ode` — Model the hidden state evolution as a
  continuous-time ODE rather than discrete recurrence. Rare in production forecasting.
- Multi-resolution decomposition (free_form) — Decompose the input into trend and
  seasonal components (moving average + residual) and model each with a separate branch
  before fusion. Inspired by N-BEATS but applied to general LSTM/TCN frameworks.
- Reversible Instance Normalization (free_form) — Normalize input per-sequence, train,
  then de-normalize output. Addresses non-stationarity without architectural changes
  to the core model.

### Known Dead Ends

- Bidirectional LSTM for forecasting — uses future information, making training correct
  but deployment impossible. Only valid for sequence classification, not forecasting.
- Global average pooling on the time axis before the output — collapses temporal
  information needed for multi-step forecasting.
- Standard `layer_insertion` adding Dense layers without reshaping — Dense layers
  expect (batch, features), not (batch, timesteps, features). Shape mismatch.
- Very large kernels (≥ 64) in 1D convolutions on short sequences (≤ 128 steps) —
  the kernel is larger than the sequence; output is near-constant.
- BatchNormalization on time series — computes statistics across the batch dimension.
  For non-stationary series with different scales, this corrupts the signal. Use
  LayerNormalization or per-instance normalization instead.

### Common Failure Modes

- **Data leakage** — Using `return_sequences=False` on the wrong LSTM layer causes
  only the last hidden state to propagate, which may implicitly use future context
  if the architecture is structured incorrectly.
- **Output shape mismatch** — For multi-step forecasting (horizon=24), the final Dense
  layer must output 24 units, not 1. A single Dense(1) output repeated via tiling
  is not the same as learning multi-step jointly.
- **Non-stationary input killing training** — Without normalization, timeseries with
  large magnitude variation (e.g. energy consumption) cause gradient explosion in the
  first few steps. Always normalize per-sequence before feeding to the model.
- **Dilation without causal padding** — Dilated Conv1D without explicit causal padding
  (left-pad only) looks ahead in time. Padding must be `(dilation_rate * (kernel_size - 1), 0)`.

## Examples

### Good Mutation Proposal
```json
{
  "architecture_name": "tcn_multiscale_trend_seasonal",
  "mutations": ["free_form", "architecture_crossover"],
  "rationale": "Decompose input into trend and residual components, model each with a separate dilated causal TCN branch, then fuse — tests whether explicit decomposition of non-stationarity improves forecasting accuracy versus learning it implicitly.",
  "free_form_description": "TrendResidualDecomposer: apply a learnable moving-average filter (Conv1D with large kernel, causal padding) to extract trend; subtract from input to get residual. Feed trend into a shallow TCN (dilations [1,2]) and residual into a deeper TCN (dilations [1,2,4,8]). Concatenate outputs and project to forecast horizon."
}
```

### Bad Mutation Proposal
```json
{
  "architecture_name": "lstm_bidirectional",
  "mutations": ["layer_insertion"],
  "rationale": "Bidirectional LSTM captures more context."
}
```
This is bad because: bidirectional is invalid for forecasting (data leakage), and
`layer_insertion` of a standard layer produces low novelty scores.
