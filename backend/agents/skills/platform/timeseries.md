---
name: platform-timeseries
description: Guides the platform agent through RESEARCH → PLAN → BUILD for time series forecasting and classification tasks. Load when dataset_type is "time_series".
agent: platform
version: 1.0
domains: [timeseries]
---

# Platform Skill — Time Series Domain

## Purpose

Time series tasks require windowed sequence inputs, per-feature normalization that does
not leak future statistics, and architectures that capture temporal dependencies. This
skill guides the agent in researching effective temporal models, planning the window
construction and normalization pipeline, and generating code that uses Keras's native
`timeseries_dataset_from_array()` — the ONLY approved windowing method on the platform.

## Key Principles

- NEVER manually roll windows with `for` loops — ALWAYS use `keras.utils.timeseries_dataset_from_array()`.
- Normalize each feature independently using its TRAINING SPLIT statistics only — future values must not inform normalization.
- The `lookback` (sequence length) and `horizon` (forecast steps) must be explicitly specified in the plan.
- For LSTM stacks: the last LSTM must have `return_sequences=False` unless followed by another LSTM or attention layer.
- For Transformer time series: add positional encoding — time steps have order that attention ignores otherwise.
- Evaluation metric is MSE/MAE for regression forecasting; accuracy for classification.

## Procedure

### RESEARCH Phase

**CRITICAL: Never ingest paper URLs from your training knowledge. Only ingest URLs returned by arXiv search results — do not use papers you already know.**

1. Run both arXiv searches using the `search_arxiv` tool — use only the returned results:
   - `"time series forecasting LSTM transformer temporal convolution 2024"`
   - `"multivariate time series prediction deep learning normalization 2023 2024"`
2. Select 3–4 papers. Prefer papers with explicit hyperparameters: lookback windows, layer sizes.
3. Ingest in parallel.
4. Fetch TF/Keras docs in parallel:
   - `"timeseries_dataset_from_array sequence windowing lookback"`
   - `"LSTM return_sequences stacked time series"`
   - `"Normalization adapt per-feature time series"`
5. Call `finalize_research()`.

### PLANNING Phase

Query at least 6 times before writing:
- `"optimal lookback window size time series forecasting"`
- `"LSTM vs Transformer univariate multivariate forecasting"`
- `"temporal convolution TCN time series depth dilation"`
- `"batch normalization time series LSTM normalization"`
- `"dropout regularization LSTM recurrent time series"`
- `"learning rate schedule time series neural network"`

The plan MUST include:
- Number of input features and whether the task is univariate or multivariate
- `lookback` value (e.g. 30–168 steps) with source justification
- `horizon` value for forecasting (e.g. 1–24 steps) or classification output classes
- Architecture type: stacked LSTM, Bidirectional LSTM, TCN, or Transformer
- Normalization: per-feature StandardScaler or Normalization layer (train split only)
- Loss function: MSE or MAE for forecasting; crossentropy for classification

### BUILD Phase

Follow the standard BUILD SEQUENCE. Key timeseries-specific steps:

1. After `approve_plan`, run `run_code` to inspect the datetime column and feature count:
   ```python
   import pandas as pd
   df = pd.read_csv('dataset.csv')
   print(df.shape, df.dtypes)
   print(df.head(3))
   ```
2. In `edit_script`:
   - Sort by datetime column, then drop the datetime column before windowing
   - Split features/target before normalizing — fit scaler on train split only
   - Use `keras.utils.timeseries_dataset_from_array(data, targets, sequence_length=lookback, batch_size=32)`
   - Build model with `Input(shape=(lookback, n_features))` → LSTM/TCN/Transformer → `Dense(horizon)` or `Dense(n_classes, softmax)`
3. For stacked LSTMs: all but the last must have `return_sequences=True`.
4. For Transformer: use `MultiHeadAttention` + `LayerNormalization` + FFN block; add positional Embedding.

## Domain-Specific Guidance

### Approved Architecture Patterns

- **Stacked LSTM** (2–3 layers, 64–128 units) — robust for univariate forecasting, shorter sequences (< 100 steps)
- **Bidirectional LSTM** (1–2 layers) — for classification where full context matters
- **TCN** (dilated Conv1D, depth 4–6, kernel 3) — for long sequences; fast to train on CPU
- **Transformer encoder** (2 blocks, 4 heads, embed_dim=64) — for multivariate with complex cross-feature patterns

### Lookback / Horizon Defaults

| Frequency | Recommended Lookback | Notes |
|---|---|---|
| Minute/Hourly | 48–168 steps | Cover 2–7 days of history |
| Daily | 30–90 days | One to three months |
| Weekly | 26–52 weeks | One year |
| Horizon | 1–horizon steps | Match the user's stated forecast length |

### Known Dead Ends

- Manual windowing with `for` loops — always replaced by `timeseries_dataset_from_array`.
- Normalizing on the full time series before splitting — leaks future statistics into training.
- LSTM `return_sequences=True` feeding directly into `Dense` — need `GlobalAveragePooling1D()` or set the last LSTM to `return_sequences=False`.
- Very large lookback (> 500) with small datasets (< 5000 rows) — the model sees fewer distinct windows; overfits.
- Conv1D without dilation for long sequences — receptive field too small; use `dilation_rate=2**i` for TCN blocks.

### Common Failure Modes

- **`timeseries_dataset_from_array` shape error** — `data` must be 2D `(n_steps, n_features)` even for univariate (reshape to `(-1, 1)` if needed).
- **LSTM input shape mismatch** — `Input(shape=(lookback, n_features))` must match the actual windowed tensor shape.
- **NaN loss in LSTM on financial data** — large variance spikes. Always clip data or use `tf.clip_by_value` before training.
- **Transformer positional encoding missing** — model output does not improve over naive baseline. Add sinusoidal or learned positional embedding after input projection.

## Examples

### Good Plan Architecture Section

```
### Selected Model: Stacked LSTM Forecaster
lookback=48 steps, horizon=12 steps  [Li et al. 2023, https://...]
Input: (48, 7) — 7 features (6 sensors + 1 target)
Normalization: StandardScaler fit on X_train only  [best practice, avoid leakage]
LSTM(128, return_sequences=True) → Dropout(0.2)   [Hochreiter 1997, https://...]
LSTM(64, return_sequences=False) → Dropout(0.2)
Dense(32, relu) → Dense(12, linear)               [12-step ahead forecast]
Loss: MSE. Optimizer: Adam(lr=1e-3). EarlyStopping(patience=20).
```

### Bad Plan Architecture Section

```
### Selected Model: LSTM
An LSTM on the time series data with MSE loss.
```
This is bad: no lookback, no horizon, no feature count, no normalization strategy, no citations.
