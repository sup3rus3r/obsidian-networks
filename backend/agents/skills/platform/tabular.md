---
name: platform-tabular
description: Guides the platform agent through RESEARCH → PLAN → BUILD for tabular classification and regression tasks. Load when dataset_type is "tabular".
agent: platform
version: 1.0
domains: [tabular]
---

# Platform Skill — Tabular Domain

## Purpose

Tabular tasks have dense numeric features after automatic encoding. The agent must
research architectures proven on structured data, build a plan grounded in evidence,
and generate a script using the normalizer-first pattern that the platform enforces.
This skill fills the gap between the generic build constraints and what actually works
well for tabular ML: feature normalization, imbalance handling, and regularisation depth.

## Key Principles

- All features are pre-encoded to float32 by the platform. Do not add any encoding layers.
- `Normalization.adapt()` must run on X_train ONLY — never on the full dataset.
- For binary classification, use AUC as primary metric; use accuracy only as secondary.
- Class imbalance (imbalance_ratio > 3) requires `class_weight` in `model.fit()` — do not leave it out.
- Residual/skip connections with `layers.Add()` require matching shapes — always project the shortcut.
- The plan must state the feature count (n_features), target type, and class balance explicitly.

## Procedure

### RESEARCH Phase

**CRITICAL: Never ingest paper URLs from your training knowledge. Only ingest URLs returned by arXiv search results — do not use papers you already know.**

1. Run both arXiv searches using the `search_arxiv` tool — use only the returned results:
   - `"tabular deep learning wide deep feature crosses 2023 2024"`
   - `"neural networks structured data regularization residual 2024"`
2. Select 3–4 papers focused on: architecture depth, feature interaction, regularisation.
3. Ingest selected papers in parallel.
4. Fetch TF/Keras docs for these topics (in parallel):
   - `"Normalization layer adapt tabular preprocessing"`
   - `"EarlyStopping ModelCheckpoint callbacks"`
   - `"class_weight imbalanced binary classification"`
5. Call `finalize_research()`.

### PLANNING Phase

Query the vector store at least 6 times before writing. Use these query angles:
- `"optimal hidden layer sizes tabular classification"`
- `"dropout regularization tabular deep network"`
- `"batch normalization vs layer normalization tabular"`
- `"learning rate schedule tabular neural network"`
- `"AUC metric binary classification Keras"`
- `"class imbalance handling neural network tabular"`

The plan MUST include:
- Exact number of input features (`n_features`) from the dataset analysis
- Whether class imbalance handling is needed (imbalance_ratio from analysis)
- Hidden layer sizes and activation choices with source URLs
- Normalization strategy (BatchNorm vs LayerNorm vs none)
- Whether to use residual connections (only for depth ≥ 4 layers)
- Primary metric justification (AUC for binary, RMSE for regression, accuracy for multiclass)

### BUILD Phase

Follow the standard BUILD SEQUENCE. Key tabular-specific steps:

1. After `approve_plan`, run `run_code` to inspect the dataset:
   ```python
   import pandas as pd
   df = pd.read_csv('dataset.csv')
   print(df.shape)
   print(df.dtypes)
   print(df.head(2))
   ```
2. In `edit_script`, implement the normalizer-first pattern:
   - `X = df[feature_cols].to_numpy(dtype='float32')`
   - Split into train/val BEFORE calling `normalizer.adapt(X_train)`
   - Build model with normalizer as first layer (Functional API)
3. For binary classification: compile with `metrics=['AUC', 'accuracy']`
4. If imbalance_ratio > 3: compute `class_weight` and pass to `model.fit()`

## Domain-Specific Guidance

### Proven Architecture Patterns

- **Shallow wide** (2–3 layers, 256–512 units) — best for < 50 features, few interactions
- **Deep with residuals** (4–6 layers, 128–256 units, skip every 2) — for complex interactions
- **Wide & Deep** (shared embedding + parallel branches) — when some features are embeddings

### Hyperparameter Defaults (cite any paper that confirms these)

| Parameter | Tabular Default | Notes |
|---|---|---|
| Activation | `relu` or `gelu` | `selu` works if no BatchNorm |
| Dropout | 0.2–0.4 | After each Dense block, not after output |
| Optimizer | Adam, lr=1e-3 | Reduce on plateau if loss noisy |
| Batch size | 256–1024 | Larger batches stable for tabular |
| Normalization | BatchNorm after Dense | Or LayerNorm if batch size < 64 |

### Known Dead Ends

- Embedding layers for any feature — the platform already encodes categoricals.
- `StringLookup` or any text processing — all columns are numeric floats.
- Sequential model — use Functional API for anything with residuals or multiple inputs.
- `normalizer.adapt(X)` on the full X before splitting — this leaks validation statistics.
- Epochs < 50 — EarlyStopping with patience=20 needs enough epochs to search; set epochs=200.

### Common Failure Modes

- **Shape mismatch in residual Add()** — shortcut Dense units must equal block output units. Always add `Dense(units)(shortcut)` before `Add()`.
- **NaN loss on first epoch** — usually caused by not dropping NaN rows. Always run `df.dropna()` after `df.replace([np.inf, -np.inf], np.nan)`.
- **AUC stuck at 0.5** — usually class imbalance not handled. Add class_weight.
- **Normalizer not adapted** — model trains but predictions are nonsense. Verify `normalizer.adapt(X_train)` is called before `model = build_model()`.

## Examples

### Good Plan Architecture Section

```
### Selected Model: Deep Residual Tabular Network
Input: (32,) — 32 numeric features after encoding [analysis]
Dense(256, gelu) → BatchNorm → Dropout(0.3)  [Smith et al. 2023, https://...]
Dense(256, gelu) → BatchNorm → Dropout(0.3)  [Smith et al. 2023, https://...]
Add(shortcut=Dense(256)(input))              [He et al. 2015, https://...]
Dense(128, gelu) → Dropout(0.2)             [...]
Output: Dense(1, sigmoid)                    [binary classification]
```

### Bad Plan Architecture Section

```
### Selected Model: Tabular Classifier
A few Dense layers with ReLU and dropout.
```
This is bad: no layer sizes, no citations, no normalization mention, no residual justification.
