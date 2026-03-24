---
name: architect-tabular
description: Mutation strategy for tabular domain architectures (MLPs, TabTransformers, feature-interaction models). Load when proposing mutations for tabular tasks.
agent: architect
version: 1.0
domains: [tabular]
---

# Architect Skill — Tabular Domain

## Purpose

Tabular data has no spatial or sequential structure — columns are independent features,
not pixels or tokens. The challenge is learning feature interactions that matter while
ignoring those that don't. Standard MLPs treat all feature pairs equally; the high-novelty
direction is architectures that learn which feature interactions are informative. This
skill guides the Architect toward mutations that address tabular data's unique structure
rather than importing vision or language patterns that don't apply.

## Key Principles

- There is no spatial or sequential structure in tabular data. Convolutions and RNNs
  applied directly to feature vectors are category errors — do not propose them.
- Feature interaction learning is the core challenge. Mutations should propose novel
  ways to model pairwise or higher-order feature interactions.
- Mixed data types (categorical + numerical) are common. Categorical features require
  embedding; numerical features require normalisation. Any architecture that doesn't
  handle both is incomplete.
- Class imbalance is frequent in tabular tasks. The architecture should not assume
  balanced classes — output layer and loss must handle imbalance.
- Tabular models should be parameter-efficient. Large models overfit severely on typical
  tabular datasets (< 10k rows). Keep total params under 500k.
- Prefer `free_form` (novel interaction mechanisms) over standard MLP stacking.

## Domain-Specific Guidance

### Approved Base Templates

- **MLP** — Default. Mutate by adding feature interaction layers, gating mechanisms,
  or attention over features.
- **TabTransformer** — Transformer applied to embedded categorical features. Mutate
  the attention mechanism or the way numerical features are incorporated.

### Known-Good Mutation Combinations

- Feature Tokenization + Attention (free_form) — Embed each numerical feature into a
  learned d-dimensional vector (one embedding per feature, not per value). Then apply
  self-attention over the feature dimension. This makes the model learn which features
  attend to which. FT-Transformer-inspired but applicable to any base.
- Gated Feature Selection (free_form) — Add a learned sparse gate before each layer:
  `gate = sigmoid(W * x + b); output = gate * x`. Forces the network to select which
  features are relevant at each layer. Tests sparsity as a regulariser.
- `architecture_crossover` with `neural_ode` — Model feature interactions as a
  continuous-time dynamical system rather than discrete layer transformations.
  The feature vector evolves through an ODE parameterised by the network.
- Multi-order Interaction Network (free_form) — Explicitly compute pairwise products
  of feature embeddings (FM-style), then feed these interaction terms along with
  raw features to a shallow MLP. Tests whether explicit second-order interactions
  help beyond implicit learning.
- Residual Feature Blocks (free_form) — Apply dense residual blocks where the skip
  connection preserves the original feature vector and is added back at each stage.
  Prevents feature information from being overwritten by transformations.

### Known Dead Ends

- Conv1D or Conv2D applied to the feature vector — there is no spatial adjacency
  between tabular columns. A convolution over features 3 and 4 is no more meaningful
  than over features 3 and 17.
- RNNs over features — feature order in tabular data is arbitrary. Sequential models
  impose a meaningless ordering.
- Very deep MLPs (> 8 layers) without residual connections — deep MLPs collapse to
  near-zero gradients for the first layers without skip connections on tabular data.
- Embedding extremely high-cardinality categorical features naively — a categorical
  feature with 1000 unique values embedded into 1000-dim space causes overfitting.
  Embedding dim should be min(50, (n_unique // 2) + 1).

### Common Failure Modes

- **Numerical features not normalised** — Tabular features with different scales
  (age in [0, 100] vs income in [0, 1,000,000]) cause gradient imbalance. Always
  standardise (mean=0, std=1) numerical features before feeding to the model.
- **Categorical features not embedded** — Passing integer-encoded categoricals directly
  to a Dense layer treats them as ordinal numbers. Categories must be embedded via
  Embedding layers.
- **Output dimension mismatch for binary vs multi-class** — Binary classification uses
  Dense(1, sigmoid) + binary_crossentropy. Multi-class uses Dense(n_classes, softmax) +
  sparse_categorical_crossentropy. Confusing these produces training errors.
- **Class imbalance ignored** — For imbalanced datasets (e.g. 95% class 0, 5% class 1),
  the model predicts the majority class constantly. Use class_weight in model.fit() or
  a focal loss variant.

## Examples

### Good Mutation Proposal
```json
{
  "architecture_name": "tabular_feature_tokenizer_sparse_gate",
  "mutations": ["free_form", "attention_variant"],
  "rationale": "Tokenize each feature into a learned embedding, apply sparse gated self-attention over features, then project to classification — tests whether explicit learned feature importance via sparse attention outperforms implicit MLP feature mixing on tabular data.",
  "free_form_description": "FeatureTokenizer: for each of N features, learn a unique embedding vector of size d; multiply by the feature value (for numerical) or look up by index (for categorical). Produces shape (batch, N, d). SparseFeatureAttention: apply multi-head attention over the feature dimension (not batch), with a sparsity-inducing top-k mask retaining only the k most relevant feature interactions per head. Project the attended features to a classification head."
}
```

### Bad Mutation Proposal
```json
{
  "architecture_name": "tabular_conv",
  "mutations": ["layer_insertion"],
  "rationale": "Adding Conv1D layers to capture local feature patterns."
}
```
This is bad because: Conv1D on tabular data has no semantic meaning (feature columns
have no spatial adjacency), and `layer_insertion` alone is low novelty.
