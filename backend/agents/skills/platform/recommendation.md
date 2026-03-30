---
name: platform-recommendation
description: Guides the platform agent through RESEARCH → PLAN → BUILD for recommendation system tasks (collaborative filtering, neural CF, attention-based recommenders). Load when the user describes a recommendation or rating prediction task.
agent: platform
version: 1.0
domains: [recommendation]
---

# Platform Skill — Recommendation Domain

## Purpose

Recommendation models take user and item IDs as integer inputs and learn dense embedding
representations. The platform's tabular encoding pipeline does NOT apply to recommendation
datasets — user_id and item_id must be passed directly as integer Inputs to Embedding layers.
This skill guides the agent in researching neural collaborative filtering architectures,
planning the user/item embedding pipeline, and generating code that builds separate
Embedding layers for users and items and combines them via dot product or MLP.

## Key Principles

- User and item IDs must be integer inputs to Embedding layers — do NOT encode them as floats.
- The number of unique users (`n_users`) and items (`n_items`) must be stated in the plan and used as `Embedding(n_users+1, embed_dim)`.
- For binary implicit feedback (clicked/not): use sigmoid output + binary_crossentropy.
- For explicit ratings (1–5 scale): normalise to [0, 1] and use MSE loss with sigmoid output.
- Embedding regularisation (L2) prevents overfitting on sparse interaction matrices.
- Do NOT use `Normalization` layer on user/item IDs — they are categorical, not numeric features.

## Procedure

### RESEARCH Phase

**CRITICAL: Never ingest paper URLs from your training knowledge. Only ingest URLs returned by arXiv search results — do not use papers you already know.**

1. Run both arXiv searches using the `search_arxiv` tool — use only the returned results:
   - `"neural collaborative filtering matrix factorization deep learning 2024"`
   - `"attention-based recommendation system user item embedding 2023 2024"`
2. Select 3–4 papers. Focus on: embedding dimension, MLP layers, loss function for implicit/explicit feedback.
3. Ingest in parallel.
4. Fetch TF/Keras docs in parallel:
   - `"Embedding layer user item collaborative filtering TensorFlow"`
   - `"Dot product concatenate user item recommendation Functional API"`
   - `"binary_crossentropy MSE rating prediction recommendation"`
5. Call `finalize_research()`.

### PLANNING Phase

Query at least 6 times before writing:
- `"embedding dimension size collaborative filtering neural"`
- `"dot product vs concatenation MLP user item interaction"`
- `"L2 regularization embedding recommendation overfitting"`
- `"implicit feedback negative sampling recommendation"`
- `"learning rate batch size neural collaborative filtering"`
- `"evaluation metric NDCG hit_rate recommendation"`

The plan MUST include:
- `n_users` and `n_items` — count unique IDs from the dataset (or from user description)
- `embed_dim` (e.g. 32–128) with source justification
- Interaction function: Dot product (GMF), concatenate + MLP (NCF), or both (NeuMF)
- Loss function: binary_crossentropy (implicit) or MSE (explicit ratings)
- Whether negative sampling is needed (implicit feedback only)
- L2 regularisation on embedding weights

### BUILD Phase

Follow the standard BUILD SEQUENCE. Key recommendation-specific steps:

1. After `approve_plan`, run `run_code` to inspect user/item IDs and rating distribution:
   ```python
   import pandas as pd
   df = pd.read_csv('dataset.csv')
   print(df.head(3))
   print('n_users:', df.iloc[:, 0].nunique(), 'n_items:', df.iloc[:, 1].nunique())
   print('Rating range:', df.iloc[:, 2].min(), df.iloc[:, 2].max())
   ```
2. In `edit_script`:
   - Two Integer Inputs: `user_input = Input(shape=(1,), dtype='int32')` and `item_input = Input(shape=(1,), dtype='int32')`
   - `user_embed = Embedding(n_users+1, embed_dim, embeddings_regularizer=L2(1e-5))(user_input)`
   - `item_embed = Embedding(n_items+1, embed_dim, embeddings_regularizer=L2(1e-5))(item_input)`
   - For Dot: `Flatten()(user_embed)` dot `Flatten()(item_embed)` → `Dense(1, sigmoid)`
   - For MLP: `Concatenate()([Flatten()(user_embed), Flatten()(item_embed)])` → `Dense(128, relu)` → `Dense(64, relu)` → `Dense(1, sigmoid)`
   - Normalise explicit ratings to [0, 1]: `ratings = (ratings - 1) / 4.0` for 1–5 scale

## Domain-Specific Guidance

### Approved Architecture Patterns

- **GMF** (Generalised Matrix Factorisation) — element-wise product of user/item embeddings + sigmoid. Fast baseline.
- **NCF-MLP** — concatenate embeddings, pass through 3 Dense layers. Better than dot for complex interactions.
- **NeuMF** — combine GMF and MLP outputs before final sigmoid. Best performance but more parameters.

### Embedding Dimension Guide

| n_users + n_items | Recommended embed_dim |
|---|---|
| < 10000 | 32–64 |
| 10000–100000 | 64–128 |
| > 100000 | 128–256 |

### Known Dead Ends

- Using `Normalization` layer on user/item IDs — IDs are categorical; normalising them destroys the mapping.
- `model.predict([user_ids, item_ids])` before fitting — Embedding weights are random; predictions are meaningless until trained.
- Embedding dimension larger than `sqrt(n_users)` or `sqrt(n_items)` — overfits sparse matrices.
- No regularisation on embeddings — popular users/items dominate gradients; embeddings for rare ones collapse.
- Rating values not normalised to [0, 1] with MSE loss — loss magnitude incompatible with Adam's default lr.

### Common Failure Modes

- **Embedding index out of range** — if a user_id or item_id equals `n_users` or `n_items` (0-indexed), it exceeds the embedding table. Use `Embedding(n_users+1, ...)` to add a safety index.
- **Model receives float IDs** — platform may encode IDs as floats. Cast explicitly: `tf.cast(user_input, tf.int32)` in the model.
- **Loss stuck at 0.693** — binary_crossentropy on 50/50 random predictions. Check that negative samples are being generated and that the model is actually receiving the interaction pairs.
- **Output out of [0, 1] range** — missing `sigmoid` on final Dense layer. Always end with `Dense(1, activation='sigmoid')`.

## Examples

### Good Plan Architecture Section

```
### Selected Model: NeuMF (Neural Matrix Factorisation)
n_users=6040, n_items=3706, embed_dim=64  [He et al. 2017, https://...]
User Embedding(6041, 64, L2=1e-5) + Item Embedding(3707, 64, L2=1e-5)

GMF branch: Flatten(user) ⊙ Flatten(item)  [element-wise product]
MLP branch: Concatenate([Flatten(user), Flatten(item)]) → Dense(256,relu) → Dense(128,relu) → Dense(64,relu)
NeuMF output: Concatenate([gmf_out, mlp_out]) → Dense(1, sigmoid)

Loss: binary_crossentropy  [implicit feedback 0/1]
Optimizer: Adam(lr=1e-3)
```

### Bad Plan Architecture Section

```
### Selected Model: Recommendation Model
Embedding layers for users and items, then some Dense layers.
```
This is bad: no n_users/n_items, no embed_dim, no interaction function, no loss function, no citations.
