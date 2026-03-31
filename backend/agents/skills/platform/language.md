---
name: platform-language
description: Guides the platform agent through RESEARCH → PLAN → BUILD for NLP tasks (text classification, embeddings, transformers). Load when dataset_type is "nlp" or the user describes a text/NLP task.
agent: platform
version: 1.0
domains: [language]
---

# Platform Skill — Language Domain

## Purpose

Language tasks on the platform use TensorFlow/Keras text processing: `TextVectorization`
for vocabulary building and `Embedding` for learned representations. This skill guides the
agent in selecting the right research queries, planning the full text preprocessing pipeline,
and generating code that correctly adapts `TextVectorization` on training data only and
feeds it into classification or sequence-to-sequence architectures.

## Key Principles

- `TextVectorization.adapt()` must run on training text ONLY — never on the full corpus.
- The vocabulary size and sequence length must be specified in the plan with source justification.
- For classification: LSTM or Transformer encoder followed by GlobalAveragePooling1D + Dense.
- For generation: decoder-only Transformer with causal masking — use `MultiHeadAttention(use_causal_mask=True)`.
- Do NOT use `hub.KerasLayer` or any TF-Hub pretrained model — the platform environment may not have internet access.
- In description mode (no dataset uploaded), ALWAYS call `fetch_tensorflow_datasets` first to find and load a real public text dataset (e.g. imdb_reviews, ag_news_subset, glue/sst2). If a dataset is loaded successfully, treat the session as DATASET MODE and use `dataset.csv`. Only fall back to synthetic/random text data if `fetch_tensorflow_datasets` returns `available=false`.
- Positional encoding for Transformers must be added explicitly — Keras `Embedding` does not add it.

## Procedure

### RESEARCH Phase

**CRITICAL: Never ingest paper URLs from your training knowledge. Only ingest URLs returned by arXiv search results — do not use papers you already know.**

1. Run both arXiv searches using the `search_arxiv` tool — use only the returned results:
   - `"text classification LSTM transformer embedding neural network 2024"`
   - `"NLP text encoding sequence model tabular classification 2023 2024"`
2. Select 3–4 papers covering: embedding size, sequence model depth, regularisation.
3. Ingest in parallel.
4. Fetch TF/Keras docs in parallel:
   - `"TextVectorization adapt vocabulary sequence_length"`
   - `"Embedding layer GlobalAveragePooling1D text classification"`
   - `"MultiHeadAttention transformer text encoder classification"`
5. Call `finalize_research()`.

### PLANNING Phase

Query at least 6 times before writing:
- `"embedding dimension text classification sequence model"`
- `"LSTM bidirectional text classification accuracy"`
- `"transformer encoder text sequence classification depth"`
- `"dropout regularization NLP embedding layers"`
- `"vocabulary size TextVectorization optimal"`
- `"learning rate schedule NLP transformer fine-tuning"`

The plan MUST include:
- `max_tokens` (vocabulary size, e.g. 10000–30000) with source
- `sequence_length` (max input length in tokens, e.g. 128–512) with source
- `embed_dim` (embedding dimension, e.g. 64–256) with source
- Whether to use Bidirectional LSTM, Transformer encoder, or simple GlobalAveragePooling
- Explicit positional encoding strategy if using Transformer
- Output layer: `Dense(1, sigmoid)` for binary, `Dense(n, softmax)` for multiclass

### BUILD Phase

Before writing any code, call `query_research` for each key implementation detail (layer sizes, learning rates, loss functions, hyperparameters) to retrieve exact values from the ingested papers. Every value in the script must match the approved plan — `create_notebook` runs an automated alignment check and will reject mismatches.

Follow the standard BUILD SEQUENCE. Key language-specific steps:

1. After `approve_plan`, run `run_code` to inspect the text column and label distribution:
   ```python
   import pandas as pd
   df = pd.read_csv('dataset.csv')
   text_col = df.select_dtypes(include='object').columns[0]
   print(df[text_col].head(3))
   print(df.iloc[:, -1].value_counts())
   ```
2. In `edit_script`:
   - Identify the text column (largest average string length object column) and label column (last column)
   - Build `TextVectorization(max_tokens=N, output_sequence_length=L)` and call `.adapt(train_texts)`
   - Use Functional API: `Input(shape=(1,), dtype=tf.string)` → `TextVectorization` → `Embedding(max_tokens+2, embed_dim)` → encoder → `GlobalAveragePooling1D()` → `Dense`
3. For Transformer encoder: add sinusoidal or learned positional Embedding before the first attention block.

## Domain-Specific Guidance

### Approved Architecture Patterns

- **Embedding + GlobalAveragePooling** (bag-of-words baseline) — for short texts (< 50 tokens), binary/multiclass classification
- **Bidirectional LSTM** (128–256 units, 1–2 stacked) — for sequences with long-range dependencies, sentiment
- **Transformer encoder** (2–4 blocks, 4–8 heads, embed_dim=128) — for complex classification, multi-sentence reasoning

### Vocabulary and Sequence Defaults (cite papers that confirm)

| Parameter | Default | Notes |
|---|---|---|
| `max_tokens` | 10000–20000 | Larger for diverse vocabulary; smaller for domain-specific text |
| `sequence_length` | 128–256 | Truncate/pad to this length |
| `embed_dim` | 64–128 | Larger for Transformer; smaller for LSTM |
| Dropout | 0.3–0.5 | After Embedding and after each encoder block |
| Batch size | 32–128 | Smaller for longer sequences |

### Known Dead Ends

- `hub.KerasLayer` BERT — requires TF-Hub access; use `Embedding` + encoder from scratch instead.
- `adapt()` on full dataset before splitting — leaks test vocabulary statistics.
- LSTM with `return_sequences=True` feeding directly into `Dense` — need GlobalAveragePooling1D or Flatten first.
- Transformer without positional encoding — attention is permutation-invariant without it; model cannot learn order.
- Very large `max_tokens` (> 50000) with small embed_dim — embedding matrix is sparse and does not converge.

### Common Failure Modes

- **Output shape error after LSTM** — `return_sequences=True` gives `(batch, seq_len, units)`. Apply `GlobalAveragePooling1D()` before the Dense output layer.
- **NaN loss from padding** — masking must be explicit. Add `mask_zero=True` to the Embedding layer so padded tokens are ignored.
- **TextVectorization shape error** — the Input must have `dtype=tf.string` and shape `(1,)` when using raw strings, or `(None,)` when using pre-tokenised integer sequences.
- **Transformer training instability** — Add LayerNorm before (pre-norm) rather than after attention and FFN blocks. Use a small learning rate (1e-4) with warmup.

## Examples

### Good Plan Architecture Section

```
### Selected Model: Transformer Encoder Text Classifier
Input: (1,) string → TextVectorization(max_tokens=15000, seq_len=128)
Embedding(15002, 128, mask_zero=True) + PositionalEmbedding(128, 128)  [Vaswani 2017, https://...]
TransformerBlock × 2: MultiHeadAttention(4 heads, key_dim=32) + LayerNorm + FFN(256) + Dropout(0.3)
GlobalAveragePooling1D → Dense(64, relu) → Dropout(0.3) → Dense(1, sigmoid)
Optimizer: Adam(lr=1e-4), loss=binary_crossentropy, metric=AUC  [Zhang 2023, https://...]
```

### Bad Plan Architecture Section

```
### Selected Model: LSTM Text Classifier
LSTM layers on the text data.
```
This is bad: no vocabulary size, no sequence length, no embedding dim, no output shape, no citations.
