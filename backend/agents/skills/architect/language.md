---
name: architect-language
description: Mutation strategy for language domain architectures (Transformers, LSTMs, text models). Load when proposing mutations for language tasks.
agent: architect
version: 1.0
domains: [language]
---

# Architect Skill — Language Domain

## Purpose

Language architectures operate on discrete token sequences. The key invariants are:
sequence length consistency, embedding dimension alignment across all layers, and
causality (for generative tasks, no future token access). Mutations that break these
invariants produce untrainable models. This skill guides the Architect toward genuinely
novel language architecture mutations that remain structurally sound.

## Key Principles

- Embedding dimensions must be consistent throughout the model. If you change the
  projection dimension in one block, adjust all subsequent blocks.
- For classification tasks, the base is an encoder-only Transformer or LSTM.
  Do not add a decoder unless the task is explicitly generative.
- Positional encoding is not optional — token order matters for language. Any mutation
  that removes or replaces standard sinusoidal/learned position embeddings must
  substitute an alternative (rotary, ALiBi, relative).
- Attention head count and key_dim must satisfy: `num_heads × key_dim = embed_dim`.
  Violating this causes shape errors.
- LSTM mutations should prefer gating mechanism changes and memory cell modifications
  over simply stacking more LSTM layers.
- Prefer `free_form` and `architecture_crossover` over `layer_insertion` or `depth_change`.

## Domain-Specific Guidance

### Approved Base Templates

- **Transformer encoder** — Default for classification and understanding tasks. Mutate
  the attention mechanism, positional encoding, or feed-forward block structure.
- **LSTM** — Better for sequential tasks with clear temporal structure and small data.
  Mutate the gating mechanism or add attention over hidden states.

### Known-Good Mutation Combinations

- Rotary Position Embeddings (free_form) — Replace sinusoidal position encoding with
  rotary embeddings applied inside the attention query/key computation. RoPE allows
  relative position generalisation and is genuinely superior on short sequences.
- `attention_variant` with sparse patterns — Replace full self-attention with local
  windowed attention (each token attends to its k nearest neighbours). Reduces O(n²)
  cost and forces the model to learn local composition rules first.
- Gated Feed-Forward Layer (free_form) — Replace the standard FFN (Linear → ReLU → Linear)
  with a gated variant: `gate = sigmoid(W_gate x); output = gate * (W_v x)`. GLU
  and SwiGLU variants are under-explored on small synthetic classification tasks.
- `architecture_crossover` with `state_space_model` — Mamba-style selective SSM as an
  alternative to attention. Still rare in classification contexts.
- Cross-layer parameter sharing (free_form) — Share weights between alternating transformer
  layers. Reduces parameters significantly; tests whether weight tying acts as a
  regulariser for synthetic text classification.

### Known Dead Ends

- Cross-attention in an encoder-only model — no second sequence exists. This is a
  structural error, not a mutation.
- Adding more LSTM layers beyond 3 — stacked LSTMs deeper than 3 layers diverge on
  small synthetic sequences. Use bidirectional + 1 attention layer instead.
- Removing layer normalisation from Transformer blocks — training diverges within 2 epochs
  without LayerNorm on synthetic data.
- Very long sequences (> 512 tokens) with full attention — O(n²) memory makes this
  infeasible on CPU/small GPU. Synthetic text tasks should use sequence length ≤ 128.
- `depth_change` alone adding more identical transformer layers — low novelty score and
  adds minimal architectural interest.

### Common Failure Modes

- **Embedding dimension mismatch** — Changing projection dim in attention but not updating
  the feed-forward input dimension. All components in a transformer block share embed_dim.
- **Causal mask applied to encoder** — Encoder tasks (classification) must use bidirectional
  attention (no causal mask). Applying causal masking to a classifier halves the effective
  context at every position.
- **Position encoding shape mismatch** — Position embeddings must match (batch, seq_len, embed_dim).
  If seq_len is dynamic (padded batches), position embeddings must be sliced or use relative encoding.
- **LSTM hidden state dimension inconsistency** — Stacked LSTMs must have matching hidden
  state dimensions or an explicit projection layer between them.
- **Gradient explosion in deep Transformers without gradient clipping** — Add
  `clipnorm=1.0` to the Adam optimizer for transformer models with > 4 layers.

## Examples

### Good Mutation Proposal
```json
{
  "architecture_name": "transformer_gated_ffn_rotary_pe",
  "mutations": ["free_form", "attention_variant"],
  "rationale": "Replace standard sinusoidal PE with rotary embeddings and FFN with SwiGLU gating — tests whether relative position generalisation combined with gated non-linearity produces more expressive language representations on small classification tasks.",
  "free_form_description": "RotaryEmbedding: computes rotation matrices from position indices and applies them to queries and keys before dot-product attention (RoPE). SwiGLUFFN: replaces ReLU FFN with gate = swish(W1 x); output = gate * (W2 x), with dimension expansion factor 4/3 to keep parameter count comparable."
}
```

### Bad Mutation Proposal
```json
{
  "architecture_name": "transformer_deep",
  "mutations": ["depth_change"],
  "rationale": "More layers means better representation."
}
```
This is bad because: `depth_change` alone is not novel, the rationale is a generic
statement not a testable hypothesis, and deeper models on small synthetic data
typically overfit faster without structural improvements.
