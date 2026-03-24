---
name: coder-language
description: Keras/TF implementation patterns for language domain architectures. Load when generating code for language/NLP tasks.
agent: coder
version: 1.0
domains: [language]
---

# Coder Skill — Language Domain

## Purpose

Language code fails most often due to: embedding dimension mismatches in transformer
blocks, causal masks applied where they shouldn't be, and sequence length inconsistencies.
This skill provides correct TF/Keras patterns for transformer components, positional
encodings, and custom language model layers.

## Key Principles

- Input shape is (batch, seq_len) = (batch, 64) of integer token IDs for synthetic data.
- After Embedding, shape is (batch, seq_len, embed_dim). All transformer layers must
  preserve this shape through the sequence dimension.
- MultiHeadAttention: `num_heads × key_dim = embed_dim`. A common error is setting
  key_dim = embed_dim instead of key_dim = embed_dim // num_heads.
- For classification: use CLS token or mean-pool over the sequence, then Dense.
  Do not feed the full (batch, seq_len, embed_dim) tensor to a classifier.

## Synthetic Data Pattern

```python
np.random.seed(42)
vocab_size, seq_len, n_classes = 1000, 64, 5
X = np.random.randint(0, vocab_size, size=(2000, seq_len)).astype(np.int32)
y = np.random.randint(0, n_classes, size=(2000,)).astype(np.int32)
```

## Correct Custom Layer Patterns

### Rotary Position Embedding
```python
class RotaryEmbedding(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (tf.cast(tf.range(0, dim, 2), tf.float32) / dim))
        self.inv_freq = inv_freq

    def call(self, x, training=None):
        # x: (batch, seq_len, dim)
        seq_len = tf.shape(x)[1]
        t = tf.cast(tf.range(seq_len), tf.float32)
        freqs = tf.einsum('i,j->ij', t, self.inv_freq)  # (seq_len, dim/2)
        emb = tf.concat([freqs, freqs], axis=-1)        # (seq_len, dim)
        cos = tf.cos(emb)[tf.newaxis]                   # (1, seq_len, dim)
        sin = tf.sin(emb)[tf.newaxis]
        x1, x2 = x[..., :self.dim//2], x[..., self.dim//2:]
        rotated = tf.concat([-x2, x1], axis=-1)
        return x * cos + rotated * sin
```

### SwiGLU Feed-Forward Layer
```python
class SwiGLUFFN(tf.keras.layers.Layer):
    def __init__(self, embed_dim, expansion=4, **kwargs):
        super().__init__(**kwargs)
        hidden = int(embed_dim * expansion * 2 / 3)
        self.W_gate = tf.keras.layers.Dense(hidden, use_bias=False)
        self.W_val  = tf.keras.layers.Dense(hidden, use_bias=False)
        self.W_out  = tf.keras.layers.Dense(embed_dim, use_bias=False)

    def call(self, x, training=None):
        gate = tf.nn.swish(self.W_gate(x))
        val  = self.W_val(x)
        return self.W_out(gate * val)
```

### Standard Transformer Block (reference implementation)
```python
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn  = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads
        )
        self.ffn   = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='gelu'),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = tf.keras.layers.Dropout(dropout_rate)
        self.drop2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=None):
        attn_out = self.attn(x, x, training=training)
        x = self.norm1(x + self.drop1(attn_out, training=training))
        ffn_out = self.ffn(x, training=training)
        return self.norm2(x + self.drop2(ffn_out, training=training))
```

## Common Implementation Errors

- **key_dim set to embed_dim instead of embed_dim // num_heads** — `MultiHeadAttention(
  num_heads=4, key_dim=128)` produces a model with 4 × 128 = 512-dim attention, not 128-dim.
  Correct: `MultiHeadAttention(num_heads=4, key_dim=32)` for embed_dim=128.
- **Causal mask on encoder** — For classification, do NOT pass `use_causal_mask=True` to
  MultiHeadAttention. Causal masking is only for generative (left-to-right) models.
- **Embedding output fed directly to Dense** — After LSTM/Transformer produces (batch, seq_len, d),
  you cannot feed this directly to Dense(n_classes). First reduce: `tf.reduce_mean(x, axis=1)`
  or take x[:, 0, :] (CLS token position).
- **Positional embedding shape mismatch** — Learned position embeddings must be sliced to the
  actual sequence length: `pos_emb = self.pos_embedding[:, :seq_len, :]`.

## Output Configuration

- Text classification: `GlobalAveragePooling1D()` + `Dense(n_classes, softmax)` + `sparse_categorical_crossentropy`
- Binary sentiment: `GlobalAveragePooling1D()` + `Dense(1, sigmoid)` + `binary_crossentropy`
