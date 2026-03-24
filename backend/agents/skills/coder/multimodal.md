---
name: coder-multimodal
description: Keras/TF implementation patterns for multimodal domain architectures. Load when generating code for multimodal tasks.
agent: coder
version: 1.0
domains: [multimodal]
---

# Coder Skill — Multimodal Domain

## Purpose

Multimodal code fails most often due to: modality shape mismatches in cross-attention
(different sequence lengths), projection dimension inconsistencies between modalities,
and tf.keras.Model not accepting multiple inputs cleanly. This skill provides correct
patterns for multi-input models, cross-modal attention, and alignment losses.

## Key Principles

- tf.keras.Model with multiple inputs: pass a list to `inputs` in the Model constructor.
  `model = tf.keras.Model(inputs=[img_input, txt_input], outputs=out)`.
- Before any cross-modal operation, project both modalities to the same dimension d.
- Cross-attention: query from one modality, key/value from the other.
  `attn = MultiHeadAttention(...); out = attn(query=vision, value=language, key=language)`.
- Synthetic bimodal data: generate image-like and text-like tensors with a shared label.

## Synthetic Data Pattern

```python
np.random.seed(42)
n_samples, n_classes = 1000, 5
# Image modality: (N, 32, 32, 3)
X_img = np.random.randn(n_samples, 32, 32, 3).astype(np.float32)
# Text modality: (N, 32) token IDs
vocab_size, seq_len = 500, 32
X_txt = np.random.randint(0, vocab_size, (n_samples, seq_len)).astype(np.int32)
# Shared label
y = np.random.randint(0, n_classes, n_samples).astype(np.int32)

from sklearn.model_selection import train_test_split
idx = np.arange(n_samples)
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)
```

## Correct Custom Layer Patterns

### Cross-Modal Attention
```python
class CrossModalAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads
        )
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, query_modality, kv_modality, training=None):
        # query: (batch, seq_q, d) — the modality that asks questions
        # kv:    (batch, seq_k, d) — the modality being queried
        attn_out = self.attn(
            query=query_modality,
            value=kv_modality,
            key=kv_modality,
            training=training,
        )
        return self.norm(query_modality + attn_out)
```

### Gated Modality Fusion
```python
class GatedFusion(tf.keras.layers.Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.gate = tf.keras.layers.Dense(1, activation='sigmoid')
        self.proj = tf.keras.layers.Dense(d_model)

    def call(self, vision, language, training=None):
        # vision, language: (batch, d_model) — already pooled
        combined = tf.concat([vision, language], axis=-1)
        g = self.gate(combined)                      # (batch, 1)
        fused = g * vision + (1 - g) * language     # (batch, d_model)
        return self.proj(fused)
```

### Full Multimodal Model (Functional API)
```python
def build_multimodal_model(d_model=64, n_classes=5):
    # Vision branch
    img_input = tf.keras.Input(shape=(32, 32, 3), name='image')
    x_v = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(img_input)
    x_v = tf.keras.layers.GlobalAveragePooling2D()(x_v)  # (batch, 32)
    x_v = tf.keras.layers.Dense(d_model)(x_v)             # (batch, d_model)
    x_v = x_v[:, tf.newaxis, :]                           # (batch, 1, d_model) for cross-attn

    # Text branch
    txt_input = tf.keras.Input(shape=(32,), dtype='int32', name='text')
    x_t = tf.keras.layers.Embedding(500, d_model)(txt_input)  # (batch, seq, d_model)

    # Cross-modal attention: vision queries text
    cross_attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=d_model // 4)
    attended = cross_attn(query=x_v, value=x_t, key=x_t)  # (batch, 1, d_model)
    attended = tf.squeeze(attended, axis=1)                 # (batch, d_model)

    # Fuse and classify
    x_t_pooled = tf.reduce_mean(x_t, axis=1)              # (batch, d_model)
    fused = tf.concat([attended, x_t_pooled], axis=-1)
    fused = tf.keras.layers.Dense(d_model, activation='relu')(fused)
    out = tf.keras.layers.Dense(n_classes, activation='softmax')(fused)

    return tf.keras.Model(inputs=[img_input, txt_input], outputs=out)
```

## Common Implementation Errors

- **MultiHeadAttention query/key/value dimension mismatch** — Query and key must have the
  same last dimension d. If vision is (batch, 1, 32) and text is (batch, seq, 64), project
  one to match the other before cross-attention.
- **Model.fit() with multiple inputs** — Pass inputs as a dict or list:
  `model.fit({'image': X_img_train, 'text': X_txt_train}, y_train, ...)`.
- **tf.squeeze on wrong axis** — After GlobalAveragePooling2D the shape is (batch, d).
  Adding `[:, tf.newaxis, :]` makes it (batch, 1, d) for cross-attention input.
  Squeeze it back with `tf.squeeze(x, axis=1)` after attention.
- **Saving multi-input model** — `model.save('output/model.keras')` works with the Keras v3
  format. Do not use `model.save('output/model.h5')` for multi-input models.

## Output Configuration

- Multimodal classification: `Dense(n_classes, softmax)` + `sparse_categorical_crossentropy`
- Contrastive alignment (evaluation): `model.predict([X_img, X_txt])` returns embeddings;
  cosine similarity between matched pairs should be > 0.5
