---
name: coder-tabular
description: Keras/TF implementation patterns for tabular domain architectures. Load when generating code for tabular/structured data tasks.
agent: coder
version: 1.0
domains: [tabular]
---

# Coder Skill — Tabular Domain

## Purpose

Tabular code fails most often due to: feeding un-normalised numerical features, treating
categorical integers as continuous values, and mismatched output dimensions for binary
vs multi-class tasks. This skill provides correct patterns for mixed-type tabular data,
feature tokenization, and attention over features.

## Key Principles

- Numerical features must be normalised (mean=0, std=1) before feeding to the model.
  Compute mean/std on training set only.
- Categorical features must pass through an Embedding layer, not a Dense layer.
  Integer IDs fed directly to Dense are treated as ordinal values — incorrect.
- Input shape: (batch, n_features). After feature tokenization: (batch, n_features, embed_dim).
- Attention over features operates on the feature dimension (axis=1), not the batch.

## Synthetic Data Pattern

```python
from sklearn.datasets import make_classification
np.random.seed(42)
X, y = make_classification(n_samples=2000, n_features=20, n_informative=10,
                            n_classes=3, n_clusters_per_class=1, random_state=42)
X = X.astype(np.float32)
y = y.astype(np.int32)

# Normalise
mean, std = X.mean(axis=0), X.std(axis=0) + 1e-8
X = (X - mean) / std

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Correct Custom Layer Patterns

### Feature Tokenizer (per-feature embedding for numerical data)
```python
class FeatureTokenizer(tf.keras.layers.Layer):
    def __init__(self, n_features, embed_dim, **kwargs):
        super().__init__(**kwargs)
        # One embedding vector per feature — shape (n_features, embed_dim)
        self.W = self.add_weight(
            name='feature_weights',
            shape=(n_features, embed_dim),
            initializer='glorot_uniform',
            trainable=True,
        )
        self.b = self.add_weight(
            name='feature_biases',
            shape=(n_features, embed_dim),
            initializer='zeros',
            trainable=True,
        )

    def call(self, inputs, training=None):
        # inputs: (batch, n_features)
        # Output: (batch, n_features, embed_dim)
        x = inputs[:, :, tf.newaxis]  # (batch, n_features, 1)
        return x * self.W + self.b    # broadcast: (batch, n_features, embed_dim)
```

### Feature-wise Self-Attention
```python
class FeatureAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads
        )
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs, training=None):
        # inputs: (batch, n_features, embed_dim) — attend over features (axis=1)
        attn_out = self.attn(inputs, inputs, training=training)
        return self.norm(inputs + attn_out, training=training)
```

### Gated Feature Selector
```python
class GatedFeatureSelector(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        n_features = input_shape[-1]
        self.gate_w = self.add_weight(
            shape=(n_features, n_features), initializer='glorot_uniform', trainable=True
        )
        self.gate_b = self.add_weight(
            shape=(n_features,), initializer='zeros', trainable=True
        )

    def call(self, inputs, training=None):
        gate = tf.nn.sigmoid(tf.matmul(inputs, self.gate_w) + self.gate_b)
        return gate * inputs
```

## Common Implementation Errors

- **Unnormalised features** — Without normalisation, a feature with values in [0, 1,000,000]
  will dominate gradients over a feature in [0, 1]. Always standardise numerical features.
- **Dense applied to categorical integers** — `Dense(32)(category_id)` treats ID=5 as
  numerically 5× ID=1. Use `Embedding(vocab_size, embed_dim)(category_id)` instead.
- **Wrong output for multi-class vs binary** — Multi-class (3 classes): `Dense(3, 'softmax')` +
  `sparse_categorical_crossentropy`. Binary: `Dense(1, 'sigmoid')` + `binary_crossentropy`.
  Confusing these produces NaN losses.
- **Attention on wrong axis** — For feature attention, `MultiHeadAttention` receives
  (batch, n_features, embed_dim). The attention is over n_features, not embed_dim.
  Do not flatten to (batch, n_features * embed_dim) before attention.

## Output Configuration

- Multi-class: `Flatten()` + `Dense(n_classes, softmax)` + `sparse_categorical_crossentropy`
- Binary: `Flatten()` + `Dense(1, sigmoid)` + `binary_crossentropy`
- Regression: `Flatten()` + `Dense(1, linear)` + `mse`
