---
name: coder-vision
description: Keras/TF implementation patterns for vision domain architectures. Load when generating code for vision tasks.
agent: coder
version: 1.0
domains: [vision]
---

# Coder Skill — Vision Domain

## Purpose

Vision code fails most often due to: wrong input shapes after pooling, broken ViT patch
embedding dimensions, and attention head/key_dim mismatches. This skill provides the
correct patterns for the most common vision architecture components and flags the errors
that cause training to fail silently or crash.

## Key Principles

- Input shape is (batch, height, width, channels) = (batch, 32, 32, 3) for synthetic data.
- After two stride-2 MaxPool2D operations, spatial dims are (8, 8). After three, (4, 4).
  Global average pooling after this point produces (batch, filters). Plan pooling accordingly.
- ViT patch embedding: with patch_size=4 and input 32×32, you get (32//4)² = 64 patches.
  Each patch is flattened to 4×4×3=48 dims, then projected to embed_dim.
- Attention heads and key_dim: always verify num_heads × key_dim = embed_dim before use.
- Custom layers must inherit tf.keras.layers.Layer and implement call(self, inputs, training=None).

## Synthetic Data Pattern

```python
np.random.seed(42)
X = np.random.normal(size=(1000, 32, 32, 3)).astype(np.float32)
y = np.random.randint(0, 10, size=(1000,)).astype(np.int32)
```

## Correct Custom Layer Patterns

### Spatial Attention Gate
```python
class SpatialAttentionGate(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(1, 1, activation='sigmoid', padding='same')

    def call(self, inputs, training=None):
        gate = self.conv(inputs)  # (batch, H, W, 1)
        return inputs * gate      # broadcast over channels
```

### Depthwise Separable Conv Block
```python
class DepthwiseSeparableBlock(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.dw = tf.keras.layers.DepthwiseConv2D(3, padding='same', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pw = tf.keras.layers.Conv2D(filters, 1, use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None):
        x = self.dw(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pw(x)
        x = self.bn2(x, training=training)
        return tf.nn.relu(x)
```

### ViT Patch Embedding
```python
class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, patch_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.projection = tf.keras.layers.Dense(embed_dim)

    def call(self, images, training=None):
        batch_size = tf.shape(images)[0]
        # Extract patches
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        patch_dims = patches.shape[-1]
        n_patches = patches.shape[1] * patches.shape[2]
        patches = tf.reshape(patches, [batch_size, n_patches, patch_dims])
        return self.projection(patches)  # (batch, n_patches, embed_dim)
```

## Common Implementation Errors

- **Spatial dimension collapse** — More than 3 MaxPool2D(2,2) on 32×32 input leaves 4×4
  spatial dims. GlobalAveragePooling2D after this works but with low spatial resolution.
  Do not add a 4th pool on 32×32 input.
- **MultiHeadAttention shape** — `tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)`.
  The `key_dim` parameter sets the per-head dimension, not the total. Total = num_heads × key_dim.
  For embed_dim=128: use num_heads=4, key_dim=32. NOT num_heads=4, key_dim=128.
- **Residual connection shape mismatch** — If Conv2D changes the number of filters, the skip
  connection must also project to the same filters via a 1×1 Conv2D. Do not add a
  residual connection between layers with different channel counts without projection.
- **ViT without positional encoding** — Patch embeddings lose positional information without PE.
  Add learned position embeddings: `pos_emb = tf.Variable(tf.zeros((1, n_patches, embed_dim)))`.

## Output Configuration

- Classification: `Dense(10, activation='softmax')` + `sparse_categorical_crossentropy`
- Binary: `Dense(1, activation='sigmoid')` + `binary_crossentropy`
- Segmentation (if generating pixel masks): output shape must match input spatial dims
