---
name: platform-vision
description: Guides the platform agent through RESEARCH → PLAN → BUILD for vision tasks (CNN, ViT, image classification/detection). Load when dataset_type is "image" or the user describes a vision/CNN task.
agent: platform
version: 1.0
domains: [vision]
---

# Platform Skill — Vision Domain

## Purpose

Vision models in TensorFlow/Keras require spatial feature maps preserved until late pooling,
and they require synthetic data generation when no image files are uploaded. This skill
guides the agent in researching architectures appropriate for the stated input resolution
and class count, planning a concrete Conv2D/ViT pipeline, and generating valid code that
avoids the most common spatial-dimension collapse and shape errors.

## Key Principles

- Do NOT flatten or use GlobalAveragePooling2D before sufficient spatial reduction.
- Do NOT use BatchNorm with batch_size < 32 — use LayerNorm or GroupNorm instead.
- In description mode (no dataset), generate synthetic tensors with `tf.random.normal(shape=(N, H, W, C))` matching the stated resolution — NEVER reference a file path.
- Skip connections are required for depth ≥ 4 conv blocks — without them models do not converge in 5 epochs on synthetic data.
- ViT patch embedding dim must equal the transformer's `d_model` in every layer.
- The `num_heads × key_dim` product must equal `embed_dim` in every MultiHeadAttention call.

## Procedure

### RESEARCH Phase

1. Run both arXiv searches in parallel:
   - `"image classification convolutional neural network efficient architecture 2024"`
   - `"vision transformer ViT patch embedding image recognition 2023 2024"`
2. Select 3–4 papers. Prefer papers that include concrete hyperparameters (kernel sizes, depths, embedding dims).
3. Ingest in parallel.
4. Fetch TF/Keras docs in parallel:
   - `"Conv2D BatchNormalization residual block functional API"`
   - `"MultiHeadAttention ViT patch embedding position encoding"`
   - `"GlobalAveragePooling2D image classification output"`
5. Call `finalize_research()`.

### PLANNING Phase

Query at least 6 times before writing:
- `"optimal Conv2D depth image classification 32x32"`
- `"skip connection residual block image classification"`
- `"batch normalization convolutional layers placement"`
- `"learning rate cosine schedule image classification"`
- `"data augmentation random flip rotation crop"`
- `"ViT patch size embedding dimension small dataset"`

The plan MUST include:
- Input tensor shape: `(H, W, C)` — stated by user or `(32, 32, 3)` default for description mode
- Whether to use CNN or ViT — justified by image resolution and dataset size
- Exact Conv2D layer depths, filter counts, kernel sizes, and stride
- Where GlobalAveragePooling2D appears (after final conv block only)
- Skip connection design (if depth ≥ 4)
- Data augmentation pipeline (random flip, rotation) in dataset mode

### BUILD Phase

Follow the standard BUILD SEQUENCE. Key vision-specific steps:

1. For dataset mode: after `approve_plan`, run `run_code` to inspect shape and sample:
   ```python
   import pandas as pd
   df = pd.read_csv('dataset.csv')
   print(df.shape, df.columns.tolist()[:5])
   ```
2. In `edit_script`:
   - For description mode: `X = tf.random.normal(shape=(500, H, W, C), dtype=tf.float32)`
   - For CNN: build with Functional API — Input → Conv2D blocks → GlobalAveragePooling2D → Dense
   - For ViT: Conv2D(embed_dim, patch_size, strides=patch_size) for patch extraction, then Reshape → TransformerBlock × N → GlobalAveragePooling1D → Dense
   - Normalize pixel values: `x = tf.cast(x, tf.float32) / 255.0` when using real images
3. For ViT: verify `embed_dim = num_heads × key_dim` before writing the script.

## Domain-Specific Guidance

### Approved Architecture Patterns

- **Small CNN** (3–4 conv blocks, 32→64→128 filters, 3×3 kernels, stride-2) — for 32×32 to 64×64 inputs, < 20 classes
- **ResNet-style** (4–6 blocks with residual shortcuts, 64→128→256 filters) — for 64×64+ inputs, > 20 classes
- **ViT-Tiny** (patch_size=4, embed_dim=64, 4 heads, 4 transformer blocks) — for 32×32+, attention-based task

### Spatial Dimension Budget (32×32 input)

| After | Resolution |
|---|---|
| Input | 32×32 |
| stride-2 pool/conv | 16×16 |
| stride-2 pool/conv | 8×8 |
| stride-2 pool/conv | 4×4 |
| GlobalAveragePooling2D | (,) — flat |

Maximum 3 stride-2 operations before GlobalAveragePooling2D on 32×32 input.
For 64×64: 4 stride-2 ops are safe.

### Known Dead Ends

- MaxPooling2D after every conv block at 32×32 input — spatial dims collapse too fast.
- ViT with patch_size=1 — generates 1024 tokens per image; too expensive.
- Cross-attention in a standard CNN encoder — no second sequence to attend to.
- `model.fit(images)` without normalizing to [0, 1] or [-1, 1] — gradients explode.
- More than 8 filters in early conv layers for 32×32 synthetic data — overkill, slow to converge.

### Common Failure Modes

- **Spatial collapse** — check: 32×32 with 4 stride-2 ops = 2×2 before GlobalAvgPool. This causes shape errors in the shortcut Add(). Never go below 4×4 before pooling.
- **ViT shape mismatch** — `embed_dim` changed in one layer but not the QKV projection size. All MultiHeadAttention in the same block must use the same `d_model`.
- **NaN loss in CNN** — missing BatchNorm after Conv2D in deep (5+ block) networks. Add BatchNorm after every Conv2D.
- **Conv2D + residual Add() shape error** — shortcuts must match the conv output channels. Project with Conv2D(filters, 1, padding='same') when channel count changes.

## Examples

### Good Plan Architecture Section

```
### Selected Model: Small ResNet-style CNN
Input: (32, 32, 3) — synthetic RGB images
Conv2D(32, 3, padding='same') → BatchNorm → ReLU   [He 2016, https://...]
Conv2D(32, 3, stride=2, padding='same') → BN → ReLU  [He 2016]
+ shortcut: Conv2D(32, 1, stride=2)(input)          [skip connection for stability]
Conv2D(64, 3, padding='same') → BN → ReLU
Conv2D(64, 3, stride=2, padding='same') → BN → ReLU
+ shortcut: Conv2D(64, 1, stride=2)(prev_out)
GlobalAveragePooling2D → Dense(10, softmax)
```

### Bad Plan Architecture Section

```
### Selected Model: CNN
Some Conv2D layers followed by pooling and Dense.
```
This is bad: no filter counts, no skip connections, no spatial dimension tracking, no citations.
