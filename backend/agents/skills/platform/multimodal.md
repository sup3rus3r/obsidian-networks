---
name: platform-multimodal
description: Guides the platform agent through RESEARCH → PLAN → BUILD for multimodal tasks (image-text contrastive learning, CLIP-style dual encoders). Load when the user describes a multimodal or cross-modal task.
agent: platform
version: 1.0
domains: [multimodal]
---

# Platform Skill — Multimodal Domain

## Purpose

Multimodal models on the platform combine an image encoder and a text encoder into a shared
embedding space. The standard loss is InfoNCE (contrastive), which pushes matching image-text
pairs together and non-matching pairs apart. This skill guides the agent in researching
contrastive learning architectures, planning the dual-encoder and projection dimensions,
and generating code that implements the InfoNCE loss with numerically stable log-softmax
and correctly normalises embeddings to unit length before computing cosine similarity.

## Key Principles

- Embeddings MUST be L2-normalised before cosine similarity: `tf.linalg.l2_normalize(embeds, axis=-1)`.
- InfoNCE loss requires matching diagonal labels — the label for image i is item i (not a fixed 0/1).
- Temperature τ must be a learnable parameter or a tuned constant (0.07 is the CLIP default).
- Both encoders must project to the SAME `projection_dim` — otherwise cosine similarity is invalid.
- In description mode: `images = np.random.randn(500, 32, 32, 3).astype(np.float32); tokens = np.random.randint(1, 5000, (500, 64)).astype(np.int32)`.
- Do NOT use `model.fit()` for contrastive training — implement the InfoNCE loop with `tf.GradientTape`.

## Procedure

### RESEARCH Phase

**CRITICAL: Never ingest paper URLs from your training knowledge. Only ingest URLs returned by arXiv search results — do not use papers you already know.**

1. Run both arXiv searches using the `search_arxiv` tool — use only the returned results:
   - `"CLIP contrastive image text pre-training dual encoder 2024"`
   - `"InfoNCE contrastive loss multimodal representation learning 2023 2024"`
2. Select 3–4 papers. Focus on: projection dimension, temperature, loss formulation, encoder architecture.
3. Ingest in parallel.
4. Fetch TF/Keras docs in parallel:
   - `"tf.linalg.l2_normalize cosine similarity embeddings TensorFlow"`
   - `"Conv2D GlobalAveragePooling2D image encoder projection"`
   - `"Embedding GlobalAveragePooling1D text encoder projection"`
5. Call `finalize_research()`.

### PLANNING Phase

Query at least 6 times before writing:
- `"CLIP dual encoder image text projection dimension"`
- `"InfoNCE contrastive loss temperature softmax formula"`
- `"image encoder CNN ViT patch embedding contrastive"`
- `"text encoder embedding transformer contrastive learning"`
- `"embedding normalisation cosine similarity contrastive"`
- `"temperature parameter learning rate contrastive loss"`

The plan MUST include:
- `projection_dim` (same for both encoders, e.g. 128–256) with source
- Image encoder architecture: Conv2D+GlobalAveragePooling2D+Dense(projection_dim)
- Text encoder architecture: Embedding+GlobalAveragePooling1D+Dense(projection_dim)
- InfoNCE loss formula written out: `-log(exp(sim_ii/τ) / Σ_j exp(sim_ij/τ))`
- Temperature τ (fixed or learnable) with source
- Whether to use a custom `tf.GradientTape` loop (required for InfoNCE)

### BUILD Phase

Follow the standard BUILD SEQUENCE. Key multimodal-specific steps:

1. In `edit_script`:
   - Build two separate encoders that both project to `projection_dim`
   - Image encoder: `Input(H,W,C)→Conv2D(32,3,relu)→GlobalAveragePooling2D→Dense(projection_dim)`
   - Text encoder: `Input(seq_len,dtype=int32)→Embedding(vocab+1,embed_dim)→GlobalAveragePooling1D→Dense(projection_dim)`
   - L2-normalise BOTH: `img_emb = tf.linalg.l2_normalize(img_emb, axis=-1)`
   - Cosine similarity matrix: `sim = tf.matmul(img_emb, txt_emb, transpose_b=True) / temperature`
   - InfoNCE loss: `labels = tf.range(batch_size); loss = (crossentropy(labels, sim) + crossentropy(labels, tf.transpose(sim))) / 2`
2. Use `tf.GradientTape` with BOTH encoders' trainable variables in one tape.
3. Save combined model (or image encoder alone) to `output/model.keras`.

## Domain-Specific Guidance

### InfoNCE Loss Implementation

```python
temperature = 0.07  # or tf.Variable(0.07, trainable=True)

def infonce_loss(img_emb, txt_emb):
    img_emb = tf.linalg.l2_normalize(img_emb, axis=-1)
    txt_emb = tf.linalg.l2_normalize(txt_emb, axis=-1)
    logits = tf.matmul(img_emb, txt_emb, transpose_b=True) / temperature
    labels = tf.range(tf.shape(logits)[0])
    loss_i2t = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    loss_t2i = tf.keras.losses.sparse_categorical_crossentropy(labels, tf.transpose(logits), from_logits=True)
    return (tf.reduce_mean(loss_i2t) + tf.reduce_mean(loss_t2i)) / 2.0
```

### Architecture Defaults

| Component | Default | Notes |
|---|---|---|
| `projection_dim` | 128 | Both encoders must project to this size |
| Temperature τ | 0.07 (learnable) | Lower → sharper distribution; CLIP used 0.07 |
| Image encoder | Conv2D + GAP + Dense | ViT for higher capacity |
| Text encoder | Embedding + GAP + Dense | Transformer for richer representations |
| Batch size | 64–256 | Larger batch = more negatives = stronger contrastive signal |

### Known Dead Ends

- Computing cosine similarity without L2 normalisation — inner product, not cosine; loss is dominated by embedding magnitude.
- Fixed label `0` for all positive pairs — InfoNCE requires the positive pair to be the diagonal: label for image i = text i.
- Very small temperature (< 0.01) — logits become too sharp; gradients vanish for non-diagonal pairs.
- `model.fit()` — InfoNCE requires the full batch similarity matrix computed inside the loss; impossible in standard `fit()`.
- Separate training for image/text encoders — they must be trained jointly so their embedding spaces align.

### Common Failure Modes

- **Loss becomes NaN** — usually un-normalised embeddings + low temperature. Always normalise before computing `tf.matmul` similarity.
- **Contrastive loss does not decrease** — batch size too small (< 16). With only 16 negatives, the contrastive signal is weak; use ≥ 64.
- **Both encoders collapse** — encoders output near-zero or constant vectors. Add `tf.debugging.check_numerics` and verify gradients are non-zero after the first step.
- **Shape error in matmul** — `img_emb` and `txt_emb` must both be `(batch, projection_dim)`. Check that `GlobalAveragePooling` reduces the spatial/sequence dimensions correctly.

## Examples

### Good Plan Architecture Section

```
### Selected Model: CLIP-style Dual Encoder
projection_dim=128, temperature=0.07 (learnable)  [Radford 2021, https://...]

Image encoder: Input(32,32,3) → Conv2D(32,3,relu)→BN → Conv2D(64,3,relu,s=2)→BN → GlobalAvgPool2D → Dense(128)
Text encoder:  Input(64,) int32 → Embedding(5001,64) → GlobalAvgPool1D → Dense(128)

L2-normalise both embeddings.
InfoNCE loss: symmetric cross-entropy on (batch×batch) cosine-similarity matrix.
Training: tf.GradientTape, joint update of both encoders.
Optimizer: Adam(lr=1e-4). Epochs: 30.
```

### Bad Plan Architecture Section

```
### Selected Model: Multimodal Model
Combine image and text features.
```
This is bad: no projection_dim, no loss formula, no normalisation strategy, no encoder architecture, no citations.
