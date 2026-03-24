---
name: architect-vision
description: Mutation strategy for vision domain architectures (CNNs, ViTs, image models). Load when proposing mutations for vision tasks.
agent: architect
version: 1.0
domains: [vision]
---

# Architect Skill — Vision Domain

## Purpose

Vision models operate on spatially-structured 2D feature maps. Mutations that ignore
spatial structure (e.g. flattening too early, using 1D convolutions, removing pooling
hierarchy) reliably fail. This skill gives the Architect concrete guidance on which
mutation directions produce genuinely novel and trainable vision architectures.

## Key Principles

- Preserve spatial hierarchy until at least the penultimate layer. Global pooling should
  come late, not early.
- Skip connections are almost always beneficial in vision — add them when increasing depth.
- Attention in vision works best at lower spatial resolutions (after pooling), not on full
  feature maps (too expensive and unstable on small synthetic data).
- Depthwise separable convolutions are an underexplored mutation direction — they reduce
  parameters dramatically while preserving receptive field.
- ViT mutations must maintain patch embedding dimensionality consistency throughout;
  the embed_dim must equal the transformer's model dimension at every layer.
- Prefer `free_form` and `architecture_crossover` operators over standard ones to achieve
  high novelty scores. Standard `layer_insertion` and `width_change` alone score poorly.

## Domain-Specific Guidance

### Approved Base Templates

- **CNN** — Use for spatial hierarchy tasks and when compute is limited. Strong baseline,
  low novelty by itself — must be mutated meaningfully.
- **ViT** — Use when proposing attention-based mutations. Requires larger embed_dim (≥ 64)
  to be stable on 32×32 synthetic data. Fewer transformer layers (2–4) work better
  than the default 6 on limited data.

### Known-Good Mutation Combinations

- `skip_connection_add` + `attention_variant` — Adding spatial attention after residual
  blocks is a genuine open area. Combine them: residual output is re-weighted by a
  learned spatial attention map before passing to the next block.
- `depthwise_separable` (free_form) + `normalization_change` — Replace standard Conv2D
  with depthwise separable convolution and switch to GroupNorm. Reduces parameters
  significantly and often improves generalisation on small datasets.
- `architecture_crossover` with `fourier_neural_operator` — Replace some convolutional
  blocks with Fourier-domain mixing layers. Genuinely unexplored for image classification.
- `architecture_crossover` with `capsule_network` — Dynamic routing instead of max
  pooling for part-whole relationships. Still rare in production vision systems.
- Multi-scale feature fusion (free_form) — Collect feature maps at multiple depths and
  fuse them via learned weights before classification. Different from U-Net skip connections
  in that the fusion is learned, not concatenated.

### Known Dead Ends

- Deep CNN (> 8 conv layers) on 32×32 synthetic input — spatial resolution collapses
  after repeated pooling. Maximum useful depth on 32×32 is 4–5 conv layers.
- Cross-attention in a ViT without a separate encoder — there is no second sequence to
  attend to. Cross-attention requires encoder-decoder structure.
- Standard `layer_insertion` adding another Dense layer after global pooling — this adds
  parameters without novel structure. Score poorly on novelty.
- Raw pixel convolution with very large kernels (≥ 11×11) — rarely better than 3×3 chains
  and causes parameter explosion.
- Removing all normalization — without BatchNorm or LayerNorm, deep vision models diverge
  on synthetic data within a few epochs.

### Common Failure Modes

- **Spatial dimension collapse** — Too many MaxPool2D layers reduce feature maps to 1×1
  before global pooling. Check: with 32×32 input and 2 stride-2 pools, resolution is 8×8.
  Three pools = 4×4. Four pools = 2×2. Stop there.
- **ViT embedding inconsistency** — Changing `embed_dim` in one layer but not others
  causes shape mismatch in the transformer blocks. All transformer layers must share
  the same model dimension.
- **Attention instability on small batches** — Multi-head attention with too many heads
  relative to key_dim causes NaN. Rule: `num_heads × key_dim = embed_dim`. For embed_dim=64,
  use 4 heads × key_dim=16, not 8 heads × key_dim=8.
- **Gradient vanishing in deep CNN without skip connections** — Any CNN with > 4 blocks
  must include skip connections (residual or dense) or it will not learn on synthetic data
  within 5 epochs.

## Examples

### Good Mutation Proposal
```json
{
  "architecture_name": "cnn_fourier_spatial_attention",
  "mutations": ["architecture_crossover", "attention_variant"],
  "rationale": "Replace mid-level conv blocks with Fourier mixing layers, then apply learned spatial attention before global pooling — tests whether frequency-domain feature mixing combined with spatial reweighting produces richer representations than pure spatial convolution.",
  "free_form_description": "FourierMixLayer: applies 2D FFT to the feature map, applies a learned complex-valued filter in frequency space, then iFFT back. Followed by a spatial attention gate: sigmoid(Conv2D(1, 1)(features)) * features."
}
```

### Bad Mutation Proposal
```json
{
  "architecture_name": "cnn_wider",
  "mutations": ["width_change"],
  "rationale": "Wider filters may improve accuracy."
}
```
This is bad because: `width_change` alone is not novel, the rationale is vague,
and there is no mathematical hypothesis being tested.
