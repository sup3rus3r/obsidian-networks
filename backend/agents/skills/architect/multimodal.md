---
name: architect-multimodal
description: Mutation strategy for multimodal domain architectures (vision+language, cross-modal fusion). Load when proposing mutations for multimodal tasks.
agent: architect
version: 1.0
domains: [multimodal]
---

# Architect Skill — Multimodal Domain

## Purpose

Multimodal architectures process and fuse information from multiple input types —
typically vision and language, but also audio, tabular, or others. The core challenge
is alignment: different modalities have different sequence lengths, different dimensionalities,
and different semantic granularities. A poorly designed fusion mechanism either ignores
one modality (modality collapse) or forces an artificial alignment that destroys each
modality's natural structure. This skill guides the Architect toward fusion mutations
that maintain each modality's structure while enabling genuine cross-modal interaction.

## Key Principles

- Process each modality through its own encoder before fusion. Never concatenate raw
  inputs from different modalities and feed to a shared network.
- Project all modality embeddings to a common dimension before any cross-modal operation.
  The projection dimension should be equal for all modalities.
- Modality collapse is the primary failure: one modality dominates and the other is
  ignored. Measure this by checking if the model performs the same with one modality
  masked.
- Cross-modal attention (each modality attending to the other) is more expressive than
  simple concatenation + MLP, but requires both sequences to be non-trivially long.
- Late fusion (fuse after full unimodal encoding) is more stable but less expressive
  than early fusion (fuse at intermediate layers).
- On synthetic data, each modality is typically generated independently with no real
  correlation between them. The model must find correlations even when they're weak.
- Prefer `free_form` (novel cross-modal attention, novel alignment mechanisms) and
  `architecture_crossover`.

## Domain-Specific Guidance

### Approved Base Templates

- **CNN + Transformer fusion** — CNN encodes visual features; Transformer encodes text.
  Fuse via concatenation + MLP. Mutate the fusion mechanism.

### Known-Good Mutation Combinations

- Cross-Modal Attention (free_form) — Vision attends over language tokens, language
  attends over spatial vision features. Implemented as two cross-attention blocks
  (one per direction). Tests whether bidirectional cross-modal attention improves
  fusion over unidirectional or concatenation.
- Contrastive Multimodal Pre-alignment (free_form) — Add a contrastive loss that
  pulls vision and language representations together for matched pairs and pushes
  them apart for unmatched pairs. CLIP-inspired but applied to small synthetic tasks.
- Gated Modality Fusion (free_form) — Learn a scalar gate for each modality:
  `gate_v = sigmoid(W_v * [vision; language]); gate_l = 1 - gate_v`. Final fusion
  is `gate_v * vision + gate_l * language`. Tests whether learned relative weighting
  outperforms equal fusion.
- `architecture_crossover` with `graph_message_passing` — Build a multimodal graph
  where vision patches and language tokens are nodes, with edges between semantically
  related cross-modal pairs. Run GNN to compute cross-modal representations.
- Hierarchical Fusion (free_form) — Fuse at multiple encoder depths: early, mid, and
  late. Each fusion level uses separate cross-attention. Final representations from
  all three levels are concatenated and projected. Tests whether multi-scale fusion
  outperforms single-point fusion.

### Known Dead Ends

- Concatenating raw image pixels and raw text token IDs as a single flat vector —
  fundamentally mismatches data types and scales. Images are float32 spatial arrays;
  token IDs are integers. These cannot be concatenated meaningfully.
- Forcing equal sequence lengths across modalities by truncation/padding — a 32×32
  image has 1024 spatial positions; a 64-token sentence has 64 positions. Padding
  the sentence to 1024 adds 960 meaningless tokens that dominate attention.
- Early fusion before any unimodal encoding — fusing raw features before each modality
  has been independently encoded prevents the model from learning modality-specific
  representations at all.
- Single shared encoder for all modalities — a single Transformer processing both
  image patches and text tokens without modality-specific layers fails to capture
  the structural differences between modalities.

### Common Failure Modes

- **Modality collapse** — One modality dominates: model accuracy on test set equals
  accuracy using only the dominant modality. Detect by ablating each modality and
  comparing performance. Fix: balance learning rates or add a modality-balancing
  regulariser.
- **Projection dimension mismatch** — Vision encoder outputs (batch, seq_v, d_v) and
  language encoder outputs (batch, seq_l, d_l). If d_v ≠ d_l, cross-attention fails.
  Always project both to shared dimension d before cross-attention.
- **Synthetic modality generation inconsistency** — If image and text are generated
  independently with no label correlation, the model has no cross-modal signal to learn.
  On synthetic data, ensure that the label is determined by both modalities jointly.
- **Attention mask mismatch in cross-attention** — If vision has seq_len=196 patches
  and language has seq_len=32 tokens, the cross-attention query/key dimensions are
  incompatible. Cross-attention query comes from one modality and key/value from the other.

## Examples

### Good Mutation Proposal
```json
{
  "architecture_name": "multimodal_gated_contrastive_fusion",
  "mutations": ["free_form", "attention_variant"],
  "rationale": "Apply learned gating to balance vision and language contributions to the fused representation, with a contrastive alignment loss during training — tests whether explicit modality balancing combined with representational alignment reduces modality collapse on synthetic bimodal tasks.",
  "free_form_description": "GatedModalFusion: after unimodal encoding, project to common dim d=128. Compute gate g = sigmoid(Linear([v; l])). Fused = g * v_proj + (1-g) * l_proj. ModalAlignmentLoss: InfoNCE loss over (vision, language) pairs within batch — matched pairs as positives, unmatched as negatives. temperature=0.07. Total loss = task_loss + 0.1 * alignment_loss."
}
```

### Bad Mutation Proposal
```json
{
  "architecture_name": "multimodal_concatenate_dense",
  "mutations": ["layer_insertion"],
  "rationale": "Concatenating features and adding Dense layers improves fusion."
}
```
This is bad because: simple concatenation + Dense is the baseline, not a mutation.
Adding a Dense layer (`layer_insertion`) to this baseline produces no structural novelty.
