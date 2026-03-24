---
name: researcher-vision
description: arXiv query strategy and paper selection guidance for vision domain research. Load when running research for vision tasks.
agent: researcher
version: 1.0
domains: [vision]
---

# Researcher Skill — Vision Domain

## Purpose

Vision research on arXiv is extremely high-volume. Generic queries like "image classification"
return thousands of papers, most of which are benchmarking papers, not novel architecture papers.
This skill provides query angles and selection criteria that reliably surface papers with
genuinely novel architectural mechanisms rather than incremental accuracy improvements.

## Key Principles

- Query for mechanisms and mathematical concepts, not tasks. "Adaptive spatial attention
  mechanisms convolutional" surfaces different (more useful) papers than "image classification CNN".
- Prioritise papers that include pseudocode, mathematical formulations, or architecture diagrams
  in their abstract or methods section.
- Recent papers (2022–2025) on architectural innovations are preferred over older benchmarking papers.
- A paper is worth downloading if its abstract describes a novel layer, operation, or training
  objective — not just if it achieves a new state-of-the-art on ImageNet.

## Procedure

1. Generate query pairs covering different architectural angles, not the same angle twice.
2. For each slot, use one query targeting attention/interaction mechanisms and one
   targeting normalization, training dynamics, or loss formulations.
3. When selecting among retrieved papers, prefer those whose abstracts mention:
   - Custom layer designs (e.g. "we propose a novel X layer")
   - Mathematical formulations (equations in the abstract)
   - Ablation studies (shows the authors tested specific mechanisms)
4. Avoid papers that only mention benchmark datasets (CIFAR, ImageNet) without
   describing a novel architectural contribution.

## High-Value Query Angles for Vision

- Spatial attention mechanisms: `"spatial attention feature recalibration CNN vision"`
- Multi-scale feature fusion: `"multi-scale feature pyramid attention fusion image"`
- Frequency-domain operations: `"Fourier domain convolution image classification spectral"`
- Efficient attention for vision: `"linear attention vision transformer efficient spatial"`
- Dynamic convolutions: `"dynamic convolution adaptive kernel image recognition"`
- Normalisation variants: `"group normalization instance normalization vision architecture"`
- Skip connection innovations: `"residual connection dense connection feature reuse CNN"`
- Hybrid CNN-Transformer: `"CNN transformer hybrid local global attention vision"`

## Paper Selection Criteria

**Select if the abstract contains:**
- A named novel layer or module (e.g. "Squeeze-and-Excitation", "Deformable Convolution")
- A mathematical expression (even if informal in the abstract)
- Terms: "novel", "propose", "we introduce", "architecture", "mechanism", "layer"

**Skip if the abstract primarily contains:**
- Only dataset names and accuracy numbers without architectural description
- "We fine-tune" or "we apply" — these are application papers, not architecture papers
- Survey or review papers

## Mechanism Extraction Focus

When extracting insights from downloaded papers, focus on:
- The mathematical form of the attention or convolution operation
- The specific innovation relative to the base (what is different from standard ResNet/ViT)
- Any SymPy-expressible computation (attention score, gating function, normalisation)
- The ablation result that shows which part of the mechanism matters most
