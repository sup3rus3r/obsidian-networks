---
name: researcher-multimodal
description: arXiv query strategy and paper selection guidance for multimodal domain research. Load when running research for multimodal tasks.
agent: researcher
version: 1.0
domains: [multimodal]
---

# Researcher Skill — Multimodal Domain

## Purpose

Multimodal research is dominated by large-scale vision-language models (CLIP, ALIGN, LLaVA)
that are impossible to implement from scratch. The most useful papers for this pipeline
describe cross-modal alignment mechanisms, fusion architectures, and training objectives
that can be applied to small models trained on synthetic bimodal data. Target the fusion
mechanism literature, not the large-scale pre-training literature.

## Key Principles

- Avoid papers requiring large-scale pre-training (CLIP, DALL-E, GPT-4V). These have no
  extractable mechanism applicable to a self-contained training script.
- Target papers describing cross-modal attention formulations, fusion strategies, and
  alignment losses that work at small scale.
- Contrastive alignment (pulling matched pairs close in embedding space) is the most
  extractable and implementable multimodal mechanism.

## Procedure

1. For each slot, generate one query targeting cross-modal attention or fusion mechanisms
   and one targeting alignment objectives or modality balance techniques.
2. Select papers describing fusion at training scale (not just inference) and with
   explicit cross-modal attention or alignment formulas.
3. Extract: the cross-modal attention formula, the alignment loss, or the modality
   gating/weighting computation.

## High-Value Query Angles for Multimodal

- Cross-modal attention: `"cross-modal attention fusion mechanism multimodal architecture"`
- Alignment loss: `"contrastive multimodal alignment loss representation learning"`
- Gated fusion: `"gated multimodal fusion attention weighting architecture"`
- Early vs late fusion: `"early late fusion comparison multimodal architecture novel"`
- Modality dropout: `"modality dropout robust multimodal learning missing modality"`
- Hierarchical fusion: `"hierarchical multimodal fusion multi-level representation"`
- Graph multimodal: `"graph multimodal fusion visual language alignment"`
- Efficient multimodal: `"efficient multimodal architecture lightweight fusion mechanism"`

## Paper Selection Criteria

**Select if abstract contains:**
- Cross-modal attention formulation
- Alignment or contrastive loss between modalities
- Gating or weighting mechanism for modality fusion
- Modality robustness (handling missing/noisy modalities)

**Skip if abstract primarily contains:**
- Large-scale pre-training requirements (billions of parameters)
- Vision-language generation (image captioning, VQA at scale)
- "We fine-tune CLIP/BLIP/LLaVA" — application paper, not architecture
- Only zero-shot transfer benchmarks

## Mechanism Extraction Focus

- The cross-modal attention: how queries from one modality attend to keys from another
- The alignment loss: InfoNCE or triplet formulation for pulling matched pairs together
- The fusion gating: how the relative weight of each modality is computed
- Modality balancing: any gradient or loss weighting to prevent modality collapse
