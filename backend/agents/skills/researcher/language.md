---
name: researcher-language
description: arXiv query strategy and paper selection guidance for language domain research. Load when running research for language/NLP tasks.
agent: researcher
version: 1.0
domains: [language]
---

# Researcher Skill — Language Domain

## Purpose

Language and NLP research is the single highest-volume domain on arXiv. Without specific
query strategies, results are dominated by LLM fine-tuning papers and RLHF/alignment work,
which have no novel architectural mechanisms to extract. This skill focuses queries toward
attention mechanism innovations, architectural efficiency improvements, and novel training
objectives — areas that produce extractable mathematical mechanisms.

## Key Principles

- Avoid queries that return LLM fine-tuning, RLHF, prompt engineering, or RAG papers —
  these have no architectural mechanism to extract.
- Target papers that describe changes to the attention computation, position encoding,
  feed-forward structure, or tokenisation — these have extractable mechanisms.
- Include the term "architecture" or "mechanism" in queries to filter toward structural papers.

## Procedure

1. For each slot, generate one query targeting attention mechanism variants and one
   targeting positional encoding or feed-forward block innovations.
2. When selecting among retrieved papers, prefer those describing novel formulas for
   attention scoring, position encoding, or gating.
3. Extract the specific mathematical operation that is novel, not just the task it
   was applied to.

## High-Value Query Angles for Language

- Attention variants: `"efficient attention mechanism linear transformer architecture"`
- Position encoding: `"rotary position embedding relative position encoding transformer"`
- Feed-forward innovations: `"gated linear unit SwiGLU feed-forward transformer layer"`
- State space models: `"state space model selective structured sequence language"`
- Sparse attention: `"sparse attention local window transformer language model"`
- Memory-augmented: `"external memory neural network language model augmentation"`
- Mixture-of-experts: `"mixture of experts sparse activation transformer architecture"`
- Training objectives: `"contrastive language representation learning novel objective"`

## Paper Selection Criteria

**Select if abstract contains:**
- A modified attention formula or scoring function
- Named novel components: "Rotary", "ALiBi", "SwiGLU", "Mamba", "RetNet"
- "We propose a novel architecture/layer/mechanism"
- Ablation study results

**Skip if abstract primarily contains:**
- "GPT", "ChatGPT", "LLaMA", "instruction tuning", "RLHF", "alignment"
- "We fine-tune a pre-trained model"
- Downstream task benchmarks only (GLUE, SuperGLUE) with no architectural description

## Mechanism Extraction Focus

- The exact attention scoring formula (query-key interaction)
- The position encoding computation (how positions affect attention weights)
- Any gating function in feed-forward layers
- State transition equations for SSM-based models
