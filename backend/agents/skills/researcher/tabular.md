---
name: researcher-tabular
description: arXiv query strategy and paper selection guidance for tabular domain research. Load when running research for tabular/structured data tasks.
agent: researcher
version: 1.0
domains: [tabular]
---

# Researcher Skill — Tabular Domain

## Purpose

Tabular deep learning research is sparser than vision or language, but has seen rapid
growth since 2020 with TabNet, FT-Transformer, TabTransformer, and SAINT. The most useful
papers for this pipeline describe novel feature interaction mechanisms, embedding strategies
for mixed-type tabular data, or training objectives that address the challenges of tabular
learning (high dimensionality, mixed types, class imbalance, limited data).

## Key Principles

- The core challenge in tabular learning is feature interaction — query for papers
  proposing novel interaction functions, not just deeper MLPs.
- Attention over features (not samples) is the key architectural innovation in tabular
  deep learning. Target this direction specifically.
- Papers that compare to gradient-boosted trees and win are more likely to contain
  genuine architectural innovations than papers that only compare to MLPs.

## Procedure

1. For each slot, generate one query targeting feature interaction mechanisms and one
   targeting embedding strategies or training objectives for tabular data.
2. Select papers with explicit feature tokenization or interaction formulas.
3. Extract: the tokenization method, the attention over features formula, or the
   interaction function.

## High-Value Query Angles for Tabular

- Feature tokenization: `"feature tokenization transformer tabular data embedding attention"`
- Attention over features: `"attention mechanism feature interaction tabular neural network"`
- Sparse feature selection: `"sparse feature selection gating tabular deep learning"`
- Mixed types: `"categorical numerical embedding tabular architecture deep learning"`
- Contrastive tabular: `"contrastive learning tabular data representation self-supervised"`
- Regularisation: `"regularization overfitting tabular deep learning architecture"`
- AutoML tabular: `"neural architecture search tabular structured data"`
- Boosting-competitive: `"tabular deep learning XGBoost competitive benchmark architecture"`

## Paper Selection Criteria

**Select if abstract contains:**
- Feature tokenization or per-feature embedding design
- Attention over feature dimension (not sequence/spatial)
- Novel interaction function between features
- Mixed categorical/numerical handling mechanism

**Skip if abstract primarily contains:**
- Only tabular benchmarks without architectural description
- Domain-specific structured data (electronic health records, financial) without
  transferable mechanisms
- Purely gradient boosting or tree-based methods (no neural component)

## Mechanism Extraction Focus

- The feature tokenization formula: how each feature becomes a vector
- The feature attention formulation: how features attend to each other
- Any gating or masking applied to features
- The interaction function for pairwise or higher-order feature combinations
