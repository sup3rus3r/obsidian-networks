---
name: researcher-recommendation
description: arXiv query strategy and paper selection guidance for recommendation domain research. Load when running research for recommendation tasks.
agent: researcher
version: 1.0
domains: [recommendation]
---

# Researcher Skill — Recommendation Domain

## Purpose

Recommendation research spans collaborative filtering, session-based recommendation,
and content-aware systems. The most useful papers describe novel interaction functions
between user and item representations, novel sequential modelling of user sessions,
or novel training objectives that address sparsity and cold-start. Avoid papers focused
on industry-specific systems (YouTube, Amazon) that aren't architecturally novel.

## Key Principles

- The user-item interaction function is where novelty lives. Target papers that propose
  new ways to compute relevance between user and item embeddings.
- Session-based recommendation (modelling the current user session as a sequence) is
  the richest area for novel architecture papers.
- Cold-start is a structurally important problem — papers that address it with content
  features contain useful mechanisms for hybrid models.

## Procedure

1. For each slot, generate one query targeting user-item interaction mechanisms and one
   targeting session modelling or cold-start solutions.
2. Select papers with explicit relevance scoring formulas or novel aggregation over
   interaction history.
3. Extract: the interaction function, the session encoding mechanism, or the cold-start
   feature fusion method.

## High-Value Query Angles for Recommendation

- Neural interaction: `"neural collaborative filtering interaction function attention mechanism"`
- Session-based: `"session-based recommendation transformer sequential architecture"`
- Graph CF: `"graph collaborative filtering LightGCN message passing recommendation"`
- Cold-start: `"cold start recommendation content feature hybrid architecture"`
- Contrastive: `"contrastive learning recommendation representation self-supervised"`
- Attention over history: `"attention user history item recommendation neural architecture"`
- Disentangled: `"disentangled user representation recommendation intent"`
- Knowledge-aware: `"knowledge graph recommendation embedding architecture"`

## Paper Selection Criteria

**Select if abstract contains:**
- Novel user-item relevance scoring function
- Sequential session encoder architecture
- Graph-based propagation for collaborative filtering
- Cold-start mechanism using item/user features

**Skip if abstract primarily contains:**
- Industry deployment descriptions without architectural novelty
- Privacy-preserving federated recommendation (different problem)
- Only offline metric improvements (NDCG, HR) without architectural description

## Mechanism Extraction Focus

- The interaction function: f(u, v) — how user and item embeddings combine to produce a score
- The session encoder: how the sequence of interacted items is processed
- The graph propagation: how user-item bipartite graph updates representations
- The cold-start fusion: how content features are combined with collaborative embeddings
