---
name: researcher-graph
description: arXiv query strategy and paper selection guidance for graph domain research. Load when running research for graph/GNN tasks.
agent: researcher
version: 1.0
domains: [graph]
---

# Researcher Skill — Graph Domain

## Purpose

Graph neural network research is rich in novel aggregation functions, message passing
schemes, and positional encoding methods. The most useful papers describe the exact
mathematical form of the message function or aggregation operator — these translate
directly into implementable mechanisms. Papers that only prove theoretical expressiveness
bounds or benchmarks on standard graphs (Cora, Citeseer) without a clear novel operation
are less useful.

## Key Principles

- Target papers with explicit message passing formulas in the abstract or methods section.
- Attention-weighted aggregation papers contain the most extractable mechanisms.
- Edge feature incorporation is a high-value but underexplored direction — prioritise
  papers that describe how edge features enter the message computation.

## Procedure

1. For each slot, generate one query targeting message passing/aggregation mechanisms
   and one targeting graph attention variants or positional encoding.
2. Select papers with explicit node update and aggregation equations.
3. Extract: the message function m(h_u, h_v, e_uv), the aggregation function AGG,
   and the node update rule h_v' = UPDATE(h_v, AGG({m_uv})).

## High-Value Query Angles for Graph

- Attention aggregation: `"graph attention network mechanism aggregation novel architecture"`
- Edge features: `"edge feature message passing graph neural network architecture"`
- Over-smoothing solutions: `"over-smoothing graph neural network residual jumping knowledge"`
- Heterogeneous graphs: `"heterogeneous graph neural network multi-relational message passing"`
- Graph Transformers: `"graph transformer attention architecture node classification"`
- Hyperbolic: `"hyperbolic graph neural network Poincaré embedding node"`
- Higher-order: `"higher-order graph neural network k-WL message passing expressiveness"`
- Positional encoding: `"graph positional encoding Laplacian eigenvector structural"`

## Paper Selection Criteria

**Select if abstract contains:**
- Explicit message passing or aggregation equation
- "We propose a novel aggregation/attention/message function"
- Multi-relational, heterogeneous, or higher-order graph operations
- Solutions to over-smoothing

**Skip if abstract primarily contains:**
- Only benchmark comparisons (Cora, Citeseer, OGB) without architectural description
- Theoretical expressiveness proofs without practical architecture changes
- Drug discovery or molecule-specific applications without transferable mechanisms

## Mechanism Extraction Focus

- The message function: m(h_u, h_v, e_uv) — how source, target, and edge features combine
- The aggregation rule: sum, mean, attention-weighted, LSTM
- The update formula: how the aggregated message updates the node representation
- Any position encoding: Laplacian eigenvectors, random walks
