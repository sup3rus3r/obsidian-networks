---
name: architect-graph
description: Mutation strategy for graph domain architectures (GCNs, GATs, GNNs). Load when proposing mutations for graph tasks.
agent: architect
version: 1.0
domains: [graph]
---

# Architect Skill — Graph Domain

## Purpose

Graph architectures process node features and adjacency structure together. The core
operation is message passing: each node aggregates information from its neighbours,
updates its representation, then passes its new representation to its neighbours in
the next layer. The key failure modes are over-smoothing (too many message passing
steps) and ignoring edge features. This skill guides the Architect toward novel graph
architecture mutations that remain computationally feasible on synthetic data.

## Key Principles

- Limit message passing depth to 2–4 layers. Beyond 4 layers, node representations
  converge to the same vector (over-smoothing), eliminating node-level distinctions.
- The adjacency matrix must be normalized before use in GCN-style aggregation.
  Unnormalized adjacency causes gradient explosion for high-degree nodes.
- Attention-weighted aggregation (GAT-style) is more expressive than mean/sum aggregation
  and still under-explored when combined with novel attention formulations.
- Edge features are almost universally ignored by base templates — incorporating them
  is a high-novelty direction.
- On synthetic data with random graphs, message passing benefits are limited because
  the graph has no meaningful structure. Mutations should produce models that can
  still learn node classification from features even when graph structure is noise.
- Prefer `free_form` (novel aggregation or message function) and `architecture_crossover`.

## Domain-Specific Guidance

### Approved Base Templates

- **GCN** — Simple spectral-based convolution. Low novelty alone; use as scaffold for
  aggregation function mutations.
- **GAT** — Attention-weighted message passing. Better base for attention variants
  and multi-head aggregation mutations.
- **GraphSAGE** — Inductive, samples neighbourhoods. Better for large graphs.
  Mutate the aggregation function (mean → LSTM → attention).

### Known-Good Mutation Combinations

- Multi-relational Message Passing (free_form) — Separate message functions for
  different edge types. Each edge type has its own learned weight matrix; messages
  are summed across types. Tests whether relation-specific transformations improve
  node representations on heterogeneous graphs.
- Hyperbolic Embedding (architecture_crossover with `hyperbolic_geometry`) — Map node
  representations to the Poincaré disk for aggregation. Hierarchical graph structures
  (trees, taxonomies) are more naturally embedded in hyperbolic space than Euclidean.
- Virtual Node Augmentation (free_form) — Add a global virtual node connected to all
  real nodes. The virtual node aggregates global graph information and broadcasts it
  back. Improves long-range communication without deep stacking.
- `attention_variant` with distance-aware weighting — Scale attention coefficients by
  shortest-path distance between nodes. Forces the model to account for graph topology
  in its attention, not just feature similarity.
- Graph Transformer (architecture_crossover) — Replace GNN layers with Transformer
  layers where the attention mask is the adjacency matrix. Full self-attention over
  node features but constrained to edges. Rare but effective on dense graphs.

### Known Dead Ends

- GCN with > 4 message passing layers — over-smoothing makes all nodes converge.
  Always add a skip connection or jumping knowledge connection if depth > 3.
- Standard dense attention applied to all node pairs — O(n²) memory for large graphs.
  On synthetic data with n=100–500 nodes this is feasible, but scores poorly on
  efficiency axis.
- Removing adjacency normalization — unnormalized GCN aggregation causes gradient
  explosion for high-degree nodes within 1–2 epochs.
- GraphSAGE with LSTM aggregation on random synthetic graphs — the LSTM expects
  ordered neighbourhoods, but random graphs have no meaningful node ordering.

### Common Failure Modes

- **Adjacency matrix not normalized** — Multiply A by D^{-1/2} A D^{-1/2} where D is
  the degree matrix. Without this, high-degree nodes dominate aggregation and cause
  unstable training.
- **Node feature dimension mismatch after message passing** — Each GNN layer transforms
  node features. If layer L outputs dim=64 but layer L+1 expects dim=128, training fails.
  All layer dimensions must be explicitly matched.
- **Over-smoothing in deep GNNs** — Nodes become indistinguishable after 4+ layers.
  Symptom: validation accuracy stagnates early and the model outputs near-identical
  predictions for all nodes. Fix: reduce depth, add skip connections, or use JK-Net.
- **Sparse adjacency in dense matrix format** — Storing the full N×N adjacency as a
  dense matrix for N > 1000 nodes causes OOM. Use sparse representations or reduce N.

## Examples

### Good Mutation Proposal
```json
{
  "architecture_name": "gat_virtual_node_hyperbolic",
  "mutations": ["free_form", "architecture_crossover"],
  "rationale": "Augment GAT with a virtual global node for long-range communication, then project node embeddings into hyperbolic space before classification — tests whether hierarchical graph structure is better captured in non-Euclidean geometry.",
  "free_form_description": "VirtualNodeGAT: add one virtual node to the graph connected to all real nodes with learnable edge weights. Run 2 GAT layers. Then apply a Poincaré disk projection: for each node embedding x, map to hyperbolic space via exp_map(x/||x||) and apply a hyperbolic linear layer before the classifier."
}
```

### Bad Mutation Proposal
```json
{
  "architecture_name": "gcn_deeper",
  "mutations": ["depth_change"],
  "rationale": "More message passing layers capture longer-range dependencies."
}
```
This is bad because: > 4 GCN layers causes over-smoothing, the rationale states
a known fact rather than an unexplored hypothesis, and `depth_change` alone has
low novelty.
