---
name: architect-recommendation
description: Mutation strategy for recommendation domain architectures (collaborative filtering, neural CF, session-based). Load when proposing mutations for recommendation tasks.
agent: architect
version: 1.0
domains: [recommendation]
---

# Architect Skill — Recommendation Domain

## Purpose

Recommendation architectures learn to match users to items based on interaction history,
side information, or both. The fundamental challenge is the sparsity of the interaction
matrix — most users have interacted with only a tiny fraction of items. Cold start (new
users or new items with no history) is a related problem that architectures must address
structurally, not as an afterthought. This skill guides the Architect toward mutations
that address sparsity, cold start, and the distinction between collaborative and
content-based signals.

## Key Principles

- Embedding lookup for users and items is the foundation. Embedding dimensions should
  be consistent (user_dim = item_dim = d) so dot-product similarity is well-defined.
- Pure collaborative filtering ignores item content — this is a known weakness on
  cold-start. Any novel mutation should incorporate item or user features alongside IDs.
- Dot-product similarity (matrix factorisation) is the simplest interaction function.
  Neural interaction functions (MLP, attention) are more expressive but overfit on
  small datasets without regularisation.
- Temporal dynamics matter: user preferences change over time. Session-based models
  (LSTM/Transformer over the user's recent interaction sequence) address this.
- On synthetic data, user-item interaction matrices are randomly sparse. Mutations
  must produce models that can still learn from sparse signals.
- Prefer `free_form` (novel interaction functions, novel session models) and
  `architecture_crossover`.

## Domain-Specific Guidance

### Approved Base Templates

- **Matrix Factorisation (MF)** — Simple embedding dot-product. Mutate the interaction
  function, the regularisation, or add side information.
- **Neural Collaborative Filtering (NCF)** — Concatenate user and item embeddings,
  pass through MLP. Mutate the fusion mechanism or add attention over item history.

### Known-Good Mutation Combinations

- Session-Based Transformer (free_form) — Encode the user's recent interaction sequence
  with a Transformer (items as tokens, positional encoding over time). Final token or
  mean pooling represents the user's current state. Tests whether sequential modelling
  of preferences outperforms static user embeddings.
- Dual-Space Interaction (free_form) — Learn separate embedding spaces for collaborative
  (ID-based) and content-based (feature-based) signals. Fuse via a learned weighting.
  Tests whether maintaining separate representation spaces reduces the collision between
  collaborative and content signals.
- Graph-Enhanced CF (architecture_crossover with `graph_message_passing`) — Build a
  bipartite user-item graph and run message passing: user representations aggregate from
  interacted items, item representations aggregate from users who liked them. LightGCN-inspired.
- Contrastive User Representation (free_form) — Add a contrastive loss: user embeddings
  for the same user at different sessions should be close; embeddings for different users
  should be far. Tests whether contrastive regularisation improves embedding quality.
- Attention-Weighted Item History (free_form) — Instead of averaging all past interactions,
  learn attention weights over the user's history. Recent interactions get higher weight
  by default (recency bias), but the model can override this.

### Known Dead Ends

- Treating recommendation as a classification problem with n_items classes —
  for n_items > 1000, the softmax over all items is extremely slow and the sparse
  positive labels cause near-zero gradients. Use pairwise or sampled softmax losses.
- Pure content-based filtering without collaborative signal — ignores user-item
  interaction patterns that are the primary signal in most recommendation tasks.
- Very high embedding dimensions (> 128) on sparse synthetic interaction matrices —
  the model memorises the training interactions without learning generalisable patterns.
  Use embedding_dim ≤ 64 and strong L2 regularisation.
- Applying dropout directly to user/item ID embeddings — IDs appear rarely; dropout
  makes the embeddings noisy for infrequent users/items. Apply dropout after the
  interaction function, not on raw embeddings.

### Common Failure Modes

- **Interaction matrix sparsity not handled** — A randomly sparse matrix with < 5%
  fill means most user-item pairs have label=0. Training on this directly results
  in the model predicting 0 for everything. Use negative sampling or pairwise losses.
- **Embedding lookup out of range** — User or item IDs must be in range [0, n_users)
  and [0, n_items). On synthetic data, ensure IDs are generated within bounds.
- **Recommendation evaluated incorrectly** — Recommendation metrics (NDCG, Recall@K)
  require ranking, not just binary prediction. Ensure the evaluation returns a ranked
  list of items and computes rank-aware metrics.
- **Cold start not addressed** — A model that only embeds user IDs cannot make
  predictions for new users. For cold-start evaluation, the architecture must have a
  content-based branch.

## Examples

### Good Mutation Proposal
```json
{
  "architecture_name": "ncf_session_transformer_contrastive",
  "mutations": ["free_form", "architecture_crossover"],
  "rationale": "Replace static user embedding with a Transformer over the user's interaction session, add contrastive loss to regularise user representations — tests whether dynamic session modelling combined with representation regularisation outperforms static collaborative filtering on cold-start users.",
  "free_form_description": "SessionTransformerUser: encode the user's last K=20 interacted items as token embeddings using the item embedding table; apply 2-layer Transformer with causal masking; use the final token as the user representation. ContrastiveLoss: for each batch, treat (user, same_user_at_different_session) as positive pairs and (user, different_user) as negative pairs; add cosine contrastive loss with margin=0.5 to the BPR ranking loss."
}
```

### Bad Mutation Proposal
```json
{
  "architecture_name": "ncf_wider",
  "mutations": ["width_change"],
  "rationale": "Wider MLP layers improve interaction modelling."
}
```
This is bad because: `width_change` on NCF is not novel, doesn't address sparsity
or cold-start, and the rationale states a generic belief rather than a hypothesis.
