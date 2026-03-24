---
name: coder-recommendation
description: Keras/TF implementation patterns for recommendation domain architectures. Load when generating code for recommendation tasks.
agent: coder
version: 1.0
domains: [recommendation]
---

# Coder Skill — Recommendation Domain

## Purpose

Recommendation code fails most often due to: embedding lookup out of range, treating
the task as classification over all items (infeasible for large item sets), and not
implementing pairwise/negative-sampling losses. This skill provides correct patterns
for user-item interaction, session modelling, and BPR-style training objectives.

## Key Principles

- Embedding IDs must be in range [0, vocab_size). Generate synthetic IDs carefully.
- Do not use softmax over all items for training — use pairwise (BPR) loss with
  negative sampling, which is faster and more realistic.
- User and item embedding dimensions must be equal for dot-product similarity.
- For session-based models, user_ids are replaced by sequence of item interaction history.

## Synthetic Data Pattern

```python
np.random.seed(42)
n_users, n_items, n_interactions = 500, 200, 5000

# Simulate sparse interaction matrix
user_ids = np.random.randint(0, n_users, n_interactions).astype(np.int32)
item_ids  = np.random.randint(0, n_items, n_interactions).astype(np.int32)
# Positive pairs: (user, item) that interacted
# Negative pairs: random item not in user's history (simplified)
neg_item_ids = np.random.randint(0, n_items, n_interactions).astype(np.int32)
```

## Correct Custom Layer Patterns

### Neural Collaborative Filtering
```python
class NCFModel(tf.keras.Model):
    def __init__(self, n_users, n_items, embed_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.user_emb = tf.keras.layers.Embedding(n_users, embed_dim,
                            embeddings_regularizer=tf.keras.regularizers.L2(1e-4))
        self.item_emb = tf.keras.layers.Embedding(n_items, embed_dim,
                            embeddings_regularizer=tf.keras.regularizers.L2(1e-4))
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1),
        ])

    def call(self, inputs, training=None):
        user_id, item_id = inputs
        u = self.user_emb(user_id)  # (batch, embed_dim)
        v = self.item_emb(item_id)  # (batch, embed_dim)
        # Combine: element-wise product + concat for MLP
        interaction = tf.concat([u, v, u * v], axis=-1)
        return self.mlp(interaction, training=training)  # (batch, 1)
```

### BPR Pairwise Training Loop
```python
def bpr_loss(pos_scores, neg_scores):
    """Bayesian Personalised Ranking loss."""
    return -tf.reduce_mean(tf.math.log(tf.nn.sigmoid(pos_scores - neg_scores) + 1e-8))

model = NCFModel(n_users, n_items, embed_dim=32)
optimizer = tf.keras.optimizers.Adam(1e-3)

@tf.function
def train_step(user_ids, pos_item_ids, neg_item_ids):
    with tf.GradientTape() as tape:
        pos_scores = model([user_ids, pos_item_ids], training=True)
        neg_scores = model([user_ids, neg_item_ids], training=True)
        loss = bpr_loss(pos_scores, neg_scores)
        loss += sum(model.losses)  # L2 regularisation losses
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

for epoch in range(30):
    # Mini-batch training
    idx = np.random.permutation(n_interactions)
    total_loss = 0.0
    for start in range(0, n_interactions, 128):
        batch = idx[start:start+128]
        loss = train_step(
            tf.constant(user_ids[batch]),
            tf.constant(item_ids[batch]),
            tf.constant(neg_item_ids[batch]),
        )
        total_loss += float(loss)
    print(f"Epoch {epoch}: loss={total_loss:.4f}")

# Save for evaluation
model.save('output/model.keras')
```

### Session-Based Transformer
```python
class SessionTransformer(tf.keras.Model):
    def __init__(self, n_items, embed_dim=32, seq_len=10, **kwargs):
        super().__init__(**kwargs)
        self.item_emb = tf.keras.layers.Embedding(n_items + 1, embed_dim, mask_zero=True)
        self.pos_emb  = tf.keras.layers.Embedding(seq_len, embed_dim)
        self.attn     = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=embed_dim // 4
        )
        self.norm     = tf.keras.layers.LayerNormalization()
        self.output_proj = tf.keras.layers.Dense(n_items)

    def call(self, item_seq, training=None):
        # item_seq: (batch, seq_len) item IDs
        positions = tf.range(tf.shape(item_seq)[1])
        x = self.item_emb(item_seq) + self.pos_emb(positions)
        x = self.attn(x, x, use_causal_mask=True, training=training)
        x = self.norm(x)
        return self.output_proj(x[:, -1, :])  # predict next item from last position
```

## Common Implementation Errors

- **Embedding index out of bounds** — `Embedding(n_items, d)(item_id)` fails if any
  `item_id >= n_items`. Always check: `assert item_ids.max() < n_items`.
- **Softmax over all items** — For n_items=1000, `Dense(1000, softmax)` + crossentropy
  with sparse labels produces near-zero gradients for all negative items. Use BPR loss.
- **User/item embedding dims different** — For dot-product similarity: user_emb(u) · item_emb(v)
  requires both to have the same embed_dim. If different, matmul fails.
- **model.fit() with BPR** — BPR requires (user, pos_item, neg_item) triplets, not (X, y) pairs.
  Use a custom training loop as shown above.

## Output Configuration

- Item ranking: final model outputs relevance score per item; at eval, rank all items by score
- Session-based next-item prediction: `Dense(n_items)` logits; at eval, argmax = predicted next item
