---
name: coder-graph
description: Keras/TF implementation patterns for graph domain architectures. Load when generating code for graph/GNN tasks.
agent: coder
version: 1.0
domains: [graph]
---

# Coder Skill — Graph Domain

## Purpose

Graph code fails most often due to: adjacency matrix not normalised, sparse matrix
operations not supported in TF eager mode, and trying to use graph libraries
(networkx, torch_geometric) that are not in the allowed import list. All graph
operations must be implemented with numpy and tensorflow only, using dense adjacency
matrices. This skill provides self-contained TF/numpy patterns for GNN layers.

## Key Principles

- Only numpy and tensorflow are available. No networkx, no scipy.sparse, no DGL,
  no PyG. All graph operations must use dense matrix multiplication.
- The adjacency matrix must be normalised: A_hat = D^{-1/2} A D^{-1/2} where D is
  the diagonal degree matrix. Compute this in numpy before training.
- Input shapes: node features X = (N, F), adjacency A = (N, N), labels y = (N,).
- GNN layers do not use the Keras standard fit() API easily — use a custom training loop
  or wrap the graph computation in a tf.keras.Model with custom call().

## Synthetic Data Pattern

```python
np.random.seed(42)
N, F, n_classes = 200, 16, 5   # nodes, features, classes

# Random graph (Erdos-Renyi style)
A = (np.random.rand(N, N) > 0.85).astype(np.float32)
A = np.maximum(A, A.T)         # make symmetric
np.fill_diagonal(A, 1)         # add self-loops

# Normalise adjacency: D^{-1/2} A D^{-1/2}
degree = A.sum(axis=1)
D_inv_sqrt = np.diag(degree ** -0.5)
A_norm = D_inv_sqrt @ A @ D_inv_sqrt

X = np.random.randn(N, F).astype(np.float32)
y = np.random.randint(0, n_classes, N).astype(np.int32)

A_norm = tf.constant(A_norm, dtype=tf.float32)
X = tf.constant(X, dtype=tf.float32)
```

## Correct Custom Layer Patterns

### GCN Layer
```python
class GCNLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.W = tf.keras.layers.Dense(units, use_bias=False)
        self.activation = tf.keras.activations.get(activation)
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs, training=None):
        x, a_norm = inputs  # x: (N, F), a_norm: (N, N)
        # Message passing: aggregate from neighbours via normalised adjacency
        aggregated = tf.matmul(a_norm, x)          # (N, F)
        transformed = self.W(aggregated)            # (N, units)
        normed = self.norm(transformed, training=training)
        return self.activation(normed)
```

### Graph Attention Layer
```python
class GATLayer(tf.keras.layers.Layer):
    def __init__(self, units, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = units // num_heads
        self.W = tf.keras.layers.Dense(units, use_bias=False)
        self.attn_src = tf.keras.layers.Dense(num_heads, use_bias=False)
        self.attn_dst = tf.keras.layers.Dense(num_heads, use_bias=False)

    def call(self, inputs, training=None):
        x, adj = inputs  # x: (N, F), adj: (N, N) binary
        h = self.W(x)    # (N, units)
        # Compute attention logits
        alpha_src = self.attn_src(h)  # (N, heads)
        alpha_dst = self.attn_dst(h)  # (N, heads)
        # Broadcast: e_ij = alpha_i + alpha_j
        e = alpha_src[:, tf.newaxis, :] + alpha_dst[tf.newaxis, :, :]  # (N, N, heads)
        e = tf.nn.leaky_relu(e)
        # Mask non-edges
        mask = (1.0 - adj[:, :, tf.newaxis]) * -1e9  # (N, N, 1)
        e = e + mask
        attn = tf.nn.softmax(e, axis=1)              # (N, N, heads)
        # Aggregate: for each node, weighted sum of neighbour features
        h_exp = tf.tile(h[tf.newaxis], [tf.shape(h)[0], 1, 1])  # (N, N, units)
        # Simplified: mean over heads after softmax-weighted aggregation
        out = tf.einsum('ijk,jl->il', attn[..., 0:1], h) / self.num_heads
        return tf.nn.elu(out)
```

## Custom Training Loop for GNN

GNNs work best with a custom training loop because the full graph must be passed as one sample:

```python
optimizer = tf.keras.optimizers.Adam(1e-3)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

@tf.function
def train_step(x, a_norm, y):
    with tf.GradientTape() as tape:
        logits = model([x, a_norm], training=True)
        loss = loss_fn(y, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

for epoch in range(50):
    loss = train_step(X, A_norm, y)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss={loss:.4f}")
```

## Common Implementation Errors

- **Unnormalised adjacency** — Raw A has large eigenvalues for high-degree nodes, causing
  gradient explosion. Always normalise: `A_hat = D^{-1/2} A D^{-1/2}`.
- **Using scipy or networkx** — Not in the allowed import list. All graph ops must use numpy/tf.
- **Calling model.fit() with graph data** — Standard Keras fit() expects batched data.
  For full-graph training (transductive), use a custom training loop.
- **Shape mismatch in multi-layer GNN** — Layer 1 output must match layer 2 input dim.
  GCN(16, units=32) → GCN(32, units=64): output of layer 1 is (N, 32) which is correct input for layer 2.

## Output Configuration

- Node classification: final GNN layer output (N, units), then `Dense(n_classes)` → `SparseCategoricalCrossentropy`
- Graph classification (single label per graph): `tf.reduce_mean(node_embeddings, axis=0)` then Dense
