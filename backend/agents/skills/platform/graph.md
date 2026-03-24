---
name: platform-graph
description: Guides the platform agent through RESEARCH → PLAN → BUILD for graph neural network tasks (node classification, link prediction with GCN/GAT). Load when the user describes a graph or network-based ML task.
agent: platform
version: 1.0
domains: [graph]
---

# Platform Skill — Graph Neural Network Domain

## Purpose

TensorFlow/Keras has no native GNN layers. Graph neural networks must be implemented from
scratch using `tf.matmul` for message passing (adjacency × node features). This skill guides
the agent in researching GNN architectures with concrete layer formulas, planning the
adjacency normalization and message-passing depth, and generating code that implements
GCN-style convolution using dense matrix operations compatible with the platform's TF environment.

## Key Principles

- There are NO pre-built GNN layers in TF/Keras — implement message passing with `tf.matmul(adj_norm, node_features)`.
- Adjacency normalization: `A_hat = D^(-1/2) (A + I) D^(-1/2)` — must be pre-computed before training.
- `node_features` shape is `(n_nodes, n_features)` — there is no batch dimension in standard transductive GNNs.
- For inductive (mini-batch) GNN: use GraphSAGE sampling, not full-graph message passing.
- Skip connections improve GNNs just as in CNNs — add them for depth ≥ 3 GCN layers.
- Output: `Dense(n_classes, softmax)` for node classification; `sigmoid` for link prediction.

## Procedure

### RESEARCH Phase

1. Run both arXiv searches in parallel:
   - `"graph convolutional network node classification GCN GAT message passing 2024"`
   - `"graph neural network spectral spatial aggregation 2023 2024"`
2. Select 3–4 papers. Focus on: adjacency normalization, layer formulas, depth, dropout placement.
3. Ingest in parallel.
4. Fetch TF/Keras docs in parallel:
   - `"tf.matmul sparse dense matrix multiplication TensorFlow"`
   - `"GradientTape custom layer graph neural network"`
   - `"sparse_categorical_crossentropy node classification TensorFlow"`
5. Call `finalize_research()`.

### PLANNING Phase

Query at least 6 times before writing:
- `"GCN layer formula adjacency normalization Kipf 2017"`
- `"GAT graph attention network multi-head message passing"`
- `"GNN depth over-smoothing node classification"`
- `"dropout regularization graph neural network training"`
- `"adjacency matrix normalization symmetric random walk"`
- `"mini-batch graph training GraphSAGE neighborhood sampling"`

The plan MUST include:
- Graph statistics: n_nodes, n_features, n_classes (from user or synthetic defaults)
- Adjacency normalization formula: `D^(-1/2)(A+I)D^(-1/2)` or `D^(-1)(A+I)` (asymmetric)
- Number of GCN/GAT layers (2–4 max — deeper causes over-smoothing)
- Hidden dimension per layer with source justification
- Whether to use full-graph transductive or mini-batch inductive (GraphSAGE) training
- Message-passing formula written out explicitly

### BUILD Phase

Follow the standard BUILD SEQUENCE. Key graph-specific steps:

1. In `edit_script`, implement GCN message passing:
   ```python
   def gcn_layer(adj_norm, features, W):
       # adj_norm: (N, N), features: (N, F), W: (F, F_out)
       return tf.nn.relu(tf.matmul(adj_norm, tf.matmul(features, W)))
   ```
2. Pre-compute normalised adjacency BEFORE the training loop:
   ```python
   A_hat = A + np.eye(N)                  # add self-loops
   D = np.diag(A_hat.sum(axis=1))
   D_inv_sqrt = np.diag(1.0 / np.sqrt(D.diagonal()))
   adj_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
   adj_norm = tf.cast(adj_norm, tf.float32)
   ```
3. Synthetic data (description mode): `node_features = np.random.randn(200, 16).astype(np.float32); adj = (np.random.rand(200,200) > 0.95).astype(np.float32); labels = np.random.randint(0, 5, (200,))`
4. Use `tf.GradientTape` for the training loop — there is no standard `model.fit()` for transductive GNNs.
5. Save the model (weights dict) or a Keras model wrapper to `output/model.keras`.

## Domain-Specific Guidance

### GCN Layer Formula (Kipf & Welling 2017)

```
H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))
where: Ã = A + I  (add self-loops)
       D̃ = degree matrix of Ã
       H^(0) = node feature matrix X
       W^(l) = trainable weight matrix
```

### Architecture Depth Guidance

| Depth (GCN layers) | Receptive field | Notes |
|---|---|---|
| 2 | 2-hop neighbourhood | Sweet spot for most node classification |
| 3 | 3-hop neighbourhood | Over-smoothing begins for dense graphs |
| 4+ | 4-hop | High risk of over-smoothing; use residual connections |

### Known Dead Ends

- Native `Conv2D` or `Conv1D` on node features — these ignore the graph structure entirely.
- Raw (unnormalised) adjacency in message passing — exploding/vanishing messages without `D^(-1/2) A D^(-1/2)`.
- More than 4 GCN layers without residual connections — node representations converge to the same vector (over-smoothing).
- Full-graph training on graphs with > 10000 nodes — adjacency matrix is too large for memory; use GraphSAGE mini-batches.
- Using a separate `model.fit()` call — the adjacency tensor must be passed at each step, which requires a custom loop or `model(adj, X)` pattern.

### Common Failure Modes

- **Shape error in tf.matmul** — `adj_norm` must be `(N, N)` float32 and `H` must be `(N, F)` float32. Check dtypes and shapes before the training loop.
- **Over-smoothing** — loss drops but node accuracy plateaus. Add dropout (0.3–0.5) between GCN layers and reduce depth to 2.
- **Adjacency not normalised** — message aggregation explodes or is dominated by high-degree nodes. Always compute `D^(-1/2) Ã D^(-1/2)`.
- **Labels shape mismatch** — `labels` should be `(n_nodes,)` integers for `sparse_categorical_crossentropy`. Verify shape before training.

## Examples

### Good Plan Architecture Section

```
### Selected Model: 2-Layer GCN (Kipf & Welling 2017)
Graph: 200 nodes, 16 features, 5 classes  [synthetic; Cora-style splits]
Adjacency: D^(-1/2)(A+I)D^(-1/2) pre-computed, shape (200,200)  [Kipf 2017, https://...]
Layer 1: tf.matmul(adj_norm, tf.matmul(X, W1)) → ReLU → Dropout(0.5)   [64 hidden units]
Layer 2: tf.matmul(adj_norm, tf.matmul(h1, W2)) → Softmax(5)            [output layer]
Loss: sparse_categorical_crossentropy (masked to training nodes)
Optimizer: Adam(lr=1e-2). Epochs: 200.
```

### Bad Plan Architecture Section

```
### Selected Model: GNN
Some graph layers for node classification.
```
This is bad: no adjacency formula, no layer formula, no hidden dimensions, no message-passing description, no citations.
