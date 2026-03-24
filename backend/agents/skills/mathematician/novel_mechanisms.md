---
name: mathematician-novel-mechanisms
description: How to derive genuinely novel mathematical mechanisms from paper content. The primary skill for driving architecture novelty. Load for every mechanism derivation call.
agent: mathematician
version: 1.0
domains: [all]
---

# Mathematician Skill — Novel Mechanism Derivation

## Purpose

The Mathematician reads paper content from the vector store and derives mathematical
mechanisms that the papers did not explicitly explore. This is the most critical step
for driving genuine architecture novelty. The failure mode is producing mechanisms that
are essentially renamed standard operations (renamed attention, renamed residual, renamed
normalisation). This skill teaches how to read mathematical content and project it into
unexplored territory.

## Key Principles

- You are not summarising what the paper did. You are finding what the paper implies but
  did not try. Every paper contains mathematical principles that were applied in one specific
  way. Your job is to find the adjacent applications that follow from the same principle.
- A mechanism is novel if a researcher who has read all the cited papers would not have
  seen this exact combination before. A mechanism that sounds novel but is equivalent to
  a standard operation under different notation is NOT novel.
- Every mechanism must have a SymPy-expressible core computation. If you cannot write the
  essential operation in SymPy, the mechanism is too vague to be implemented.
- Produce mechanisms that cross paper boundaries. The most novel mechanisms emerge from
  combining a principle from paper A with a structure from paper B in a way neither paper
  considered.

## The Three-Step Derivation Process

### Step 1: Extract the Mathematical Core

For each paper chunk in the vector store content, identify:
- The **primary equation** (the one that defines the paper's contribution)
- The **constraint assumptions** (what the authors held fixed while proposing this)
- The **hidden variables** (what the equation touches that the paper didn't vary)

Example: A paper on sparse attention defines attention score `a_ij = softmax(q_i · k_j / sqrt(d))` but only applies sparsity via top-k masking. The hidden variable is *how* sparsity is applied — top-k is one choice among many (learned gates, distance thresholds, structured patterns).

### Step 2: Identify the Unexplored Space

Ask for each paper: "What did the authors explicitly hold constant that could have been varied?"

Common unexplored dimensions:
- **Aggregation function** — Papers that use mean aggregation; what about learned weighted aggregation?
- **Normalisation position** — Pre-norm vs post-norm vs adaptive norm based on input statistics
- **Gate parameterisation** — Papers using sigmoid gates; what about softmax-normalised multi-way gates?
- **Interaction order** — Papers computing pairwise interactions; what about explicit third-order terms?
- **Distance metric** — Papers using Euclidean distance; what about hyperbolic or cosine distance?
- **Time scale** — Papers operating at one temporal scale; what about simultaneously multi-scale?
- **Mixing across dimensions** — Papers mixing along the sequence dimension; what about mixing along the feature dimension?

### Step 3: Formulate the Novel Mechanism

Combine an unexplored dimension from Step 2 with a constraint from a *different* paper.
The result is a mechanism that neither paper considered.

**Format for each derived mechanism:**
- `name`: short snake_case, descriptive of what it does (not of where it came from)
- `description`: one sentence stating the novel hypothesis — what property does this mechanism test?
- `sympy_expression`: the core computation in SymPy syntax, using symbolic variable names

## SymPy Expression Standards

The SymPy expression must represent the *essential computation* of the mechanism —
not the full forward pass. It should be a formula involving named symbols.

### Valid SymPy patterns:

```python
# Gating mechanism
sigmoid(W_g * x) * (W_v * x)

# Attention scoring
softmax(Q * K.T / sqrt(d)) * V

# Normalised interaction
(x - mean(x, axis=1)) / (std(x, axis=1) + eps) * gamma + beta

# Hyperbolic projection
tanh(norm(x)) * x / norm(x)

# Multi-scale convolution fusion
alpha * conv(x, k_small) + (1 - alpha) * conv(x, k_large)  # where alpha = sigmoid(W * x)
```

### Invalid SymPy patterns (too vague or wrong):

```python
# Too vague — not a formula
attention(x, context)

# Undefined functions that are not in sympy
LeakyReLU(x)  # write max(alpha * x, x) instead

# Tensor operations not expressible symbolically
matmul(Q, K.T)  # write Q * K.T instead (understood as matrix product)
```

When the mechanism involves a tensor operation, express the scalar formula for a single
element of the output and note the shape in the description.

## Novelty Classification

Use this classification to verify each mechanism before returning:

### Tier 1 — Genuinely Novel (target this)
- Combines mathematical principles from two different papers in a way neither proposed
- Changes a structural assumption that every existing method takes for granted
- Introduces a computation that has a clear hypothesis but no established name
- Example: "Frequency-domain gating: apply FFT to hidden states, gate by learned frequency
  mask, IFFT back — tests whether frequency-selective filtering outperforms spatial gating"

### Tier 2 — Variant (acceptable, note it)
- A known mechanism applied in a new context or domain
- A known mechanism with one parameter changed to be data-dependent
- Example: "Adaptive layer normalisation where gamma/beta are generated from input
  statistics rather than learned as fixed parameters"

### Tier 3 — Derivative (reject and retry)
- A standard mechanism with a different activation function
- A standard mechanism described with unfamiliar notation
- Any mechanism equivalent to: vanilla attention, residual connection, batch/layer norm,
  standard MLP, standard convolution
- Example: "Attention with tanh instead of softmax" — this is just a different activation,
  not a novel mechanism

If a mechanism falls into Tier 3, do not return it. Derive a replacement.

## Common Failure Modes to Avoid

- **Renaming standard operations**: "Neural Interaction Layer" that is just a Dense + ReLU.
  Always check: could this be described as a standard layer? If yes, it is Tier 3.
- **Vague descriptions without a mathematical core**: "A layer that captures complex patterns".
  Every mechanism must have a specific formula.
- **SymPy expressions that parse but don't match the description**: The expression must
  implement what the description claims. If the description says "hyperbolic projection"
  but the expression is just tanh(x), these are inconsistent.
- **Mechanisms that cannot be implemented in TensorFlow**: Every mechanism must be
  implementable as `tf.keras.layers.Layer.call()` using numpy and tensorflow ops only.
  No scipy, no networkx, no torch.

## Novelty Feedback Integration

If `context["novelty_feedback"]` is present, read it before deriving mechanisms.
This feedback from the Critic explains why previous mechanisms scored low on novelty.
Explicitly avoid the directions flagged in the feedback and target the suggestions
provided. The feedback format is:

```
NOVELTY FEEDBACK (generation N):
- Candidate <name> scored low on novelty. Reason: <why>.
- Mutations used (<list>) too similar to existing candidates.
- Suggested directions:
  1. <direction 1>
  2. <direction 2>
```

Use this to steer derivation away from recently explored territory.

## Examples

### Good Mechanism (Tier 1)
```json
{
  "name": "spectral_temporal_gate",
  "description": "Apply FFT along the time axis of LSTM hidden states, learn a complex-valued frequency mask, apply, then IFFT back — tests whether frequency-selective temporal filtering outperforms standard attention-weighted aggregation for periodic signal modelling.",
  "sympy_expression": "ifft(mask_r * fft(h) + I * mask_i * fft(h))"
}
```

### Bad Mechanism (Tier 3 — reject)
```json
{
  "name": "enhanced_attention",
  "description": "Attention mechanism with improved scaling for better stability.",
  "sympy_expression": "softmax(Q * K.T / sqrt(2 * d)) * V"
}
```
This is just scaled dot-product attention with a different constant. It is Tier 3.
Reject and derive something genuinely novel instead.
