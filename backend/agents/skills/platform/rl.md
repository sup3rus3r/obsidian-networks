---
name: platform-rl
description: Guides the platform agent through RESEARCH → PLAN → BUILD for reinforcement learning tasks (DQN, policy gradient, actor-critic). Load when the user describes an RL task or dataset_type indicates trading/sequential decision-making.
agent: platform
version: 1.0
domains: [rl]
---

# Platform Skill — Reinforcement Learning Domain

## Purpose

RL training does not use supervised `model.fit()`. It requires a custom loop that interacts
with an environment, collects experience, and updates a policy or value network using
`tf.GradientTape`. This skill guides the agent in selecting the right RL algorithm, planning
the neural network policy architecture and training loop structure, and generating code that
integrates with gymnasium environments (or synthetic fallback) correctly.

## Key Principles

- NEVER use `model.fit()` for RL — the training loop must use `tf.GradientTape` with environment interaction.
- The policy network (DQN Q-network, actor, or policy) must be saved to `output/model.keras`.
- For DQN: use a replay buffer (deque or np.ndarray) with minimum fill before training starts.
- For policy gradient / actor-critic: compute returns with discount factor γ; normalise returns before computing loss.
- Target network for DQN: hard-copy weights every N steps — do NOT update the target network every step.
- Gymnasium environments: always call `env.reset()` → `env.step(action)` → handle `terminated | truncated`.

## Procedure

### RESEARCH Phase

**CRITICAL: Never ingest paper URLs from your training knowledge. Only ingest URLs returned by arXiv search results. Do not use URLs you already know (e.g. FinRL, DQN papers, classic RL papers) — these are old and defeat the purpose of research.**

1. Run both arXiv searches in parallel using the `search_arxiv` tool — use the results returned, not papers you know:
   - `"deep Q-network DQN replay buffer target network 2024"`
   - `"actor-critic policy gradient advantage PPO reinforcement learning 2025"`
2. Select 3–4 papers from the search results only. For DQN: replay buffer size, target network update frequency. For actor-critic: advantage estimation, clipping.
3. Ingest only the URLs from those search results in parallel.
4. Fetch TF/Keras docs in parallel:
   - `"GradientTape custom training loop reinforcement learning TensorFlow"`
   - `"Dense network policy function approximation RL"`
   - `"gymnasium environment step reset action space observation"`
5. Call `finalize_research()`.

### PLANNING Phase

Query at least 6 times before writing:
- `"DQN replay buffer size minimum fill before training"`
- `"target network hard update frequency DQN training"`
- `"policy gradient advantage estimation normalisation"`
- `"neural network architecture RL policy function approximation"`
- `"exploration epsilon greedy decay schedule DQN"`
- `"reward clipping normalisation deep reinforcement learning"`

The plan MUST include:
- RL algorithm type: DQN, REINFORCE, A2C, or PPO — with justification from research
- State space and action space sizes (from the environment or user description)
- Policy/Q-network architecture: input → Dense layers → output (Q-values or action probabilities)
- Training loop description: steps per episode, replay buffer design (DQN), return computation (PG)
- Exploration strategy: ε-greedy with decay schedule (DQN), stochastic policy (PG)
- Target network update frequency (DQN only)
- Episode count and convergence criterion

### BUILD Phase

Before writing any code, call `query_research` for each key implementation detail (layer sizes, learning rates, loss functions, hyperparameters) to retrieve exact values from the ingested papers. Every value in the script must match the approved plan — `create_notebook` runs an automated alignment check and will reject mismatches.

Follow the standard BUILD SEQUENCE. Key RL-specific steps:

1. In `edit_script`, the training loop structure must be:
   ```python
   for episode in range(num_episodes):
       obs, _ = env.reset()
       done = False
       while not done:
           action = select_action(obs, epsilon)
           next_obs, reward, terminated, truncated, _ = env.step(action)
           done = terminated or truncated
           store_transition(obs, action, reward, next_obs, done)
           if len(replay_buffer) >= min_buffer_size:
               train_step()
           obs = next_obs
       epsilon = max(epsilon_min, epsilon * epsilon_decay)
   ```
2. For DQN: separate `q_network` and `target_network`; hard-copy weights every `target_update_freq` steps.
3. For REINFORCE: store episode trajectory, compute discounted returns, compute policy gradient loss with `tf.GradientTape`.
4. Gymnasium environment: try `LunarLander-v3`, fallback to `CartPole-v1` if unavailable; use synthetic data as last resort if gymnasium is not installed.
5. Save policy network: `q_network.save('output/model.keras')` or `actor.save('output/model.keras')`.

## Domain-Specific Guidance

### Algorithm Selection Guide

- **DQN** — discrete action space (CartPole, LunarLander), stable but requires replay buffer and target network
- **REINFORCE** — discrete or continuous, no replay buffer, high variance — use only for simple environments
- **A2C/PPO** — continuous or discrete, more stable than REINFORCE; use when DQN is not suitable

### Policy Network Architecture

```
For discrete action space (DQN Q-network or policy):
  Input(state_dim,) → Dense(128, relu) → Dense(128, relu) → Dense(n_actions, linear)  [Q-values]
  or
  Input(state_dim,) → Dense(128, relu) → Dense(n_actions, softmax)                    [probabilities]

For continuous action space (actor):
  Input(state_dim,) → Dense(256, relu) → Dense(256, relu)
  → Dense(n_actions, tanh) [mean], Dense(n_actions, softplus) [std]  [Gaussian policy]
```

### DQN Hyperparameter Defaults

| Parameter | Default | Notes |
|---|---|---|
| Replay buffer size | 10000–100000 | Fill at least 1000 before first train step |
| Batch size | 32–64 | Sample randomly from buffer |
| Target update freq | every 100–500 steps | Hard copy, not soft update |
| ε start / end | 1.0 → 0.01 | Decay over first 50% of training steps |
| γ (discount) | 0.99 | Lower (0.95) for short-horizon tasks |
| Learning rate | 1e-3 to 5e-4 | Adam optimizer |

### Known Dead Ends

- `model.fit()` for any RL algorithm — cannot interleave environment interaction with gradient updates.
- Updating the target network every step — target moves too fast; loss oscillates. Hard-copy every 100–500 steps.
- No minimum buffer fill before training — early training on tiny buffers is noisy and causes divergence.
- Very large neural networks (> 3 layers, > 512 units) for simple environments — overfits the replay buffer; use 2 layers of 128–256 units.
- Missing `terminated | truncated` check — the `done` flag requires both; ignoring truncation causes incorrect value bootstrapping.

### Common Failure Modes

- **Q-values explode** — no reward clipping or normalisation. Clip rewards to [-1, 1] for Atari-style tasks; normalise returns to zero-mean for policy gradient.
- **Policy gradient variance too high** — no baseline. For REINFORCE, subtract a running mean of returns as a baseline.
- **gymnasium environment not installed** — wrap import in try/except; fall back to `CartPole-v1` (ships with gym) or generate synthetic state transitions.
- **Actor-critic critic loss dominates** — use separate optimizers and loss weights; scale critic loss by 0.5 relative to actor loss.

## Examples

### Good Plan Training Loop Description

```
## 5. Training Strategy
Algorithm: DQN with experience replay and target network  [Mnih et al. 2015, https://...]
Q-network: Input(8,) → Dense(128,relu) → Dense(128,relu) → Dense(4,linear)  [4 discrete actions]
Target network: hard copy every 200 steps
Replay buffer: deque(maxlen=50000), min fill=1000 before first training step
ε: 1.0 → 0.01 over 10000 steps (linear decay)
Optimizer: Adam(lr=5e-4), loss: MSE on TD-error
Episodes: 500. Convergence: mean reward > 200 over 10 episodes.
Save q_network to output/model.keras.
```

### Bad Plan Training Loop Description

```
## 5. Training Strategy
Train the model using model.fit() on state-action pairs for 100 epochs.
```
This is bad: RL does not have a fixed dataset; `model.fit()` cannot interact with the environment.
