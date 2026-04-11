---
name: coder-rl
description: Keras/TF implementation patterns for RL domain using tensor_optix. Load when generating code for any RL task (DQN, PPO, SAC, trading, control).
agent: coder
version: 1.0
domains: [rl]
---

# Coder Skill — Reinforcement Learning Domain (tensor_optix)

## Purpose

RL code in this project must use `tensor_optix` components (TFPPOAgent, TFDQNAgent,
ObsNormalizer, compute_gae, etc.) rather than hand-rolled equivalents. The most
dangerous mistake is applying `RewardNormalizer` to a financial/trading reward — the
PnL magnitude IS the signal and must never be scaled or clipped. This skill tells you
exactly which components to use and when.

## Key Principles

- Import from `tensor_optix`, not from a hand-rolled implementation. Use `TFPPOAgent`,
  `TFDQNAgent`, `TFSACAgent`, `compute_gae`, `make_minibatches`, `ObsNormalizer`.
- `ObsNormalizer` is safe and encouraged for ALL RL tasks — always normalize observations.
- `RewardNormalizer` must NEVER be used with PPO (`TFPPOAgent`). PPO reduces variance
  through GAE advantage normalization, not reward normalization. Applying `RewardNormalizer`
  corrupts the reward before GAE runs and provides no benefit. `reward_normalizer=None`
  is always correct for `TFPPOAgent`.
- `RewardNormalizer` may only be used with DQN or SAC on environments where the reward
  scale is arbitrary and carries no semantic meaning, and there is no GAE step. Never use
  it when the reward is a financial signal (PnL, return %, profit) regardless of algorithm.
- The actor must output raw logits (discrete) or `[mean, log_std]` (continuous) — never
  apply softmax inside the model when using `TFPPOAgent`; it applies softmax internally.
- Always call `obs_normalizer.update(obs_batch)` on each collected rollout before
  `obs_normalizer.normalize(obs)` inside `act()`. Never update during inference.

## tensor_optix Component Reference

### ObsNormalizer — always use this
```python
from tensor_optix import ObsNormalizer
obs_norm = ObsNormalizer(obs_shape=(obs_dim,), clip=10.0)
# update after each rollout:
obs_norm.update(np.array(rollout_obs))
# normalize before passing to actor/critic:
obs_in = obs_norm.normalize(obs)
```

### TFPPOAgent (discrete) — standard pattern
```python
from tensor_optix import TFPPOAgent
from tensor_optix.core.types import HyperparamSet

actor = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='tanh', input_shape=(obs_dim,)),
    tf.keras.layers.Dense(256, activation='tanh'),
    tf.keras.layers.Dense(n_actions),  # raw logits — NO softmax
])
critic = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='tanh', input_shape=(obs_dim,)),
    tf.keras.layers.Dense(256, activation='tanh'),
    tf.keras.layers.Dense(1),
])
agent = TFPPOAgent(
    actor=actor,
    critic=critic,
    optimizer=tf.keras.optimizers.Adam(3e-4),
    hyperparams=HyperparamSet(params={
        "clip_ratio": 0.2, "entropy_coef": 0.01, "vf_coef": 0.5,
        "gamma": 0.99, "gae_lambda": 0.95,
        "n_epochs": 10, "minibatch_size": 64, "max_grad_norm": 0.5,
    }, episode_id=0),
    reward_normalizer=None,   # always None for PPO — GAE handles variance reduction
)
```

### compute_gae — use instead of hand-rolling returns
```python
from tensor_optix import compute_gae
advantages, returns = compute_gae(
    rewards, values, dones,
    gamma=0.99, gae_lambda=0.95,
    last_value=0.0,   # 0.0 if episode terminated; critic(next_obs) if truncated
)
```

### PrioritizedReplayBuffer — use for DQN instead of a plain deque
```python
from tensor_optix import PrioritizedReplayBuffer
buffer = PrioritizedReplayBuffer(capacity=50000, obs_dim=obs_dim)
buffer.add(obs, action, reward, next_obs, done)
batch = buffer.sample(batch_size=64)
```

## Trading / Financial RL Pattern

When the task involves trading, portfolio management, or any environment where reward =
PnL, return %, or profit:

```python
# CORRECT
agent = TFPPOAgent(actor=actor, critic=critic, optimizer=opt,
                   hyperparams=hp, reward_normalizer=None)

# WRONG — always wrong for PPO regardless of domain
from tensor_optix import RewardNormalizer
agent = TFPPOAgent(..., reward_normalizer=RewardNormalizer())  # NEVER with PPO

# State/observation features ARE safe to normalize
obs_norm = ObsNormalizer(obs_shape=(feature_dim,))  # observation space — fine
```

The observation space has arbitrary scale and benefits from normalization. The reward
(PnL) has meaningful scale that the critic must learn to interpret as-is.

## Synthetic Data Pattern (trading fallback)

When no live environment is available:
```python
import numpy as np
rng = np.random.default_rng(42)
obs_dim, n_actions = 16, 3  # features, {buy, hold, sell}
# Simulate price returns as the reward signal — do NOT normalize these
log_returns = rng.normal(0.0, 0.01, size=5000).astype(np.float32)
# Simulate observation features (technical indicators, price-derived signals, etc.)
features = rng.standard_normal((5000, obs_dim)).astype(np.float32)
```

## Common Implementation Errors

- **RewardNormalizer on PnL** — divides the financial signal by its own std, collapsing
  a $1000 win and a $10 win to the same normalized value. The critic can no longer
  distinguish profitable from marginal episodes.
- **softmax inside actor model** — `TFPPOAgent` calls `tf.nn.log_softmax` internally on
  the logits. Adding softmax in the model layer causes double-softmax and collapses the
  action distribution to near-uniform immediately.
- **obs_normalizer.update() during act()** — contaminates running stats with deployment
  observations; update only on collected rollouts, never on single inference steps.
- **Skipping `last_value` in compute_gae** — if a rollout window was cut mid-episode
  (truncated, not terminated), passing `last_value=0.0` underestimates future returns
  for the truncation boundary. Pass `critic(final_obs)` instead.
