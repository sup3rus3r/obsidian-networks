"""
Reinforcement Learning domain handler — DQN, Actor-Critic, PPO policy networks.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from .base_domain import BaseDomain, TF_CODE_SYSTEM, MUTATION_SYSTEM, MECHANISM_SYSTEM


class RLDomain(BaseDomain):
    name = "rl"
    supported_architectures = ["dqn", "actor_critic", "ppo"]
    mutation_operators = [
        "layer_insertion", "activation_change", "depth_change",
        "width_change", "skip_connection_add", "free_form", "architecture_crossover",
    ]
    metrics = ["loss", "episode_reward", "policy_loss", "value_loss", "memory_mb", "training_time_s"]

    base_templates = {
        "dqn": {
            "type"         : "dqn",
            "input_shape"  : [8],
            "n_actions"    : 4,
            "normalization": "layer_norm",
            "activation"   : "relu",
            "layers": [
                {"type": "dense", "units": 128, "activation": "relu"},
                {"type": "dense", "units": 128, "activation": "relu"},
                {"type": "dense", "units": 64,  "activation": "relu"},
            ],
            "dueling": False,
        },
        "actor_critic": {
            "type"       : "actor_critic",
            "input_shape": [8],
            "n_actions"  : 4,
            "activation" : "tanh",
            "shared_layers": [
                {"type": "dense", "units": 128, "activation": "tanh"},
                {"type": "dense", "units": 128, "activation": "tanh"},
            ],
            "actor_head" : [{"type": "dense", "units": 4,  "activation": "softmax"}],
            "critic_head": [{"type": "dense", "units": 1,  "activation": "linear"}],
        },
        "ppo": {
            "type"       : "ppo",
            "input_shape": [8],
            "n_actions"  : 4,
            "activation" : "tanh",
            "clip_epsilon": 0.2,
            "layers": [
                {"type": "dense", "units": 256, "activation": "tanh"},
                {"type": "dense", "units": 256, "activation": "tanh"},
            ],
            "actor_head" : [{"type": "dense", "units": 4,  "activation": "softmax"}],
            "critic_head": [{"type": "dense", "units": 1,  "activation": "linear"}],
        },
    }

    async def generate_mechanism(self, research_insights: str, llm_caller: Callable) -> list[dict]:
        system = MECHANISM_SYSTEM + "\nDomain: reinforcement learning (policy networks, value functions, actor-critic methods, exploration strategies)."
        prompt = f"Research insights:\n{research_insights}\n\nDerive 3 novel mechanisms for RL policy/value networks. JSON array:"
        raw = await llm_caller(prompt, system=system, force_claude=True, max_tokens=1200)
        try:
            start = raw.find("["); end = raw.rfind("]") + 1
            return json.loads(raw[start:end])
        except Exception:
            return [{"name": "intrinsic_curiosity_gate", "description": "Gated curiosity signal modulating exploration bonus", "sympy_expression": "r_i = eta * ||phi(s_t+1) - f(phi(s_t), a_t)||^2"}]

    async def propose_mutations(self, base_arch: str, mechanisms: list[dict], llm_caller: Callable, failed_patterns: list[dict] | None = None, **kwargs) -> list[dict]:
        template         = self.get_base_template(base_arch)
        task_description = kwargs.get("task_description", "")
        explored_summary = kwargs.get("explored_summary")

        system      = MUTATION_SYSTEM + f"\nDomain: reinforcement learning. Available operators: {self.mutation_operators}."
        failure_ctx = self._format_failure_context(failed_patterns)
        goal_ctx    = f"Research goal: {task_description}" if task_description else ""
        explored_ctx = (
            f"Already-explored architecture space (avoid these regions — novelty score rewards distance from them):\n{explored_summary}"
            if explored_summary else ""
        )

        prompt = (
            f"Base architecture:\n{json.dumps(template, indent=2)}"
            f"\n\nMechanisms to implement (use these as the PRIMARY inspiration):\n{json.dumps(mechanisms, indent=2)}"
            + (f"\n\n{goal_ctx}"      if goal_ctx      else "")
            + (f"\n\n{failure_ctx}"   if failure_ctx   else "")
            + (f"\n\n{explored_ctx}"  if explored_ctx  else "")
            + "\n\nPropose 3 mutations. Strongly prefer 'free_form' or 'architecture_crossover' "
              "when the mechanisms suggest ideas beyond standard operators. JSON array:"
        )
        raw = await llm_caller(prompt, system=system, force_claude=True, max_tokens=1800)
        try:
            start = raw.find("["); end = raw.rfind("]") + 1
            proposals = json.loads(raw[start:end])
        except Exception:
            proposals = [{"architecture_name": f"{base_arch}_mutant", "mutations": ["free_form"], "rationale": "Default"}]

        from agents.mutations import apply_mutations

        results = []
        for p in proposals:
            mutation_list = p.get("mutations", [])
            spec = apply_mutations(template, mutation_list)
            if "free_form" in mutation_list:
                spec["free_form_description"] = p.get("free_form_description", p.get("rationale", ""))
                spec["mechanism_names"] = [m.get("name", "") for m in mechanisms]
            results.append({
                "architecture_name": p.get("architecture_name", f"{base_arch}_mutant"),
                "base_template"    : base_arch,
                "mutations"        : mutation_list,
                "spec"             : spec,
                "rationale"        : p.get("rationale", ""),
            })
        return results

    async def generate_code(self, arch_spec: dict, llm_caller: Callable, mechanisms: list[dict] | None = None, rationale: str | None = None) -> str:
        arch_type = arch_spec.get("type", "dqn")
        system = (
            TF_CODE_SYSTEM +
            f"\nDOMAIN: Reinforcement learning ({arch_type.upper()})."
            "\nENVIRONMENT: Use gymnasium (import gymnasium as gym; env = gym.make('LunarLander-v3')). "
            "If gymnasium is unavailable fall back to a synthetic CartPole-like loop: "
            "obs_dim=8, n_actions=4, generate random transitions with np.random."
            "\nTRAINING: Custom loop — NO model.fit(). Collect episodes, compute returns, update weights with tf.GradientTape."
            "\nFor DQN: experience replay buffer, epsilon-greedy exploration, target network updated every 10 episodes."
            "\nFor Actor-Critic/PPO: compute advantage estimates (GAE or n-step returns), separate policy/value losses."
            "\nSave policy network to output/model.keras. EPISODES: 50 (short for evaluation). "
            "Print mean episode reward every 10 episodes."
        )

        prompt = self._build_code_prompt(arch_spec, mechanisms, rationale)
        code = await llm_caller(prompt, system=system, force_claude=True, max_tokens=4000)
        if "```python" in code: code = code.split("```python")[1].split("```")[0]
        elif "```" in code:     code = code.split("```")[1].split("```")[0]
        return code.strip()

    def generate_synthetic_data(self, size: int = 1000, params: dict | None = None) -> Any:
        import numpy as np
        p = params or {}
        rng = np.random.default_rng(p.get("seed", 42))
        obs_dim  = p.get("obs_dim", 8)
        n_actions = p.get("n_actions", 4)
        states   = rng.standard_normal((size, obs_dim)).astype("float32")
        actions  = rng.integers(0, n_actions, size=size).astype("int32")
        rewards  = rng.standard_normal(size).astype("float32")
        return {"states": states, "actions": actions, "rewards": rewards}

    async def evaluate(self, checkpoint_path: str, test_data: Any) -> dict:
        try:
            import tensorflow as tf
            import numpy as np
            model   = tf.keras.models.load_model(checkpoint_path)
            states  = test_data.get("states") if isinstance(test_data, dict) else None
            if states is None:
                return {"loss": 0.0, "episode_reward": 0.0}
            preds   = model.predict(states[:100], verbose=0)
            # Proxy metric: entropy of action distribution (higher = more exploratory)
            probs   = tf.nn.softmax(preds, axis=-1).numpy() if preds.ndim == 2 else preds
            entropy = float(-np.mean(np.sum(probs * np.log(probs + 1e-8), axis=-1)))
            return {"loss": 0.0, "episode_reward": entropy}
        except Exception as e:
            return {"loss": 999.0, "episode_reward": 0.0, "error": str(e)}
