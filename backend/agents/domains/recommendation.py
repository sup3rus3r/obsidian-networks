"""
Recommendation domain handler — Collaborative Filtering, Attention-based recommenders.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from .base_domain import BaseDomain


class RecommendationDomain(BaseDomain):
    name = "recommendation"
    supported_architectures = ["embedding_cf", "attention_rec", "ncf"]
    mutation_operators = [
        "layer_insertion", "attention_variant", "normalization_change",
        "activation_change", "depth_change", "width_change",
    ]
    metrics = ["loss", "rmse", "ndcg", "hit_rate", "memory_mb", "inference_time_ms", "training_time_s"]

    base_templates = {
        "embedding_cf": {
            "type"       : "embedding_cf",
            "n_users"    : 500,
            "n_items"    : 1000,
            "embed_dim"  : 32,
            "normalization": "batch_norm",
            "activation" : "relu",
            "layers": [
                {"type": "user_embedding", "n_users": 500,  "embed_dim": 32},
                {"type": "item_embedding", "n_items": 1000, "embed_dim": 32},
                {"type": "dot_product"},
                {"type": "dense", "units": 1, "activation": "sigmoid"},
            ],
        },
        "attention_rec": {
            "type"       : "attention_rec",
            "n_users"    : 500,
            "n_items"    : 1000,
            "embed_dim"  : 64,
            "normalization": "layer_norm",
            "activation" : "relu",
            "attention"  : {"type": "multi_head", "num_heads": 4, "key_dim": 16},
            "layers": [
                {"type": "user_embedding", "n_users": 500,  "embed_dim": 64},
                {"type": "item_embedding", "n_items": 1000, "embed_dim": 64},
                {"type": "cross_attention", "num_heads": 4, "key_dim": 16},
                {"type": "dense", "units": 32, "activation": "relu"},
                {"type": "dense", "units": 1,  "activation": "sigmoid"},
            ],
        },
    }

    async def generate_mechanism(self, research_insights: str, llm_caller: Callable) -> list[dict]:
        prompt = f"""
You are a recommendation systems researcher. Based on:

{research_insights}

Derive 3 novel mechanisms for recommendation models.
JSON array: [{{"name": str, "description": str, "sympy_expression": str}}]
"""
        raw = await llm_caller(prompt, force_claude=True, max_tokens=1200)
        try:
            start = raw.find("["); end = raw.rfind("]") + 1
            return json.loads(raw[start:end])
        except Exception:
            return [{"name": "popularity_debiasing", "description": "Debias recommendations from popularity", "sympy_expression": "score(u,i) - lambda * log(freq(i))"}]

    async def propose_mutations(self, base_arch: str, mechanisms: list[dict], llm_caller: Callable) -> list[dict]:
        template = self.get_base_template(base_arch)
        prompt   = f"""
Recommendation architecture: {json.dumps(template, indent=2)}
Mechanisms: {json.dumps(mechanisms, indent=2)}
Propose 3 mutations. Operators: {self.mutation_operators}
JSON: [{{"architecture_name": str, "mutations": [str], "rationale": str}}]
"""
        raw = await llm_caller(prompt, force_claude=True, max_tokens=1200)
        try:
            start = raw.find("["); end = raw.rfind("]") + 1
            proposals = json.loads(raw[start:end])
        except Exception:
            proposals = [{"architecture_name": f"{base_arch}_mutant", "mutations": ["attention_variant"], "rationale": "Default"}]

        from agents.mutations import apply_mutations
        return [{
            "architecture_name": p.get("architecture_name", f"{base_arch}_mutant"),
            "base_template"    : base_arch,
            "mutations"        : p.get("mutations", []),
            "spec"             : apply_mutations(template, p.get("mutations", [])),
            "rationale"        : p.get("rationale", ""),
        } for p in proposals]

    async def generate_code(self, arch_spec: dict, llm_caller: Callable) -> str:
        prompt = f"""
Write a complete TensorFlow/Keras training script for this recommendation architecture:

{json.dumps(arch_spec, indent=2)}

Requirements:
- n_users=500, n_items=1000, embed_dim=32
- Generate sparse ratings: user_ids = np.random.randint(0, 500, 10000); item_ids = np.random.randint(0, 1000, 10000); ratings = np.random.uniform(1, 5, 10000).astype(np.float32)
- Normalize ratings to [0,1]: ratings = (ratings - 1) / 4
- Model: user Embedding + item Embedding → Dot → Dense(1, sigmoid)
- Loss: MSE, optimizer: Adam
- Train 10 epochs
- Save to output/model.keras
- tensorflow, numpy only

Return ONLY Python code.
"""
        code = await llm_caller(prompt, force_claude=True, max_tokens=3000)
        if "```python" in code: code = code.split("```python")[1].split("```")[0]
        elif "```" in code: code = code.split("```")[1].split("```")[0]
        return code.strip()

    def generate_synthetic_data(self, size: int = 500, params: dict | None = None) -> Any:
        from agents.synthetic_data import generate_recommendation
        p = params or {}
        return generate_recommendation(n_users=size, seed=p.get("seed", 42))

    async def evaluate(self, checkpoint_path: str, test_data: Any) -> dict:
        try:
            import tensorflow as tf
            import numpy as np
            test     = test_data.get("test", {})
            user_ids = test.get("user_ids")
            item_ids = test.get("item_ids")
            ratings  = test.get("ratings")
            if user_ids is None:
                return {"rmse": 999.0, "loss": 999.0}
            model  = tf.keras.models.load_model(checkpoint_path)
            preds  = model.predict([user_ids, item_ids], verbose=0).flatten()
            rmse   = float(np.sqrt(np.mean((preds - ratings / 5.0) ** 2)))
            return {"rmse": rmse, "loss": rmse}
        except Exception as e:
            return {"rmse": 999.0, "loss": 999.0, "error": str(e)}
