"""
Recommendation domain handler — Collaborative Filtering, Attention-based recommenders.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from .base_domain import BaseDomain, TF_CODE_SYSTEM, MUTATION_SYSTEM, MECHANISM_SYSTEM


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
        system = MECHANISM_SYSTEM + "\nDomain: recommendation systems (collaborative filtering, attention-based recommenders)."
        prompt = f"Research insights:\n{research_insights}\n\nDerive 3 novel mechanisms for recommendation models. JSON array:"
        raw = await llm_caller(prompt, system=system, force_claude=True, max_tokens=1200)
        try:
            start = raw.find("["); end = raw.rfind("]") + 1
            return json.loads(raw[start:end])
        except Exception:
            return [{"name": "popularity_debiasing", "description": "Debias recommendations from popularity", "sympy_expression": "score(u,i) - lambda * log(freq(i))"}]

    async def propose_mutations(self, base_arch: str, mechanisms: list[dict], llm_caller: Callable) -> list[dict]:
        template = self.get_base_template(base_arch)
        system   = MUTATION_SYSTEM + f"\nDomain: recommendation systems. Available operators: {self.mutation_operators}."
        prompt   = f"Base architecture:\n{json.dumps(template, indent=2)}\n\nMechanisms:\n{json.dumps(mechanisms, indent=2)}\n\nPropose 3 mutations. JSON array:"
        raw = await llm_caller(prompt, system=system, force_claude=True, max_tokens=1200)
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

    async def generate_code(self, arch_spec: dict, llm_caller: Callable, mechanisms: list[dict] | None = None, rationale: str | None = None) -> str:
        system = (
            TF_CODE_SYSTEM +
            "\nDOMAIN: Recommendation systems (collaborative filtering)."
            "\nSYNTHETIC DATA: n_users=500, n_items=1000; "
            "user_ids=np.random.randint(0,500,10000); item_ids=np.random.randint(0,1000,10000); "
            "ratings=((np.random.uniform(1,5,10000)-1)/4).astype(np.float32). "
            "User Embedding + Item Embedding → Dot → Dense(1,sigmoid)."
            "\nLOSS: mse. EPOCHS: 10."
        )
        ctx = self._format_mechanism_context(mechanisms, rationale)
        prompt = f"Architecture spec to implement:\n{json.dumps(arch_spec, indent=2)}" + (f"\n\n{ctx}" if ctx else "")
        code = await llm_caller(prompt, system=system, force_claude=True, max_tokens=3000)
        if "```python" in code: code = code.split("```python")[1].split("```")[0]
        elif "```" in code:     code = code.split("```")[1].split("```")[0]
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
