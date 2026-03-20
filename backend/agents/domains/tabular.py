"""
Tabular domain handler — MLP, ResNet-Tabular architectures.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from .base_domain import BaseDomain, TF_CODE_SYSTEM, MUTATION_SYSTEM, MECHANISM_SYSTEM


class TabularDomain(BaseDomain):
    name = "tabular"
    supported_architectures = ["mlp", "resnet_tabular", "tabnet"]
    mutation_operators = [
        "layer_insertion", "normalization_change", "activation_change",
        "depth_change", "width_change", "skip_connection_add",
    ]
    metrics = ["loss", "accuracy", "auc", "rmse", "memory_mb", "inference_time_ms", "training_time_s"]

    base_templates = {
        "mlp": {
            "type"         : "mlp",
            "n_features"   : 20,
            "task"         : "classification",
            "normalization": "batch_norm",
            "activation"   : "relu",
            "layers": [
                {"type": "dense", "units": 256, "activation": "relu",  "dropout": 0.3},
                {"type": "dense", "units": 128, "activation": "relu",  "dropout": 0.3},
                {"type": "dense", "units": 64,  "activation": "relu",  "dropout": 0.2},
                {"type": "dense", "units": 2,   "activation": "softmax"},
            ],
        },
        "resnet_tabular": {
            "type"         : "resnet_tabular",
            "n_features"   : 20,
            "task"         : "classification",
            "normalization": "batch_norm",
            "activation"   : "relu",
            "layers": [
                {"type": "dense",    "units": 256, "activation": "relu"},
                {"type": "residual", "units": 256, "n_blocks": 3, "dropout": 0.2},
                {"type": "dense",    "units": 128, "activation": "relu"},
                {"type": "dense",    "units": 2,   "activation": "softmax"},
            ],
            "skip_connections": [{"from": 0, "to": 2}],
        },
    }

    async def generate_mechanism(self, research_insights: str, llm_caller: Callable) -> list[dict]:
        system = MECHANISM_SYSTEM + "\nDomain: tabular data (MLPs, ResNets, feature interaction networks)."
        prompt = f"Research insights:\n{research_insights}\n\nDerive 3 novel mechanisms for tabular neural networks. JSON array:"
        raw = await llm_caller(prompt, system=system, force_claude=True, max_tokens=1200)
        try:
            start = raw.find("["); end = raw.rfind("]") + 1
            return json.loads(raw[start:end])
        except Exception:
            return [{"name": "feature_interaction", "description": "Explicit pairwise feature interactions", "sympy_expression": "sum_ij(w_ij * x_i * x_j)"}]

    async def propose_mutations(self, base_arch: str, mechanisms: list[dict], llm_caller: Callable) -> list[dict]:
        template = self.get_base_template(base_arch)
        system   = MUTATION_SYSTEM + f"\nDomain: tabular data. Available operators: {self.mutation_operators}."
        prompt   = f"Base architecture:\n{json.dumps(template, indent=2)}\n\nMechanisms:\n{json.dumps(mechanisms, indent=2)}\n\nPropose 3 mutations. JSON array:"
        raw = await llm_caller(prompt, system=system, force_claude=True, max_tokens=1200)
        try:
            start = raw.find("["); end = raw.rfind("]") + 1
            proposals = json.loads(raw[start:end])
        except Exception:
            proposals = [{"architecture_name": f"{base_arch}_mutant", "mutations": ["skip_connection_add", "width_change"], "rationale": "Default"}]

        from agents.mutations import apply_mutations
        return [{
            "architecture_name": p.get("architecture_name", f"{base_arch}_mutant"),
            "base_template"    : base_arch,
            "mutations"        : p.get("mutations", []),
            "spec"             : apply_mutations(template, p.get("mutations", [])),
            "rationale"        : p.get("rationale", ""),
        } for p in proposals]

    async def generate_code(self, arch_spec: dict, llm_caller: Callable) -> str:
        system = (
            TF_CODE_SYSTEM +
            "\nDOMAIN: Tabular classification."
            "\nSYNTHETIC DATA: from sklearn.datasets import make_classification; "
            "X, y = make_classification(n_samples=1000, n_features=20, random_state=42); "
            "X = X.astype(np.float32); y = y.astype(np.int32). "
            "Add tf.keras.layers.Normalization() as first layer, adapt on X_train."
            "\nLOSS: sparse_categorical_crossentropy. METRICS: accuracy. EPOCHS: 20."
        )
        prompt = f"Architecture spec to implement:\n{json.dumps(arch_spec, indent=2)}"
        code = await llm_caller(prompt, system=system, force_claude=True, max_tokens=3000)
        if "```python" in code: code = code.split("```python")[1].split("```")[0]
        elif "```" in code:     code = code.split("```")[1].split("```")[0]
        return code.strip()

    def generate_synthetic_data(self, size: int = 1000, params: dict | None = None) -> Any:
        from agents.synthetic_data import generate_tabular
        p = params or {}
        return generate_tabular(size=size, task=p.get("task", "classification"), seed=p.get("seed", 42))

    async def evaluate(self, checkpoint_path: str, test_data: Any) -> dict:
        try:
            import tensorflow as tf
            import numpy as np
            X_test, y_test = test_data[1], test_data[3]
            model = tf.keras.models.load_model(checkpoint_path)
            loss, acc = model.evaluate(X_test, y_test, verbose=0)
            return {"loss": float(loss), "accuracy": float(acc)}
        except Exception as e:
            return {"loss": 999.0, "accuracy": 0.0, "error": str(e)}
