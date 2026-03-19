"""
Graph domain handler — GCN, GAT, GraphSAGE architectures.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from .base_domain import BaseDomain


class GraphDomain(BaseDomain):
    name = "graph"
    supported_architectures = ["gcn", "gat", "graphsage", "gin"]
    mutation_operators = [
        "layer_insertion", "attention_variant", "normalization_change",
        "activation_change", "depth_change", "width_change",
    ]
    metrics = ["node_accuracy", "link_auc", "loss", "memory_mb", "inference_time_ms", "training_time_s"]

    base_templates = {
        "gcn": {
            "type"         : "gcn",
            "n_features"   : 16,
            "n_classes"    : 5,
            "normalization": "batch_norm",
            "activation"   : "relu",
            "layers": [
                {"type": "gcn_conv", "out_channels": 64,  "activation": "relu", "dropout": 0.3},
                {"type": "gcn_conv", "out_channels": 32,  "activation": "relu", "dropout": 0.3},
                {"type": "gcn_conv", "out_channels": 5,   "activation": "softmax"},
            ],
        },
        "gat": {
            "type"        : "gat",
            "n_features"  : 16,
            "n_classes"   : 5,
            "normalization": "layer_norm",
            "activation"  : "elu",
            "attention"   : {"type": "multi_head", "num_heads": 4, "key_dim": 16},
            "layers": [
                {"type": "gat_conv", "out_channels": 16, "heads": 4, "activation": "elu", "dropout": 0.3},
                {"type": "gat_conv", "out_channels": 5,  "heads": 1, "activation": "softmax"},
            ],
        },
    }

    async def generate_mechanism(self, research_insights: str, llm_caller: Callable) -> list[dict]:
        prompt = f"""
You are a graph neural network researcher. Based on:

{research_insights}

Derive 3 novel mechanisms for GNNs.
JSON array: [{{"name": str, "description": str, "sympy_expression": str}}]
"""
        raw = await llm_caller(prompt, force_claude=True, max_tokens=1200)
        try:
            start = raw.find("["); end = raw.rfind("]") + 1
            return json.loads(raw[start:end])
        except Exception:
            return [{"name": "topology_aware_attention", "description": "Attention weighted by graph topology", "sympy_expression": "alpha_ij * (h_i + h_j) / deg(i)"}]

    async def propose_mutations(self, base_arch: str, mechanisms: list[dict], llm_caller: Callable) -> list[dict]:
        template = self.get_base_template(base_arch)
        prompt   = f"""
GNN architecture: {json.dumps(template, indent=2)}
Mechanisms: {json.dumps(mechanisms, indent=2)}
Propose 3 mutations. Operators: {self.mutation_operators}
JSON: [{{"architecture_name": str, "mutations": [str], "rationale": str}}]
"""
        raw = await llm_caller(prompt, force_claude=True, max_tokens=1200)
        try:
            start = raw.find("["); end = raw.rfind("]") + 1
            proposals = json.loads(raw[start:end])
        except Exception:
            proposals = [{"architecture_name": f"{base_arch}_mutant", "mutations": ["depth_change"], "rationale": "Default"}]

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
Write a complete TensorFlow/Keras training script for this GNN architecture:

{json.dumps(arch_spec, indent=2)}

Requirements:
- Since TF doesn't have native GNN layers, implement a simple message-passing GCN from scratch
- Generate: node_features = np.random.randn(200, 16).astype(np.float32)
- Generate: adj_matrix = (np.random.rand(200, 200) > 0.95).astype(np.float32) (sparse adjacency)
- Labels: np.random.randint(0, 5, (200,))
- Train 5 epochs, Adam, sparse_categorical_crossentropy
- Save model to output/model.keras
- tensorflow, numpy only

Return ONLY Python code.
"""
        code = await llm_caller(prompt, force_claude=True, max_tokens=3000)
        if "```python" in code: code = code.split("```python")[1].split("```")[0]
        elif "```" in code: code = code.split("```")[1].split("```")[0]
        return code.strip()

    def generate_synthetic_data(self, size: int = 200, params: dict | None = None) -> Any:
        from agents.synthetic_data import generate_graph
        p = params or {}
        return generate_graph(n_nodes=size, n_classes=p.get("n_classes", 5), seed=p.get("seed", 42))

    async def evaluate(self, checkpoint_path: str, test_data: Any) -> dict:
        try:
            import tensorflow as tf
            import numpy as np
            node_features = test_data["node_features"]
            labels        = test_data["labels"]
            test_mask     = test_data["test_mask"]
            model = tf.keras.models.load_model(checkpoint_path)
            preds = model.predict(node_features, verbose=0)
            pred_classes = np.argmax(preds[test_mask], axis=1)
            true_classes = labels[test_mask]
            acc = float(np.mean(pred_classes == true_classes))
            return {"node_accuracy": acc, "loss": float(-np.log(max(acc, 1e-6)))}
        except Exception as e:
            return {"node_accuracy": 0.0, "loss": 999.0, "error": str(e)}
