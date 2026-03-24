"""
Graph domain handler — GCN, GAT, GraphSAGE architectures.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from .base_domain import BaseDomain, TF_CODE_SYSTEM, MUTATION_SYSTEM, MECHANISM_SYSTEM


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
        system = MECHANISM_SYSTEM + "\nDomain: graph neural networks (GCN, GAT, GraphSAGE, node classification)."
        prompt = f"Research insights:\n{research_insights}\n\nDerive 3 novel mechanisms for GNNs. JSON array:"
        raw = await llm_caller(prompt, system=system, force_claude=True, max_tokens=1200)
        try:
            start = raw.find("["); end = raw.rfind("]") + 1
            return json.loads(raw[start:end])
        except Exception:
            return [{"name": "topology_aware_attention", "description": "Attention weighted by graph topology", "sympy_expression": "alpha_ij * (h_i + h_j) / deg(i)"}]

    async def propose_mutations(self, base_arch: str, mechanisms: list[dict], llm_caller: Callable, failed_patterns: list[dict] | None = None, **kwargs) -> list[dict]:
        template         = self.get_base_template(base_arch)
        system           = MUTATION_SYSTEM + f"\nDomain: graph neural networks. Available operators: {self.mutation_operators}."
        failure_ctx      = self._format_failure_context(failed_patterns)
        explored_summary = kwargs.get("explored_summary")
        explored_ctx     = (
            f"Already-explored architecture space (avoid these regions — novelty score rewards distance from them):\n{explored_summary}"
            if explored_summary else ""
        )
        prompt           = (
            f"Base architecture:\n{json.dumps(template, indent=2)}"
            f"\n\nMechanisms:\n{json.dumps(mechanisms, indent=2)}"
            + (f"\n\n{failure_ctx}" if failure_ctx else "")
            + (f"\n\n{explored_ctx}" if explored_ctx else "")
            + "\n\nPropose 3 mutations. JSON array:"
        )
        raw = await llm_caller(prompt, system=system, force_claude=True, max_tokens=1200)
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

    async def generate_code(self, arch_spec: dict, llm_caller: Callable, mechanisms: list[dict] | None = None, rationale: str | None = None) -> str:
        system = (
            TF_CODE_SYSTEM +
            "\nDOMAIN: Graph neural networks (node classification)."
            "\nSYNTHETIC DATA: node_features = np.random.randn(200,16).astype(np.float32); "
            "adj = (np.random.rand(200,200)>0.95).astype(np.float32); "
            "labels = np.random.randint(0,5,(200,)).astype(np.int32). "
            "TF has no native GNN layers — implement message-passing GCN from scratch using tf.matmul."
            "\nLOSS: sparse_categorical_crossentropy. EPOCHS: 5."
        )

        prompt = self._build_code_prompt(arch_spec, mechanisms, rationale)
        code = await llm_caller(prompt, system=system, force_claude=True, max_tokens=3000)
        if "```python" in code: code = code.split("```python")[1].split("```")[0]
        elif "```" in code:     code = code.split("```")[1].split("```")[0]
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
