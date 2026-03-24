"""
Vision domain handler — CNN, ViT, and image-based architectures.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from .base_domain import BaseDomain, TF_CODE_SYSTEM, MUTATION_SYSTEM, MECHANISM_SYSTEM


class VisionDomain(BaseDomain):
    name = "vision"
    supported_architectures = ["cnn", "vit", "efficientnet", "unet", "detr"]
    mutation_operators = [
        "layer_insertion", "kernel_size_change", "skip_connection_add",
        "attention_variant", "normalization_change", "activation_change",
        "depth_change", "width_change",
    ]
    metrics = ["accuracy", "top5_accuracy", "loss", "memory_mb", "inference_time_ms", "training_time_s"]

    base_templates = {
        "cnn": {
            "type"         : "cnn",
            "input_shape"  : [32, 32, 3],
            "normalization": "batch_norm",
            "activation"   : "relu",
            "layers": [
                {"type": "conv2d", "filters": 32,  "kernel_size": 3, "activation": "relu"},
                {"type": "conv2d", "filters": 64,  "kernel_size": 3, "activation": "relu"},
                {"type": "pool",   "pool_type": "max", "pool_size": 2},
                {"type": "conv2d", "filters": 128, "kernel_size": 3, "activation": "relu"},
                {"type": "pool",   "pool_type": "global_avg"},
                {"type": "dense",  "units": 256, "activation": "relu", "dropout": 0.3},
                {"type": "dense",  "units": 10,  "activation": "softmax"},
            ],
        },
        "vit": {
            "type"        : "vit",
            "input_shape" : [32, 32, 3],
            "patch_size"  : 4,
            "embed_dim"   : 128,
            "normalization": "layer_norm",
            "activation"  : "gelu",
            "attention"   : {"type": "multi_head", "num_heads": 4, "key_dim": 32},
            "layers": [
                {"type": "patch_embed",   "patch_size": 4, "embed_dim": 128},
                {"type": "transformer",   "num_layers": 6, "mlp_dim": 256, "dropout": 0.1},
                {"type": "class_token",   "embed_dim": 128},
                {"type": "dense",         "units": 10, "activation": "softmax"},
            ],
        },
    }

    async def generate_mechanism(self, research_insights: str, llm_caller: Callable) -> list[dict]:
        system = MECHANISM_SYSTEM + "\nDomain: computer vision (CNNs, ViTs, image classification/detection)."
        prompt = f"Research insights:\n{research_insights}\n\nDerive 3 novel mechanisms for image models. JSON array:"
        raw = await llm_caller(prompt, system=system, force_claude=True, max_tokens=1500)
        try:
            start = raw.find("[")
            end   = raw.rfind("]") + 1
            return json.loads(raw[start:end])
        except Exception:
            return [{"name": "residual_attention", "description": "Attention-weighted residual connections", "sympy_expression": "x + attention(x) * x"}]

    async def propose_mutations(self, base_arch: str, mechanisms: list[dict], llm_caller: Callable, failed_patterns: list[dict] | None = None, **kwargs) -> list[dict]:
        template         = self.get_base_template(base_arch)
        system           = MUTATION_SYSTEM + f"\nDomain: computer vision. Available operators: {self.mutation_operators}."
        failure_ctx      = self._format_failure_context(failed_patterns)
        explored_summary = kwargs.get("explored_summary")
        task_description = kwargs.get("task_description", "")
        explored_ctx     = (
            f"Already-explored architecture space (avoid these regions — novelty score rewards distance from them):\n{explored_summary}"
            if explored_summary else ""
        )
        goal_ctx         = f"Research goal: {task_description}" if task_description else ""
        prompt           = (
            f"Base architecture:\n{json.dumps(template, indent=2)}"
            f"\n\nMechanisms:\n{json.dumps(mechanisms, indent=2)}"
            + (f"\n\n{goal_ctx}" if goal_ctx else "")
            + (f"\n\n{failure_ctx}" if failure_ctx else "")
            + (f"\n\n{explored_ctx}" if explored_ctx else "")
            + "\n\nPropose 3 mutations. JSON array:"
        )
        raw = await llm_caller(prompt, system=system, force_claude=True, max_tokens=1500)
        try:
            start = raw.find("[")
            end   = raw.rfind("]") + 1
            proposals = json.loads(raw[start:end])
        except Exception:
            proposals = [{"architecture_name": f"{base_arch}_mutant_1", "mutations": ["skip_connection_add", "attention_variant"], "rationale": "Default mutation"}]

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
            "\nDOMAIN: Vision classification (images)."
            "\nSYNTHETIC DATA: X = tf.random.normal(shape=(1000,32,32,3)); "
            "y = tf.random.uniform(shape=(1000,), minval=0, maxval=10, dtype=tf.int32)"
            "\nLOSS: sparse_categorical_crossentropy. METRICS: accuracy. EPOCHS: 5."
        )

        prompt = self._build_code_prompt(arch_spec, mechanisms, rationale)
        code = await llm_caller(prompt, system=system, force_claude=True, max_tokens=3000)
        if "```python" in code: code = code.split("```python")[1].split("```")[0]
        elif "```" in code:     code = code.split("```")[1].split("```")[0]
        return code.strip()

    def generate_synthetic_data(self, size: int = 1000, params: dict | None = None) -> Any:
        from agents.synthetic_data import generate_image
        p = params or {}
        return generate_image(
            size=size,
            resolution=p.get("resolution", 32),
            n_classes=p.get("n_classes", 10),
            seed=p.get("seed", 42),
        )

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
