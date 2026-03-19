"""
Vision domain handler — CNN, ViT, and image-based architectures.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from .base_domain import BaseDomain


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
        prompt = f"""
You are a computer vision researcher. Based on these research insights:

{research_insights}

Derive 3 novel architectural mechanisms for image models. For each mechanism provide:
1. name: short identifier (snake_case)
2. description: what it does and why it helps
3. sympy_expression: a mathematical expression describing the key operation (use sympy syntax)

Respond with a JSON array only. Example:
[{{"name": "adaptive_pooling", "description": "Pool features adaptively based on spatial variance", "sympy_expression": "pool(x, sigma(x))"}}]
"""
        raw = await llm_caller(prompt, force_claude=True, max_tokens=1500)
        try:
            start = raw.find("[")
            end   = raw.rfind("]") + 1
            return json.loads(raw[start:end])
        except Exception:
            return [{"name": "residual_attention", "description": "Attention-weighted residual connections", "sympy_expression": "x + attention(x) * x"}]

    async def propose_mutations(self, base_arch: str, mechanisms: list[dict], llm_caller: Callable) -> list[dict]:
        template = self.get_base_template(base_arch)
        prompt   = f"""
You are a neural architecture search expert for computer vision.

Base architecture: {json.dumps(template, indent=2)}

Novel mechanisms discovered:
{json.dumps(mechanisms, indent=2)}

Propose 3 architecture mutations that incorporate these mechanisms. Each mutation should:
- Have a unique descriptive name
- List which mutation operators to apply (from: {self.mutation_operators})
- Include rationale tied to a specific mechanism

Respond with JSON array only:
[{{"architecture_name": "cnn_adaptive_attention", "mutations": ["attention_variant", "skip_connection_add"], "rationale": "Adds attention weighted by mechanism..."}}]
"""
        raw = await llm_caller(prompt, force_claude=True, max_tokens=1500)
        try:
            start = raw.find("[")
            end   = raw.rfind("]") + 1
            proposals = json.loads(raw[start:end])
        except Exception:
            proposals = [{"architecture_name": f"{base_arch}_mutant_1", "mutations": ["skip_connection_add", "attention_variant"], "rationale": "Default mutation"}]

        # Apply mutations to produce actual specs
        from agents.mutations import apply_mutations, random_mutations
        results = []
        for p in proposals:
            spec = apply_mutations(template, p.get("mutations", []))
            results.append({
                "architecture_name": p.get("architecture_name", f"{base_arch}_mutant"),
                "base_template"    : base_arch,
                "mutations"        : p.get("mutations", []),
                "spec"             : spec,
                "rationale"        : p.get("rationale", ""),
            })
        return results

    async def generate_code(self, arch_spec: dict, llm_caller: Callable) -> str:
        prompt = f"""
Write a complete, self-contained TensorFlow/Keras training script for this vision architecture:

{json.dumps(arch_spec, indent=2)}

Requirements:
- Use tensorflow.keras functional API
- Generate synthetic image data: tf.random.normal(shape=(1000, 32, 32, 3))
- Generate synthetic labels: tf.random.uniform(shape=(1000,), minval=0, maxval=10, dtype=tf.int32)
- Train for 5 epochs with Adam optimizer
- Save model to output/model.keras
- Print final training accuracy
- Use ONLY tensorflow, numpy imports
- No matplotlib, no real datasets

Return ONLY the Python code, no explanation.
"""
        code = await llm_caller(prompt, force_claude=True, max_tokens=3000)
        # Strip markdown fences if present
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
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
