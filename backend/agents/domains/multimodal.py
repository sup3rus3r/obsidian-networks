"""
Multimodal domain handler — CLIP, Flamingo-style architectures.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from .base_domain import BaseDomain


class MultimodalDomain(BaseDomain):
    name = "multimodal"
    supported_architectures = ["clip", "flamingo"]
    mutation_operators = [
        "layer_insertion", "attention_variant", "normalization_change",
        "fusion_type_change", "depth_change", "width_change",
    ]
    metrics = ["contrastive_loss", "cross_modal_accuracy", "memory_mb", "inference_time_ms", "training_time_s"]

    base_templates = {
        "clip": {
            "type"        : "clip",
            "image_encoder": {
                "type"       : "vit",
                "input_shape": [32, 32, 3],
                "patch_size" : 4,
                "embed_dim"  : 128,
                "num_layers" : 4,
                "num_heads"  : 4,
            },
            "text_encoder": {
                "type"      : "transformer",
                "vocab_size": 5000,
                "seq_len"   : 64,
                "embed_dim" : 128,
                "num_layers": 4,
                "num_heads" : 4,
            },
            "projection_dim": 128,
            "temperature"   : 0.07,
            "fusion"        : "contrastive",
            "normalization" : "layer_norm",
            "activation"    : "gelu",
        },
        "flamingo": {
            "type"        : "flamingo",
            "image_encoder": {"type": "vit", "embed_dim": 128, "num_layers": 4},
            "text_decoder" : {"type": "gpt", "embed_dim": 128, "num_layers": 4},
            "cross_attention_layers": 2,
            "fusion"       : "cross_attention",
            "normalization": "layer_norm",
            "activation"   : "gelu",
        },
    }

    async def generate_mechanism(self, research_insights: str, llm_caller: Callable) -> list[dict]:
        prompt = f"""
You are a multimodal AI researcher. Based on:

{research_insights}

Derive 3 novel mechanisms for image-text models.
JSON array: [{{"name": str, "description": str, "sympy_expression": str}}]
"""
        raw = await llm_caller(prompt, force_claude=True, max_tokens=1200)
        try:
            start = raw.find("["); end = raw.rfind("]") + 1
            return json.loads(raw[start:end])
        except Exception:
            return [{"name": "asymmetric_contrastive", "description": "Asymmetric temperature per modality", "sympy_expression": "-log(exp(sim/t_i) / sum(exp(sim_j/t_i)))"}]

    async def propose_mutations(self, base_arch: str, mechanisms: list[dict], llm_caller: Callable) -> list[dict]:
        template = self.get_base_template(base_arch)
        prompt   = f"""
Multimodal architecture: {json.dumps(template, indent=2)}
Mechanisms: {json.dumps(mechanisms, indent=2)}
Propose 3 mutations. Operators: {self.mutation_operators}
JSON: [{{"architecture_name": str, "mutations": [str], "rationale": str}}]
"""
        raw = await llm_caller(prompt, force_claude=True, max_tokens=1200)
        try:
            start = raw.find("["); end = raw.rfind("]") + 1
            proposals = json.loads(raw[start:end])
        except Exception:
            proposals = [{"architecture_name": f"{base_arch}_mutant", "mutations": ["fusion_type_change"], "rationale": "Default"}]

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
Write a complete TensorFlow/Keras training script for this multimodal architecture:

{json.dumps(arch_spec, indent=2)}

Requirements:
- Images: np.random.randn(500, 32, 32, 3).astype(np.float32)
- Text tokens: np.random.randint(1, 5000, (500, 64))
- Use contrastive loss (InfoNCE): -log(exp(sim_ii/t) / sum_j(exp(sim_ij/t)))
- Simple image encoder: Conv2D + GlobalAvgPool + Dense(128)
- Simple text encoder: Embedding + GlobalAvgPool + Dense(128)
- Normalize both to unit length, compute cosine similarity matrix
- Train 5 epochs, Adam
- Save to output/model.keras
- tensorflow, numpy only

Return ONLY Python code.
"""
        code = await llm_caller(prompt, force_claude=True, max_tokens=3000)
        if "```python" in code: code = code.split("```python")[1].split("```")[0]
        elif "```" in code: code = code.split("```")[1].split("```")[0]
        return code.strip()

    def generate_synthetic_data(self, size: int = 500, params: dict | None = None) -> Any:
        from agents.synthetic_data import generate_multimodal
        p = params or {}
        return generate_multimodal(size=size, seed=p.get("seed", 42))

    async def evaluate(self, checkpoint_path: str, test_data: Any) -> dict:
        try:
            import tensorflow as tf
            import numpy as np
            test = test_data.get("test", {})
            images = test.get("images")
            tokens = test.get("tokens")
            if images is None or tokens is None:
                return {"contrastive_loss": 999.0, "cross_modal_accuracy": 0.0}
            model = tf.keras.models.load_model(checkpoint_path)
            # Simplified eval: measure reconstruction loss
            result = model.evaluate([images, tokens], verbose=0)
            loss = result[0] if isinstance(result, (list, tuple)) else float(result)
            return {"contrastive_loss": float(loss), "cross_modal_accuracy": 0.0}
        except Exception as e:
            return {"contrastive_loss": 999.0, "cross_modal_accuracy": 0.0, "error": str(e)}
