"""
Audio domain handler — Conformer, CNN-Audio, speech/sound architectures.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from .base_domain import BaseDomain


class AudioDomain(BaseDomain):
    name = "audio"
    supported_architectures = ["conformer", "cnn_audio"]
    mutation_operators = [
        "layer_insertion", "kernel_size_change", "attention_variant",
        "normalization_change", "depth_change", "width_change",
    ]
    metrics = ["loss", "accuracy", "memory_mb", "inference_time_ms", "training_time_s"]

    base_templates = {
        "cnn_audio": {
            "type"         : "cnn_audio",
            "input_shape"  : [64, 32, 1],   # (n_mels, time_frames, channels)
            "normalization": "batch_norm",
            "activation"   : "relu",
            "layers": [
                {"type": "conv2d", "filters": 32,  "kernel_size": 3, "activation": "relu"},
                {"type": "conv2d", "filters": 64,  "kernel_size": 3, "activation": "relu"},
                {"type": "pool",   "pool_type": "max", "pool_size": 2},
                {"type": "conv2d", "filters": 128, "kernel_size": 3, "activation": "relu"},
                {"type": "pool",   "pool_type": "global_avg"},
                {"type": "dense",  "units": 128, "activation": "relu", "dropout": 0.3},
                {"type": "dense",  "units": 10,  "activation": "softmax"},
            ],
        },
        "conformer": {
            "type"        : "conformer",
            "input_shape" : [64, 32],   # (n_mels, time_frames)
            "embed_dim"   : 128,
            "normalization": "layer_norm",
            "activation"  : "swish",
            "attention"   : {"type": "multi_head", "num_heads": 4, "key_dim": 32},
            "layers": [
                {"type": "conv_subsampling", "filters": 128, "kernel_size": 3, "stride": 2},
                {"type": "conformer_block",  "num_layers": 4, "ffn_dim": 256, "conv_kernel": 31, "dropout": 0.1},
                {"type": "pool",             "pool_type": "global_avg"},
                {"type": "dense",            "units": 10, "activation": "softmax"},
            ],
        },
    }

    async def generate_mechanism(self, research_insights: str, llm_caller: Callable) -> list[dict]:
        prompt = f"""
You are an audio/speech ML researcher. Based on:

{research_insights}

Derive 3 novel mechanisms for audio neural networks.
JSON array: [{{"name": str, "description": str, "sympy_expression": str}}]
"""
        raw = await llm_caller(prompt, force_claude=True, max_tokens=1200)
        try:
            start = raw.find("["); end = raw.rfind("]") + 1
            return json.loads(raw[start:end])
        except Exception:
            return [{"name": "frequency_attention", "description": "Attention over frequency bins", "sympy_expression": "softmax(Q_f @ K_f.T) @ V_f"}]

    async def propose_mutations(self, base_arch: str, mechanisms: list[dict], llm_caller: Callable) -> list[dict]:
        template = self.get_base_template(base_arch)
        prompt   = f"""
Audio architecture: {json.dumps(template, indent=2)}
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
Write a complete TensorFlow/Keras training script for this audio architecture:

{json.dumps(arch_spec, indent=2)}

Requirements:
- Input: mel-spectrogram tensors shape (N, 64, 32, 1)
- Generate synthetic data: np.random.uniform(0, 1, (1000, 64, 32, 1))
- Labels: np.random.randint(0, 10, (1000,))
- Train 5 epochs, Adam optimizer
- Save to output/model.keras
- tensorflow, numpy only

Return ONLY Python code.
"""
        code = await llm_caller(prompt, force_claude=True, max_tokens=3000)
        if "```python" in code: code = code.split("```python")[1].split("```")[0]
        elif "```" in code: code = code.split("```")[1].split("```")[0]
        return code.strip()

    def generate_synthetic_data(self, size: int = 1000, params: dict | None = None) -> Any:
        from agents.synthetic_data import generate_audio
        p = params or {}
        return generate_audio(size=size, n_classes=p.get("n_classes", 10), seed=p.get("seed", 42))

    async def evaluate(self, checkpoint_path: str, test_data: Any) -> dict:
        try:
            import tensorflow as tf
            import numpy as np
            X_test, y_test = test_data[1], test_data[3]
            # Add channel dim if missing
            if X_test.ndim == 3:
                X_test = X_test[..., np.newaxis]
            model = tf.keras.models.load_model(checkpoint_path)
            loss, acc = model.evaluate(X_test, y_test, verbose=0)
            return {"loss": float(loss), "accuracy": float(acc)}
        except Exception as e:
            return {"loss": 999.0, "accuracy": 0.0, "error": str(e)}
