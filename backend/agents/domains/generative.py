"""
Generative domain handler — GAN, VAE, Diffusion architectures.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from .base_domain import BaseDomain


class GenerativeDomain(BaseDomain):
    name = "generative"
    supported_architectures = ["gan", "vae", "diffusion"]
    mutation_operators = [
        "layer_insertion", "normalization_change", "activation_change",
        "depth_change", "width_change", "skip_connection_add",
    ]
    metrics = ["loss", "generator_loss", "discriminator_loss", "reconstruction_loss", "kl_loss", "memory_mb", "training_time_s"]

    base_templates = {
        "vae": {
            "type"       : "vae",
            "input_shape": [32, 32, 3],
            "latent_dim" : 128,
            "normalization": "batch_norm",
            "activation" : "relu",
            "encoder": [
                {"type": "conv2d", "filters": 32,  "kernel_size": 3, "stride": 2, "activation": "relu"},
                {"type": "conv2d", "filters": 64,  "kernel_size": 3, "stride": 2, "activation": "relu"},
                {"type": "flatten"},
                {"type": "dense", "units": 256, "activation": "relu"},
            ],
            "decoder": [
                {"type": "dense",         "units": 256, "activation": "relu"},
                {"type": "reshape",       "shape": [8, 8, 4]},
                {"type": "conv2d_transpose", "filters": 64, "kernel_size": 3, "stride": 2, "activation": "relu"},
                {"type": "conv2d_transpose", "filters": 32, "kernel_size": 3, "stride": 2, "activation": "relu"},
                {"type": "conv2d_transpose", "filters": 3,  "kernel_size": 3, "stride": 1, "activation": "sigmoid"},
            ],
        },
        "gan": {
            "type"       : "gan",
            "input_shape": [32, 32, 3],
            "latent_dim" : 128,
            "normalization": "batch_norm",
            "activation" : "leaky_relu",
            "generator": [
                {"type": "dense",    "units": 8*8*128, "activation": "relu"},
                {"type": "reshape",  "shape": [8, 8, 128]},
                {"type": "conv2d_transpose", "filters": 64, "kernel_size": 4, "stride": 2, "activation": "relu"},
                {"type": "conv2d_transpose", "filters": 32, "kernel_size": 4, "stride": 2, "activation": "relu"},
                {"type": "conv2d_transpose", "filters": 3,  "kernel_size": 4, "stride": 1, "activation": "tanh"},
            ],
            "discriminator": [
                {"type": "conv2d",  "filters": 64,  "kernel_size": 4, "stride": 2, "activation": "leaky_relu"},
                {"type": "conv2d",  "filters": 128, "kernel_size": 4, "stride": 2, "activation": "leaky_relu"},
                {"type": "flatten"},
                {"type": "dense",   "units": 1, "activation": "sigmoid"},
            ],
        },
    }

    async def generate_mechanism(self, research_insights: str, llm_caller: Callable) -> list[dict]:
        prompt = f"""
You are a generative model researcher. Based on:

{research_insights}

Derive 3 novel mechanisms for generative neural networks (GANs, VAEs, Diffusion).
JSON array: [{{"name": str, "description": str, "sympy_expression": str}}]
"""
        raw = await llm_caller(prompt, force_claude=True, max_tokens=1200)
        try:
            start = raw.find("["); end = raw.rfind("]") + 1
            return json.loads(raw[start:end])
        except Exception:
            return [{"name": "adaptive_kl_weight", "description": "KL weight annealed by training step", "sympy_expression": "beta(t) = min(1, t/T_anneal)"}]

    async def propose_mutations(self, base_arch: str, mechanisms: list[dict], llm_caller: Callable) -> list[dict]:
        template = self.get_base_template(base_arch)
        prompt   = f"""
Generative architecture: {json.dumps(template, indent=2)}
Mechanisms: {json.dumps(mechanisms, indent=2)}
Propose 3 mutations. Operators: {self.mutation_operators}
JSON: [{{"architecture_name": str, "mutations": [str], "rationale": str}}]
"""
        raw = await llm_caller(prompt, force_claude=True, max_tokens=1200)
        try:
            start = raw.find("["); end = raw.rfind("]") + 1
            proposals = json.loads(raw[start:end])
        except Exception:
            proposals = [{"architecture_name": f"{base_arch}_mutant", "mutations": ["normalization_change"], "rationale": "Default"}]

        from agents.mutations import apply_mutations
        return [{
            "architecture_name": p.get("architecture_name", f"{base_arch}_mutant"),
            "base_template"    : base_arch,
            "mutations"        : p.get("mutations", []),
            "spec"             : apply_mutations(template, p.get("mutations", [])),
            "rationale"        : p.get("rationale", ""),
        } for p in proposals]

    async def generate_code(self, arch_spec: dict, llm_caller: Callable) -> str:
        arch_type = arch_spec.get("type", "vae")
        prompt = f"""
Write a complete TensorFlow/Keras training script for this {arch_type} architecture:

{json.dumps(arch_spec, indent=2)}

Requirements:
- Generate synthetic images: X_train = np.random.uniform(0, 1, (500, 32, 32, 3)).astype(np.float32)
- For VAE: implement encoder, reparameterization trick (z = mu + eps * sigma), decoder, ELBO loss
- For GAN: implement generator + discriminator with alternating training loop
- Train 5 epochs
- Save the main model (encoder for VAE, generator for GAN) to output/model.keras
- tensorflow, numpy only

Return ONLY Python code.
"""
        code = await llm_caller(prompt, force_claude=True, max_tokens=4000)
        if "```python" in code: code = code.split("```python")[1].split("```")[0]
        elif "```" in code: code = code.split("```")[1].split("```")[0]
        return code.strip()

    def generate_synthetic_data(self, size: int = 500, params: dict | None = None) -> Any:
        from agents.synthetic_data import generate_generative
        p = params or {}
        return generate_generative(size=size, seed=p.get("seed", 42))

    async def evaluate(self, checkpoint_path: str, test_data: Any) -> dict:
        try:
            import tensorflow as tf
            import numpy as np
            test   = test_data.get("test", {})
            images = test.get("images")
            noise  = test.get("noise")
            if images is None:
                return {"loss": 999.0, "reconstruction_loss": 999.0}
            model  = tf.keras.models.load_model(checkpoint_path)
            # For VAE: reconstruction loss. For GAN generator: just run forward pass.
            try:
                result = model.predict(noise, verbose=0) if noise is not None else model.predict(images, verbose=0)
                recon_loss = float(np.mean((result - images[:len(result)]) ** 2)) if result.shape == images[:len(result)].shape else 0.0
            except Exception:
                recon_loss = 0.0
            return {"loss": recon_loss, "reconstruction_loss": recon_loss}
        except Exception as e:
            return {"loss": 999.0, "reconstruction_loss": 999.0, "error": str(e)}
