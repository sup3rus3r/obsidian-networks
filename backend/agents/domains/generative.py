"""
Generative domain handler — GAN, VAE, Diffusion architectures.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from .base_domain import BaseDomain, TF_CODE_SYSTEM, MUTATION_SYSTEM, MECHANISM_SYSTEM


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
        system = MECHANISM_SYSTEM + "\nDomain: generative models (GANs, VAEs, Diffusion models, image synthesis)."
        prompt = f"Research insights:\n{research_insights}\n\nDerive 3 novel mechanisms for generative models. JSON array:"
        raw = await llm_caller(prompt, system=system, force_claude=True, max_tokens=1200)
        try:
            start = raw.find("["); end = raw.rfind("]") + 1
            return json.loads(raw[start:end])
        except Exception:
            return [{"name": "adaptive_kl_weight", "description": "KL weight annealed by training step", "sympy_expression": "beta(t) = min(1, t/T_anneal)"}]

    async def propose_mutations(self, base_arch: str, mechanisms: list[dict], llm_caller: Callable, failed_patterns: list[dict] | None = None) -> list[dict]:
        template    = self.get_base_template(base_arch)
        system      = MUTATION_SYSTEM + f"\nDomain: generative models. Available operators: {self.mutation_operators}."
        failure_ctx = self._format_failure_context(failed_patterns)
        prompt      = (
            f"Base architecture:\n{json.dumps(template, indent=2)}"
            f"\n\nMechanisms:\n{json.dumps(mechanisms, indent=2)}"
            + (f"\n\n{failure_ctx}" if failure_ctx else "")
            + "\n\nPropose 3 mutations. JSON array:"
        )
        raw = await llm_caller(prompt, system=system, force_claude=True, max_tokens=1200)
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

    async def generate_code(self, arch_spec: dict, llm_caller: Callable, mechanisms: list[dict] | None = None, rationale: str | None = None) -> str:
        arch_type = arch_spec.get("type", "vae")
        system = (
            TF_CODE_SYSTEM +
            f"\nDOMAIN: Generative models ({arch_type.upper()})."
            "\nSYNTHETIC DATA: X = np.random.uniform(0,1,(500,32,32,3)).astype(np.float32)"
            "\nFor VAE: encoder → reparameterization (z=mu+eps*sigma) → decoder, minimize ELBO loss."
            "\nFor GAN: alternate generator/discriminator training steps each batch."
            "\nSave main model (encoder for VAE, generator for GAN) to output/model.keras. EPOCHS: 5."
        )
        ctx = self._format_mechanism_context(mechanisms, rationale)
        prompt = f"Architecture spec to implement:\n{json.dumps(arch_spec, indent=2)}" + (f"\n\n{ctx}" if ctx else "")
        code = await llm_caller(prompt, system=system, force_claude=True, max_tokens=4000)
        if "```python" in code: code = code.split("```python")[1].split("```")[0]
        elif "```" in code:     code = code.split("```")[1].split("```")[0]
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
