"""
Language domain handler — Transformer, LSTM, GPT-style architectures.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from .base_domain import BaseDomain, TF_CODE_SYSTEM, MUTATION_SYSTEM, MECHANISM_SYSTEM


class LanguageDomain(BaseDomain):
    name = "language"
    supported_architectures = ["transformer", "lstm", "cnn_text"]
    mutation_operators = [
        "layer_insertion", "attention_variant", "normalization_change",
        "activation_change", "depth_change", "width_change",
    ]
    metrics = ["loss", "perplexity", "accuracy", "memory_mb", "inference_time_ms", "training_time_s"]

    base_templates = {
        "transformer": {
            "type"        : "transformer",
            "vocab_size"  : 10000,
            "seq_len"     : 128,
            "embed_dim"   : 128,
            "normalization": "layer_norm",
            "activation"  : "gelu",
            "attention"   : {"type": "multi_head", "num_heads": 4, "key_dim": 32},
            "layers": [
                {"type": "embedding",    "vocab_size": 10000, "embed_dim": 128},
                {"type": "pos_encoding", "max_len": 128},
                {"type": "transformer_block", "num_layers": 4, "mlp_dim": 256, "dropout": 0.1},
                {"type": "pool",         "pool_type": "global_avg"},
                {"type": "dense",        "units": 5, "activation": "softmax"},
            ],
        },
        "lstm": {
            "type"        : "lstm",
            "vocab_size"  : 10000,
            "seq_len"     : 128,
            "embed_dim"   : 64,
            "normalization": "layer_norm",
            "activation"  : "tanh",
            "layers": [
                {"type": "embedding", "vocab_size": 10000, "embed_dim": 64},
                {"type": "lstm",      "units": 128, "return_sequences": True, "dropout": 0.2},
                {"type": "lstm",      "units": 64,  "return_sequences": False, "dropout": 0.2},
                {"type": "dense",     "units": 5, "activation": "softmax"},
            ],
        },
    }

    async def generate_mechanism(self, research_insights: str, llm_caller: Callable) -> list[dict]:
        system = MECHANISM_SYSTEM + "\nDomain: NLP / language modeling (Transformers, LSTMs, text classification)."
        prompt = f"Research insights:\n{research_insights}\n\nDerive 3 novel mechanisms for language models. JSON array:"
        raw = await llm_caller(prompt, system=system, force_claude=True, max_tokens=1200)
        try:
            start = raw.find("["); end = raw.rfind("]") + 1
            return json.loads(raw[start:end])
        except Exception:
            return [{"name": "relative_position_bias", "description": "Position-relative attention bias", "sympy_expression": "attn(q,k) + pos_bias(i-j)"}]

    async def propose_mutations(self, base_arch: str, mechanisms: list[dict], llm_caller: Callable, failed_patterns: list[dict] | None = None, **kwargs) -> list[dict]:
        template         = self.get_base_template(base_arch)
        system           = MUTATION_SYSTEM + f"\nDomain: language models. Available operators: {self.mutation_operators}."
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
            proposals = [{"architecture_name": f"{base_arch}_mutant", "mutations": ["attention_variant", "depth_change"], "rationale": "Default"}]

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
            "\nDOMAIN: Language / NLP (text classification)."
            "\nSYNTHETIC DATA: X = np.random.randint(1, 10000, (1000, 128)).astype(np.int32); "
            "y = np.random.randint(0, 5, (1000,)).astype(np.int32)"
            "\nLOSS: sparse_categorical_crossentropy. METRICS: accuracy. EPOCHS: 5."
        )
        prompt = self._build_code_prompt(arch_spec, mechanisms, rationale)
        code = await llm_caller(prompt, system=system, force_claude=True, max_tokens=3000)
        if "```python" in code: code = code.split("```python")[1].split("```")[0]
        elif "```" in code:     code = code.split("```")[1].split("```")[0]
        return code.strip()

    def generate_synthetic_data(self, size: int = 1000, params: dict | None = None) -> Any:
        from agents.synthetic_data import generate_text
        p = params or {}
        return generate_text(size=size, seq_len=p.get("seq_len", 128), n_classes=p.get("n_classes", 5), seed=p.get("seed", 42))

    async def evaluate(self, checkpoint_path: str, test_data: Any) -> dict:
        try:
            import tensorflow as tf
            import numpy as np
            X_test, y_test = test_data[1], test_data[3]
            model = tf.keras.models.load_model(checkpoint_path)
            loss, acc = model.evaluate(X_test, y_test, verbose=0)
            perplexity = float(np.exp(loss))
            return {"loss": float(loss), "accuracy": float(acc), "perplexity": perplexity}
        except Exception as e:
            return {"loss": 999.0, "accuracy": 0.0, "perplexity": 999.0, "error": str(e)}
