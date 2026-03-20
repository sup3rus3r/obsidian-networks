"""
Time Series domain handler — LSTM, Transformer-TS, TCN architectures.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from .base_domain import BaseDomain, TF_CODE_SYSTEM, MUTATION_SYSTEM, MECHANISM_SYSTEM


class TimeSeriesDomain(BaseDomain):
    name = "timeseries"
    supported_architectures = ["lstm_ts", "transformer_ts", "tcn"]
    mutation_operators = [
        "layer_insertion", "attention_variant", "normalization_change",
        "activation_change", "depth_change", "width_change",
    ]
    metrics = ["mse", "mae", "loss", "accuracy", "memory_mb", "inference_time_ms", "training_time_s"]

    base_templates = {
        "lstm_ts": {
            "type"        : "lstm_ts",
            "seq_len"     : 50,
            "n_features"  : 1,
            "forecast_horizon": 10,
            "normalization": "layer_norm",
            "activation"  : "tanh",
            "layers": [
                {"type": "lstm",  "units": 64,  "return_sequences": True,  "dropout": 0.2},
                {"type": "lstm",  "units": 32,  "return_sequences": False, "dropout": 0.2},
                {"type": "dense", "units": 64,  "activation": "relu"},
                {"type": "dense", "units": 10,  "activation": "linear"},  # forecast_horizon outputs
            ],
        },
        "transformer_ts": {
            "type"        : "transformer_ts",
            "seq_len"     : 50,
            "n_features"  : 1,
            "embed_dim"   : 64,
            "forecast_horizon": 10,
            "normalization": "layer_norm",
            "activation"  : "gelu",
            "attention"   : {"type": "multi_head", "num_heads": 4, "key_dim": 16},
            "layers": [
                {"type": "linear_proj",       "embed_dim": 64},
                {"type": "pos_encoding",      "max_len": 50},
                {"type": "transformer_block", "num_layers": 3, "mlp_dim": 128, "dropout": 0.1},
                {"type": "dense",             "units": 10, "activation": "linear"},
            ],
        },
    }

    async def generate_mechanism(self, research_insights: str, llm_caller: Callable) -> list[dict]:
        system = MECHANISM_SYSTEM + "\nDomain: time series forecasting (LSTMs, Transformers, TCNs)."
        prompt = f"Research insights:\n{research_insights}\n\nDerive 3 novel mechanisms for temporal sequence models. JSON array:"
        raw = await llm_caller(prompt, system=system, force_claude=True, max_tokens=1200)
        try:
            start = raw.find("["); end = raw.rfind("]") + 1
            return json.loads(raw[start:end])
        except Exception:
            return [{"name": "temporal_attention", "description": "Attention over time steps", "sympy_expression": "softmax(Q_t @ K_t.T / sqrt(d)) @ V_t"}]

    async def propose_mutations(self, base_arch: str, mechanisms: list[dict], llm_caller: Callable, failed_patterns: list[dict] | None = None) -> list[dict]:
        template    = self.get_base_template(base_arch)
        system      = MUTATION_SYSTEM + f"\nDomain: time series. Available operators: {self.mutation_operators}."
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
            proposals = [{"architecture_name": f"{base_arch}_mutant", "mutations": ["attention_variant"], "rationale": "Default"}]

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
            "\nDOMAIN: Time series forecasting."
            "\nSYNTHETIC DATA: X = np.cumsum(np.random.randn(1000,50,1), axis=1).astype(np.float32); "
            "y = X[:,-10:,0].astype(np.float32) (last 10 steps as forecast target)"
            "\nLOSS: mse. METRICS: mae. EPOCHS: 5."
        )

        prompt = self._build_code_prompt(arch_spec, mechanisms, rationale)
        code = await llm_caller(prompt, system=system, force_claude=True, max_tokens=3000)
        if "```python" in code: code = code.split("```python")[1].split("```")[0]
        elif "```" in code:     code = code.split("```")[1].split("```")[0]
        return code.strip()

    def generate_synthetic_data(self, size: int = 1000, params: dict | None = None) -> Any:
        from agents.synthetic_data import generate_timeseries
        p = params or {}
        return generate_timeseries(size=size, seq_len=p.get("seq_len", 50), seed=p.get("seed", 42))

    async def evaluate(self, checkpoint_path: str, test_data: Any) -> dict:
        try:
            import tensorflow as tf
            import numpy as np
            X_test, y_test = test_data[1], test_data[3]
            model = tf.keras.models.load_model(checkpoint_path)
            y_pred = model.predict(X_test, verbose=0)
            mse = float(np.mean((y_pred - y_test) ** 2))
            mae = float(np.mean(np.abs(y_pred - y_test)))
            return {"mse": mse, "mae": mae, "loss": mse}
        except Exception as e:
            return {"mse": 999.0, "mae": 999.0, "loss": 999.0, "error": str(e)}
