"""
Time Series domain handler — LSTM, Transformer-TS, TCN architectures.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from .base_domain import BaseDomain, TF_CODE_SYSTEM, MUTATION_SYSTEM, MECHANISM_SYSTEM


class TimeSeriesDomain(BaseDomain):
    name = "timeseries"
    supported_architectures = ["lstm_ts", "transformer_ts", "tcn", "s4_ts", "neural_ode_ts", "liquid_ts"]
    mutation_operators = [
        "layer_insertion", "attention_variant", "normalization_change",
        "activation_change", "depth_change", "width_change",
        "free_form", "architecture_crossover",
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
        "s4_ts": {
            "type"          : "s4_ts",
            "seq_len"       : 50,
            "n_features"    : 1,
            "forecast_horizon": 10,
            "state_dim"     : 64,
            "normalization" : "layer_norm",
            "activation"    : "gelu",
            "layers": [
                {"type": "linear_proj",  "units": 64},
                {"type": "s4_block",     "state_dim": 64, "num_layers": 4, "dropout": 0.1},
                {"type": "dense",        "units": 32, "activation": "gelu"},
                {"type": "dense",        "units": 10, "activation": "linear"},
            ],
        },
        "neural_ode_ts": {
            "type"          : "neural_ode_ts",
            "seq_len"       : 50,
            "n_features"    : 1,
            "forecast_horizon": 10,
            "hidden_dim"    : 64,
            "normalization" : "layer_norm",
            "activation"    : "tanh",
            "ode_solver"    : "euler",
            "num_steps"     : 10,
            "layers": [
                {"type": "input_proj", "units": 64},
                {"type": "ode_block",  "hidden_dim": 64, "num_steps": 10, "solver": "euler"},
                {"type": "dense",      "units": 32, "activation": "tanh"},
                {"type": "dense",      "units": 10, "activation": "linear"},
            ],
        },
        "liquid_ts": {
            "type"           : "liquid_ts",
            "seq_len"        : 50,
            "n_features"     : 1,
            "forecast_horizon": 10,
            "reservoir_dim"  : 128,
            "spectral_radius": 0.9,
            "sparsity"       : 0.8,
            "normalization"  : "rms_norm",
            "activation"     : "tanh",
            "layers": [
                {"type": "input_proj",   "units": 32},
                {"type": "liquid_block", "reservoir_dim": 128, "sparsity": 0.8, "spectral_radius": 0.9},
                {"type": "readout",      "units": 64, "activation": "relu"},
                {"type": "dense",        "units": 10, "activation": "linear"},
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

    async def propose_mutations(self, base_arch: str, mechanisms: list[dict], llm_caller: Callable, failed_patterns: list[dict] | None = None, **kwargs) -> list[dict]:
        explored_summary: str | None = kwargs.get("explored_summary")
        template    = self.get_base_template(base_arch)
        system      = MUTATION_SYSTEM + f"\nDomain: time series. Available operators: {self.mutation_operators}."
        failure_ctx = self._format_failure_context(failed_patterns)

        explored_ctx     = (
            f"Already-explored architecture space (avoid these regions — novelty score rewards distance from them):\n{explored_summary}"
            if explored_summary else ""
        )
        task_description = kwargs.get("task_description", "")
        goal_ctx         = f"Research goal: {task_description}" if task_description else ""

        prompt = (
            f"Base architecture:\n{json.dumps(template, indent=2)}"
            f"\n\nMechanisms to implement (use these as the PRIMARY inspiration):\n{json.dumps(mechanisms, indent=2)}"
            + (f"\n\n{goal_ctx}" if goal_ctx else "")
            + (f"\n\n{failure_ctx}" if failure_ctx else "")
            + (f"\n\n{explored_ctx}" if explored_ctx else "")
            + "\n\nPropose 3 mutations. Strongly prefer 'free_form' or 'architecture_crossover' "
              "when the mechanisms suggest ideas beyond standard operators. JSON array:"
        )
        raw = await llm_caller(prompt, system=system, force_claude=True, max_tokens=1800)
        try:
            start = raw.find("["); end = raw.rfind("]") + 1
            proposals = json.loads(raw[start:end])
        except Exception:
            proposals = [{"architecture_name": f"{base_arch}_mutant", "mutations": ["attention_variant"], "rationale": "Default"}]

        from agents.mutations import apply_mutations

        results = []
        for p in proposals:
            mutation_list = p.get("mutations", [])
            spec = apply_mutations(template, mutation_list)

            # Embed mechanism descriptions into the spec so the FAISS novelty index
            # can distinguish free_form / crossover variants from standard ones.
            if "free_form" in mutation_list:
                spec["free_form_description"] = p.get("free_form_description", p.get("rationale", ""))
                spec["mechanism_names"] = [m.get("name", "") for m in mechanisms]

            results.append({
                "architecture_name": p.get("architecture_name", f"{base_arch}_mutant"),
                "base_template"    : base_arch,
                "mutations"        : mutation_list,
                "spec"             : spec,
                "rationale"        : p.get("rationale", ""),
            })
        return results

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
