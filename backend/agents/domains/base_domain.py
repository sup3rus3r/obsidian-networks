"""
BaseDomain — abstract interface all domain handlers must implement.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

# Shared system prompt for all generate_code calls — large enough (>1024 tokens)
# to trigger Anthropic prompt caching. Identical across all domains so the KV
# state is cached server-side for the full 5-minute TTL.
TF_CODE_SYSTEM = """\
You are an expert machine learning engineer specializing in TensorFlow 2.x and Keras \
architecture research. Your sole task is to generate complete, self-contained, \
executable Python training scripts from architecture specifications.

TENSORFLOW / KERAS CODING RULES — follow exactly:

1. Imports
   - Always begin: `import tensorflow as tf` then `import numpy as np`.
   - Access Keras only as `tf.keras.*`. Never write `import keras` standalone.
   - Allowed third-party imports: tensorflow, numpy, os, math, json, sklearn \
(make_classification / preprocessing only). Nothing else.

2. Model construction
   - Use the tf.keras Functional API unless the spec explicitly says Sequential.
   - Pattern: inputs = tf.keras.Input(shape=...); x = Layer()(inputs); \
model = tf.keras.Model(inputs, outputs, name='model')
   - For multi-input models pass a list to tf.keras.Model: \
model = tf.keras.Model(inputs=[inp_a, inp_b], outputs=out)

3. Data types
   - All input arrays: float32. Cast explicitly: X = X.astype(np.float32)
   - Integer label arrays: int32. Cast explicitly: y = y.astype(np.int32)
   - Regression targets: float32.

4. Synthetic data
   - All training data must be generated inside the script using numpy or \
tensorflow random ops. No file reads, no downloads, no external datasets.
   - Use np.random.seed(42) at the top for reproducibility.

5. Output directory
   - Always create before saving: import os; os.makedirs('output', exist_ok=True)
   - Save the final model: model.save('output/model.keras') — exact path required.
   - Do not save intermediate checkpoints; only the final model.

6. Training loop
   - Use model.fit(X_train, y_train, validation_split=0.2, epochs=N, \
batch_size=B, verbose=1, callbacks=[early_stop])
   - Always include EarlyStopping: \
tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, \
restore_best_weights=True)
   - Print training results with verbose=1 so metrics appear in stdout.

7. Compilation
   - optimizer: tf.keras.optimizers.Adam(learning_rate=1e-3) unless spec says otherwise.
   - loss: sparse_categorical_crossentropy for integer multi-class labels; \
binary_crossentropy for binary; mse for regression.
   - metrics: ['accuracy'] for classification; ['mae'] for regression.

8. Parameter efficiency
   - Keep total trainable parameters under 10 million for synthetic datasets < 2000 \
samples — smaller models train faster and generalize better on limited data.
   - Add Dropout (rate 0.2–0.4) between Dense layers with > 64 units.
   - Add BatchNormalization or LayerNormalization where the spec indicates.

9. FORBIDDEN — never include:
   - subprocess, os.system, os.popen, or any shell commands.
   - requests, urllib, httpx, or any network calls.
   - matplotlib, seaborn, plotly, or any visualization imports.
   - pip install statements in any form.
   - File reads: open(), pd.read_csv(), np.load() for external files.
   - model.summary() or verbose model printing beyond epoch logs.

10. Output format
    - Return ONLY valid Python source code.
    - No markdown fences (no ```python blocks).
    - No prose explanations before or after the code.
    - No inline comments explaining the architecture — code must be clean and minimal.
    - The script must run successfully with: python script.py
"""

# Shared system prompt for propose_mutations / generate_mechanism calls.
# Shorter — caching benefit is secondary here but still applied.
MUTATION_SYSTEM = """\
You are a neural architecture search expert. Your task is to propose \
meaningful architectural mutations that incorporate novel mechanisms into \
existing base architectures. Output ONLY valid JSON — no prose, no markdown.

JSON output rules:
- Return a valid JSON array.
- All string values must use double quotes.
- No trailing commas.
- architecture_name: descriptive snake_case identifier for the mutated architecture.
- mutations: list of operator names from the provided operator set.
- rationale: one sentence tying the mutation to a specific mechanism.
"""

MECHANISM_SYSTEM = """\
You are an expert ML researcher. Your task is to derive novel mathematical \
mechanisms from research insights that can be applied to neural architecture \
design. Output ONLY valid JSON — no prose, no markdown.

JSON output rules:
- Return a valid JSON array of mechanism objects.
- name: short snake_case identifier.
- description: one sentence explaining the mechanism and why it helps.
- sympy_expression: a valid mathematical expression in sympy syntax \
representing the core computation.
"""


class BaseDomain(ABC):
    """Abstract base class for all domain handlers."""

    # Subclasses must define these class-level attributes
    name: str
    supported_architectures: list[str]
    base_templates: dict[str, dict]      # arch_name → spec template dict
    mutation_operators: list[str]        # names from mutations.MUTATION_REGISTRY
    metrics: list[str]                   # domain-specific metric names

    @abstractmethod
    async def generate_mechanism(
        self,
        research_insights: str,
        llm_caller: Callable[[str], Any],
    ) -> list[dict]:
        """
        Derive novel mechanisms from research insights via LLM.

        Returns list of dicts: [{"name": str, "description": str, "sympy_expression": str}]
        """
        pass

    @abstractmethod
    async def propose_mutations(
        self,
        base_arch: str,
        mechanisms: list[dict],
        llm_caller: Callable[[str], Any],
        failed_patterns: list[dict] | None = None,
    ) -> list[dict]:
        """
        Propose architecture mutations based on mechanisms.

        Returns list of mutation specs: [{"architecture_name": str, "mutations": [str], "spec": dict, "rationale": str}]
        """
        pass

    @abstractmethod
    async def generate_code(
        self,
        arch_spec: dict,
        llm_caller: Callable[[str], Any],
        mechanisms: list[dict] | None = None,
        rationale: str | None = None,
    ) -> str:
        """
        Generate executable Python training code for an architecture spec.
        Code must be self-contained and use synthetic data internally.
        mechanisms: mathematical mechanisms derived from papers — implement these in the code.
        rationale: why this architecture was proposed — guides the implementation focus.
        """
        pass

    @abstractmethod
    def generate_synthetic_data(self, size: int = 1000, params: dict | None = None) -> Any:
        """
        Generate domain-appropriate synthetic data.
        Returns whatever format the domain's training code expects.
        """
        pass

    @abstractmethod
    async def evaluate(
        self,
        checkpoint_path: str,
        test_data: Any,
    ) -> dict:
        """
        Load a trained checkpoint and evaluate on test data.
        Returns dict of metric_name → float.
        """
        pass

    def get_base_template(self, arch_name: str) -> dict:
        """Return a deep copy of a base architecture template.

        If the exact name is not found (e.g. 'lstm_mutant' from a previous
        generation's winner), strip common mutation suffixes and try the base
        name before raising.
        """
        import copy
        import re
        template = self.base_templates.get(arch_name)
        if template is None:
            # Strip mutation suffixes: _mutant, _v2, _mutant_v2, etc.
            base_name = re.sub(r'(_mutant)?(_v\d+)?$', '', arch_name).strip('_')
            template = self.base_templates.get(base_name)
        if template is None:
            # Final fallback: use the first available template
            if self.base_templates:
                template = next(iter(self.base_templates.values()))
            else:
                raise ValueError(f"Unknown architecture '{arch_name}' for domain '{self.name}'")
        return copy.deepcopy(template)

    def list_architectures(self) -> list[str]:
        return list(self.base_templates.keys())

    @staticmethod
    def _format_failure_context(failed_patterns: list[dict] | None) -> str:
        """Format previously failed candidates for inclusion in propose_mutations prompt."""
        if not failed_patterns:
            return ""
        lines = []
        for f in failed_patterns:
            name    = f.get("architecture_name", "?")
            score   = f.get("composite_score", 0)
            muts    = f.get("mutations", [])
            why     = f.get("failure_reason", "")
            line = f"  - {name} (score {score:.2f}): mutations={muts}"
            if why:
                line += f" — {why}"
            lines.append(line)
        return (
            "Previously tried architectures that scored poorly — DO NOT repeat these mutation combinations:\n"
            + "\n".join(lines)
            + "\n\nPropose genuinely different mutations that explore new areas."
        )

    @staticmethod
    def _format_mechanism_context(mechanisms: list[dict] | None, rationale: str | None) -> str:
        """Format mechanism and rationale data for inclusion in generate_code user prompt."""
        parts: list[str] = []
        if rationale:
            parts.append(f"Design rationale (what this mutation is trying to achieve):\n{rationale}")
        if mechanisms:
            mech_lines = []
            for m in mechanisms:
                line = f"  - {m.get('name', '?')}: {m.get('description', '')}"
                if m.get("sympy_expression"):
                    line += f"\n    Math: {m['sympy_expression']}"
                mech_lines.append(line)
            parts.append(
                "Mathematical mechanisms derived from research papers — implement these in the code:\n"
                + "\n".join(mech_lines)
                + "\n\nDo NOT just reflect these in comments. Actually implement the specific "
                "mathematical operations described above where architecturally appropriate."
            )
        return "\n\n".join(parts)
