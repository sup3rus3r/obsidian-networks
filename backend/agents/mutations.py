"""
Mutation operators for architecture evolution.

Each operator takes an architecture spec dict and returns a mutated copy.
The ArchitectAgent uses these to propose variants of base templates.
"""
from __future__ import annotations

import copy
import random
from abc import ABC, abstractmethod
from typing import Any


# ── Base class ────────────────────────────────────────────────────────────────

class MutationOperator(ABC):
    name: str
    description: str

    @abstractmethod
    def apply(self, spec: dict[str, Any], rng: random.Random | None = None) -> dict[str, Any]:
        """Return a mutated copy of the architecture spec."""
        pass

    def __call__(self, spec: dict[str, Any], rng: random.Random | None = None) -> dict[str, Any]:
        return self.apply(copy.deepcopy(spec), rng)


# ── Concrete operators ────────────────────────────────────────────────────────

class LayerInsertion(MutationOperator):
    """Insert an extra layer (Dense/Conv/Attention) at a random depth."""
    name = "layer_insertion"
    description = "Insert an additional layer at a random position in the network"

    def apply(self, spec: dict, rng: random.Random | None = None) -> dict:
        rng = rng or random.Random()
        layers = spec.get("layers", [])
        if not layers:
            return spec
        pos = rng.randint(0, len(layers) - 1)
        base_units = layers[pos].get("units", layers[pos].get("filters", 64))
        new_layer = {
            "type"      : layers[pos].get("type", "dense"),
            "units"     : int(base_units * rng.choice([0.5, 1.0, 2.0])),
            "activation": rng.choice(["relu", "gelu", "swish"]),
            "dropout"   : round(rng.uniform(0.0, 0.4), 2),
        }
        layers.insert(pos + 1, new_layer)
        spec["layers"] = layers
        spec["mutation"] = self.name
        return spec


class KernelSizeChange(MutationOperator):
    """Change the kernel size of convolutional layers."""
    name = "kernel_size_change"
    description = "Mutate Conv layer kernel sizes (3→5→7 or reverse)"

    _OPTIONS = [1, 3, 5, 7]

    def apply(self, spec: dict, rng: random.Random | None = None) -> dict:
        rng = rng or random.Random()
        for layer in spec.get("layers", []):
            if layer.get("type") in ("conv2d", "conv1d", "conv3d", "depthwise_conv"):
                current = layer.get("kernel_size", 3)
                options = [k for k in self._OPTIONS if k != current]
                layer["kernel_size"] = rng.choice(options)
        spec["mutation"] = self.name
        return spec


class SkipConnectionAdd(MutationOperator):
    """Add a residual/skip connection between layers."""
    name = "skip_connection_add"
    description = "Add residual connections between non-adjacent layers"

    def apply(self, spec: dict, rng: random.Random | None = None) -> dict:
        rng = rng or random.Random()
        layers = spec.get("layers", [])
        if len(layers) < 3:
            return spec
        skip_from = rng.randint(0, len(layers) - 3)
        skip_to   = rng.randint(skip_from + 2, len(layers) - 1)
        spec.setdefault("skip_connections", []).append({
            "from": skip_from,
            "to"  : skip_to,
        })
        spec["mutation"] = self.name
        return spec


class AttentionVariant(MutationOperator):
    """Swap attention mechanism type or change number of heads."""
    name = "attention_variant"
    description = "Mutate attention: change head count, use multi-query or grouped-query attention"

    _VARIANTS = ["multi_head", "multi_query", "grouped_query", "linear_attention"]

    def apply(self, spec: dict, rng: random.Random | None = None) -> dict:
        rng = rng or random.Random()
        attn = spec.get("attention", {})
        if not attn:
            # Add attention block to spec
            spec["attention"] = {
                "type"     : rng.choice(self._VARIANTS),
                "num_heads": rng.choice([4, 8, 16]),
                "key_dim"  : rng.choice([32, 64, 128]),
            }
        else:
            attn["type"]      = rng.choice([v for v in self._VARIANTS if v != attn.get("type")])
            attn["num_heads"] = rng.choice([4, 8, 16])
        spec["mutation"] = self.name
        return spec


class NormalizationChange(MutationOperator):
    """Swap normalization layer type."""
    name = "normalization_change"
    description = "Switch between BatchNorm, LayerNorm, GroupNorm, RMSNorm"

    _OPTIONS = ["batch_norm", "layer_norm", "group_norm", "rms_norm", "instance_norm"]

    def apply(self, spec: dict, rng: random.Random | None = None) -> dict:
        rng = rng or random.Random()
        current = spec.get("normalization", "batch_norm")
        options = [n for n in self._OPTIONS if n != current]
        spec["normalization"] = rng.choice(options)
        spec["mutation"] = self.name
        return spec


class ActivationChange(MutationOperator):
    """Change activation functions globally."""
    name = "activation_change"
    description = "Swap activation functions (ReLU → GELU → Swish → Mish)"

    _OPTIONS = ["relu", "gelu", "swish", "mish", "elu", "selu"]

    def apply(self, spec: dict, rng: random.Random | None = None) -> dict:
        rng = rng or random.Random()
        current = spec.get("activation", "relu")
        new_act = rng.choice([a for a in self._OPTIONS if a != current])
        spec["activation"] = new_act
        for layer in spec.get("layers", []):
            if "activation" in layer:
                layer["activation"] = new_act
        spec["mutation"] = self.name
        return spec


class DepthChange(MutationOperator):
    """Add or remove layers to change network depth."""
    name = "depth_change"
    description = "Increase or decrease network depth by ±1-2 layers"

    def apply(self, spec: dict, rng: random.Random | None = None) -> dict:
        rng = rng or random.Random()
        layers = spec.get("layers", [])
        if not layers:
            return spec
        delta = rng.choice([-2, -1, 1, 2])
        if delta > 0:
            # Duplicate last hidden layer
            for _ in range(delta):
                new_layer = copy.deepcopy(layers[-2]) if len(layers) > 1 else copy.deepcopy(layers[-1])
                layers.insert(-1, new_layer)
        else:
            # Remove layers (keep at least 2)
            to_remove = min(abs(delta), len(layers) - 2)
            for _ in range(to_remove):
                if len(layers) > 2:
                    layers.pop(-2)
        spec["layers"] = layers
        spec["mutation"] = self.name
        return spec


class WidthChange(MutationOperator):
    """Scale the width (units/filters) of all hidden layers."""
    name = "width_change"
    description = "Scale hidden layer widths by a factor (0.5×, 1.5×, 2×)"

    def apply(self, spec: dict, rng: random.Random | None = None) -> dict:
        rng = rng or random.Random()
        factor = rng.choice([0.5, 1.5, 2.0])
        for layer in spec.get("layers", [])[:-1]:  # skip output layer
            for key in ("units", "filters", "dim"):
                if key in layer:
                    layer[key] = max(8, int(layer[key] * factor))
        spec["mutation"] = self.name
        return spec


class FusionTypeChange(MutationOperator):
    """Change how modalities or branches are fused (for multimodal)."""
    name = "fusion_type_change"
    description = "Change multimodal fusion strategy (concat, add, cross-attention, gated)"

    _OPTIONS = ["concatenate", "add", "cross_attention", "gated", "bilinear"]

    def apply(self, spec: dict, rng: random.Random | None = None) -> dict:
        rng = rng or random.Random()
        current = spec.get("fusion", "concatenate")
        spec["fusion"] = rng.choice([f for f in self._OPTIONS if f != current])
        spec["mutation"] = self.name
        return spec


class FreeFormMutation(MutationOperator):
    """Invent a novel structural idea not representable by any standard operator.

    The LLM proposes a free-form description; the Coder implements it as a
    custom tf.keras.layers.Layer.  The spec carries the description so the
    FAISS novelty index can distinguish it from standard variants.
    """
    name = "free_form"
    description = (
        "Invent a completely new architectural structure guided by the mechanism "
        "descriptions — not derivable from any existing operator"
    )

    def apply(self, spec: dict, rng: random.Random | None = None) -> dict:
        spec["mutation"] = self.name
        spec.setdefault("free_form_structure", "novel mechanism — see rationale and mechanisms")
        return spec


class ArchitectureCrossover(MutationOperator):
    """Hybridize the base architecture with a structurally different paradigm.

    Adds a 'crossover_paradigm' field to the spec so the Coder knows which
    foreign architectural concept to graft in, and the FAISS index sees the
    hybrid identity.
    """
    name = "architecture_crossover"
    description = "Hybridize the base architecture with elements from a structurally different computational paradigm"

    _PARADIGMS = [
        "state_space_model",
        "neural_ode",
        "reservoir_computing",
        "capsule_network",
        "hyperbolic_geometry",
        "graph_message_passing",
        "fourier_neural_operator",
        "liquid_time_constant",
    ]

    def apply(self, spec: dict, rng: random.Random | None = None) -> dict:
        rng = rng or random.Random()
        spec["crossover_paradigm"] = rng.choice(self._PARADIGMS)
        spec["mutation"] = self.name
        return spec


# ── Registry ──────────────────────────────────────────────────────────────────

MUTATION_REGISTRY: dict[str, MutationOperator] = {
    op.name: op()
    for op in [
        LayerInsertion,
        KernelSizeChange,
        SkipConnectionAdd,
        AttentionVariant,
        NormalizationChange,
        ActivationChange,
        DepthChange,
        WidthChange,
        FusionTypeChange,
        FreeFormMutation,
        ArchitectureCrossover,
    ]
}


def apply_mutations(
    spec: dict,
    mutation_names: list[str],
    rng: random.Random | None = None,
) -> dict:
    """Apply a sequence of named mutations to an architecture spec."""
    result = copy.deepcopy(spec)
    for name in mutation_names:
        op = MUTATION_REGISTRY.get(name)
        if op:
            result = op(result, rng)
    return result


def random_mutations(
    spec: dict,
    n: int = 2,
    domain_operators: list[str] | None = None,
    rng: random.Random | None = None,
) -> dict:
    """Apply n random mutations, optionally restricted to domain-specific operators."""
    rng      = rng or random.Random()
    pool     = domain_operators or list(MUTATION_REGISTRY.keys())
    selected = rng.sample(pool, min(n, len(pool)))
    return apply_mutations(spec, selected, rng)
