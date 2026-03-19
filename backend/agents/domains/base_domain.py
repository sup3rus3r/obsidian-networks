"""
BaseDomain — abstract interface all domain handlers must implement.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable


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
    ) -> str:
        """
        Generate executable Python training code for an architecture spec.
        Code must be self-contained and use synthetic data internally.
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
        """Return a deep copy of a base architecture template."""
        import copy
        template = self.base_templates.get(arch_name)
        if template is None:
            raise ValueError(f"Unknown architecture '{arch_name}' for domain '{self.name}'")
        return copy.deepcopy(template)

    def list_architectures(self) -> list[str]:
        return list(self.base_templates.keys())
