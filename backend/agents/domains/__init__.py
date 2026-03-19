# Domain handlers for autonomous research mode
from .vision import VisionDomain
from .language import LanguageDomain
from .audio import AudioDomain
from .timeseries import TimeSeriesDomain
from .graph import GraphDomain
from .multimodal import MultimodalDomain
from .tabular import TabularDomain
from .recommendation import RecommendationDomain
from .generative import GenerativeDomain

DOMAIN_REGISTRY = {
    "vision"        : VisionDomain,
    "language"      : LanguageDomain,
    "audio"         : AudioDomain,
    "timeseries"    : TimeSeriesDomain,
    "graph"         : GraphDomain,
    "multimodal"    : MultimodalDomain,
    "tabular"       : TabularDomain,
    "recommendation": RecommendationDomain,
    "generative"    : GenerativeDomain,
}


def get_domain(name: str):
    """Instantiate and return a domain handler by name."""
    cls = DOMAIN_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown domain: {name!r}. Available: {list(DOMAIN_REGISTRY)}")
    return cls()
