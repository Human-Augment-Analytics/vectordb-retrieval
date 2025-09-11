from .base_algorithm import BaseAlgorithm
from .exact_search import ExactSearch
from .approximate_search import ApproximateSearch
from .hnsw import HNSW
from typing import Any, Dict, Type

from .base import BaseAlgorithm
from .exact import ExactSearch
from .approximate import ApproximateSearch
from .hnsw import HNSW

# Map algorithm types to their classes
ALGORITHM_REGISTRY = {
    "ExactSearch": ExactSearch,
    "ApproximateSearch": ApproximateSearch,
    "HNSW": HNSW
}

def get_algorithm_instance(algorithm_type: str, dimension: int, **params) -> BaseAlgorithm:
    """
    Factory function to create an algorithm instance based on type and parameters.

    Args:
        algorithm_type: Type of algorithm to create
        dimension: Dimensionality of vectors
        **params: Algorithm-specific parameters

    Returns:
        Algorithm instance
    """
    if algorithm_type not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown algorithm type: {algorithm_type}. Available types: {list(ALGORITHM_REGISTRY.keys())}")

    algorithm_class = ALGORITHM_REGISTRY[algorithm_type]
    
    # For HNSW, we need to provide a name parameter
    if algorithm_type == "HNSW":
        name = params.pop('name', algorithm_type)
        return algorithm_class(name=name, dimension=dimension, **params)
    else:
        return algorithm_class(dimension=dimension, **params)
__all__ = ["BaseAlgorithm", "ExactSearch", "ApproximateSearch", "HNSW"]
