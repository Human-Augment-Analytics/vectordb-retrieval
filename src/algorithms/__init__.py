from typing import Any, Dict, Type

# DiskANN support has been removed; keep registry limited to active algorithms.

from .approximate_search import ApproximateSearch
from .base_algorithm import BaseAlgorithm
from .covertree import CoverTree
from .exact_search import ExactSearch
from .hnsw import HNSW
from .lsh import LSH
from .modular import (
    BaseIndexer,
    BaseSearcher,
    CompositeAlgorithm,
    INDEXER_REGISTRY,
    SEARCHER_REGISTRY,
    get_indexer_class,
    get_searcher_class,
    register_indexer,
    register_searcher,
)


# Map algorithm types to their classes
ALGORITHM_REGISTRY: Dict[str, Type[BaseAlgorithm]] = {
    "ExactSearch": ExactSearch,
    "ApproximateSearch": ApproximateSearch,
    "HNSW": HNSW,
    "LSH": LSH,
    "CoverTree": CoverTree,
    "Composite": CompositeAlgorithm,
    "CompositeAlgorithm": CompositeAlgorithm,
    "Modular": CompositeAlgorithm,
}


def get_algorithm_instance(algorithm_type: str, dimension: int, **params: Any) -> BaseAlgorithm:
    """Factory function to create an algorithm instance based on type and parameters."""

    if algorithm_type not in ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown algorithm type: {algorithm_type}. Available types: {list(ALGORITHM_REGISTRY.keys())}"
        )

    algorithm_class = ALGORITHM_REGISTRY[algorithm_type]
    name = params.pop("name", algorithm_type)
    return algorithm_class(name=name, dimension=dimension, **params)


__all__ = [
    "BaseAlgorithm",
    "ExactSearch",
    "ApproximateSearch",
    "HNSW",
    "LSH",
    "CoverTree",
    "CompositeAlgorithm",
    "BaseIndexer",
    "BaseSearcher",
    "register_indexer",
    "register_searcher",
    "INDEXER_REGISTRY",
    "SEARCHER_REGISTRY",
    "get_indexer_class",
    "get_searcher_class",
]
