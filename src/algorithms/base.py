from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Any, Optional

class BaseAlgorithm(ABC):
    """
    Base class for vector retrieval algorithms.
    """

    def __init__(self, dimension: int, metric: str = "l2", **kwargs):
        """
        Initialize the algorithm.

        Args:
            dimension: Dimensionality of vectors
            metric: Distance metric to use (e.g., "l2", "ip")
            **kwargs: Additional algorithm-specific parameters
        """
        self.dimension = dimension
        self.metric = metric
        self.parameters = {
            "dimension": dimension,
            "metric": metric,
            **kwargs
        }
        self.index = None

    @abstractmethod
    def build_index(self, vectors: np.ndarray) -> None:
        """
        Build the search index from a set of vectors.

        Args:
            vectors: Vector data to index (n_vectors × dimension)
        """
        pass

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> np.ndarray:
        """
        Find the k nearest neighbors for a single query vector.

        Args:
            query: Query vector
            k: Number of nearest neighbors to retrieve

        Returns:
            Array of indices of the nearest neighbors
        """
        pass

    def batch_search(self, queries: np.ndarray, k: int) -> List[np.ndarray]:
        """
        Find the k nearest neighbors for multiple query vectors.
        Default implementation calls search() for each query.

        Args:
            queries: Query vectors (n_queries × dimension)
            k: Number of nearest neighbors to retrieve

        Returns:
            List of arrays containing indices of nearest neighbors for each query
        """
        results = []
        for i in range(queries.shape[0]):
            results.append(self.search(queries[i], k))
        return results

    def get_memory_usage(self) -> float:
        """
        Get the memory usage of the index in MB.
        Default implementation returns an estimate based on the index type.

        Returns:
            Estimated memory usage in MB
        """
        return 0.0  # Override in subclasses

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the parameters of the algorithm.

        Returns:
            Dictionary of algorithm parameters
        """
        return self.parameters
