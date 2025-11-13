from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class BaseAlgorithm(ABC):
    """
    Base class for all vector retrieval algorithms.
    All new algorithms should inherit from this class and implement its methods.
    """

    def __init__(self, name: str, dimension: int, **kwargs):
        """
        Initialize the algorithm.

        Args:
            name: Name of the algorithm
            dimension: Dimensionality of the vectors
            **kwargs: Additional algorithm-specific parameters
        """
        self.name = name
        self.dimension = dimension
        self.vectors = None
        self.metadata = None
        self.index_built = False
        self.config = kwargs
        self.build_time = -1.0
        self.index_memory_usage = -1.0
        self._operation_counts: Dict[str, Any] = {}
        self.reset_operation_counts()

    @abstractmethod
    def build_index(self, vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Build the index for the given vectors.

        Args:
            vectors: Vectors to index (n_vectors, dimension)
            metadata: Optional metadata for each vector
        """
        pass

    @abstractmethod
    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the index for the k nearest neighbors of the query vector.

        Args:
            query: Query vector (dimension,)
            k: Number of nearest neighbors to return

        Returns:
            Tuple of (distances, indices) where:
                distances: Distances to the k nearest neighbors (k,)
                indices: Indices of the k nearest neighbors (k,)
        """
        pass

    @abstractmethod
    def batch_search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the index for the k nearest neighbors of the query vectors in batch.

        Args:
            queries: Query vectors (n_queries, dimension)
            k: Number of nearest neighbors to return

        Returns:
            Tuple of (distances, indices) where:
                distances: Distances to the k nearest neighbors (n_queries, k)
                indices: Indices of the k nearest neighbors (n_queries, k)
        """
        pass

    def get_name(self) -> str:
        """
        Get the name of the algorithm.

        Returns:
            Name of the algorithm
        """
        return self.name

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the parameters of the algorithm.

        Returns:
            Dictionary of parameters
        """
        return self.config

    def reset_operation_counts(self) -> None:
        """Reset accumulated vector comparison counters."""
        self._operation_counts = {"search_ops": 0.0}

    def record_operation(self, key: str, value: float, source: Optional[str] = None) -> None:
        """Accumulate numeric counters and optionally note their provenance."""
        current = float(self._operation_counts.get(key, 0.0))
        self._operation_counts[key] = current + float(value)
        if source:
            source_key = f"{key}_source"
            existing = self._operation_counts.get(source_key)
            if existing is None:
                self._operation_counts[source_key] = source
            elif existing != source:
                self._operation_counts[source_key] = "mixed"

    def get_operation_counts(self) -> Dict[str, Any]:
        """Return a shallow copy of the accumulated counters."""
        return dict(self._operation_counts)

    def __str__(self) -> str:
        return f"{self.name} (dimension={self.dimension}, parameters={self.config})"
