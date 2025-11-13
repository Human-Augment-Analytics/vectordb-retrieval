import numpy as np
from typing import Dict, List, Any
from .base import BaseAlgorithm

class ExactSearch(BaseAlgorithm):
    """
    Exact (brute force) nearest neighbor search.
    """

    def __init__(self, dimension: int, metric: str = "l2", **kwargs):
        """
        Initialize the exact search algorithm.

        Args:
            dimension: Dimensionality of vectors
            metric: Distance metric to use (e.g., "l2", "ip")
            **kwargs: Additional parameters (ignored for exact search)
        """
        super().__init__(dimension, metric, **kwargs)

    def build_index(self, vectors: np.ndarray) -> None:
        """
        Build the search index (just store the vectors for exact search).

        Args:
            vectors: Vector data to index (n_vectors × dimension)
        """
        self.index = vectors

    def search(self, query: np.ndarray, k: int) -> np.ndarray:
        """
        Find the k nearest neighbors for a single query vector.

        Args:
            query: Query vector
            k: Number of nearest neighbors to retrieve

        Returns:
            Array of indices of the nearest neighbors
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Compute distances based on metric
        if self.metric == "l2":
            # L2 distance
            distances = np.linalg.norm(self.index - query, axis=1)
        elif self.metric == "ip":
            # Inner product (negative for nearest neighbors)
            distances = -np.dot(self.index, query)
        elif self.metric == "cosine":
            # Cosine similarity (negative for nearest neighbors)
            norm_query = query / np.linalg.norm(query)
            norm_vectors = self.index / np.linalg.norm(self.index, axis=1, keepdims=True)
            distances = -np.dot(norm_vectors, norm_query)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        # Get top k indices
        k = min(k, len(distances))  # Handle case where k > number of vectors
        indices = np.argsort(distances)[:k]

        self.record_operation("search_ops", float(self.index.shape[0]), source="python.bruteforce")

        return indices

    def batch_search(self, queries: np.ndarray, k: int) -> List[np.ndarray]:
        """
        Find the k nearest neighbors for multiple query vectors.

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

        Returns:
            Memory usage in MB
        """
        if self.index is None:
            return 0.0

        # Calculate memory usage: bytes per float * number of elements / (1024^2)
        return self.index.nbytes / (1024 * 1024)
