import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from .base_algorithm import BaseAlgorithm

class ExactSearch(BaseAlgorithm):
    """
    Exact search algorithm using brute force approach.
    This serves as a baseline for accuracy comparison.
    """

    def __init__(self, dimension: int, metric: str = "l2", **kwargs):
        """
        Initialize the exact search algorithm.

        Args:
            dimension: Dimensionality of the vectors
            metric: Distance metric to use ('l2', 'cosine', 'dot')
            **kwargs: Additional parameters
        """
        super().__init__("ExactSearch", dimension, metric=metric, **kwargs)
        self.metric = metric

    def build_index(self, vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Store the vectors for brute force search.

        Args:
            vectors: Vectors to index (n_vectors, dimension)
            metadata: Optional metadata for each vector
        """
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected vectors of dimension {self.dimension}, got {vectors.shape[1]}")

        self.vectors = vectors
        self.metadata = metadata

        # Normalize vectors if using cosine similarity
        if self.metric == "cosine":
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            self.vectors = self.vectors / norms

        self.index_built = True

    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors using brute force.

        Args:
            query: Query vector (dimension,)
            k: Number of nearest neighbors to return

        Returns:
            Tuple of (distances, indices)
        """
        if not self.index_built:
            raise RuntimeError("Index not built. Call build_index() first.")

        if query.shape[0] != self.dimension:
            raise ValueError(f"Expected query of dimension {self.dimension}, got {query.shape[0]}")

        # Reshape query to (1, dimension) if needed
        if len(query.shape) == 1:
            query = query.reshape(1, -1)

        # Calculate distances based on the metric
        if self.metric == "l2":
            distances = np.linalg.norm(self.vectors - query, axis=1)
        elif self.metric == "cosine":
            # Normalize query
            query_norm = query / np.linalg.norm(query, axis=1, keepdims=True)
            # Cosine distance = 1 - cosine similarity
            distances = 1 - np.dot(self.vectors, query_norm.T).flatten()
        elif self.metric == "dot":
            # Negative dot product (higher is better, so we negate to get distances)
            distances = -np.dot(self.vectors, query.T).flatten()
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        # Get top k indices
        if k > len(distances):
            k = len(distances)

        indices = np.argsort(distances)[:k]
        return distances[indices], indices

    def batch_search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors for multiple queries.

        Args:
            queries: Query vectors (n_queries, dimension)
            k: Number of nearest neighbors to return

        Returns:
            Tuple of (distances, indices)
        """
        if not self.index_built:
            raise RuntimeError("Index not built. Call build_index() first.")

        n_queries = queries.shape[0]
        distances = np.zeros((n_queries, k))
        indices = np.zeros((n_queries, k), dtype=np.int64)

        for i in range(n_queries):
            d, idx = self.search(queries[i], k=k)
            distances[i] = d
            indices[i] = idx

        return distances, indices
