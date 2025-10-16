import faiss
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from .base_algorithm import BaseAlgorithm

class ExactSearch(BaseAlgorithm):
    """
    Exact nearest neighbor search using Faiss IndexFlatL2.
    This provides the ground truth for comparison.
    """

    def __init__(self, name: str, dimension: int, metric: str = 'l2', **kwargs):
        """
        Initialize the ExactSearch algorithm.

        Args:
            name: Name of the algorithm instance
            dimension: Dimensionality of the vectors
            metric: Distance metric ('l2' or 'ip')
            **kwargs: Additional parameters
        """
        super().__init__(name, dimension, **kwargs)
        self.metric = faiss.METRIC_L2 if metric == 'l2' else faiss.METRIC_INNER_PRODUCT
        self.index = None

    def build_index(self, vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Build the Faiss index.

        Args:
            vectors: Vectors to index (n_vectors, dimension)
            metadata: Optional metadata (not used by this algorithm)
        """
        if vectors.dtype == np.float32 and vectors.flags["C_CONTIGUOUS"]:
            self.vectors = vectors
        else:
            self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.index = faiss.IndexFlat(self.dimension, self.metric)
        self.index.add(self.vectors)
        self.index_built = True

    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the index for the k nearest neighbors.

        Args:
            query: Query vector (dimension,)
            k: Number of nearest neighbors to return

        Returns:
            Tuple of (distances, indices)
        """
        if not self.index_built:
            raise RuntimeError("Index has not been built yet.")
            
        # Faiss expects a 2D array for queries
        query_vector = np.array([query], dtype=np.float32)
        distances, indices = self.index.search(query_vector, k)
        
        return distances[0], indices[0]

    def batch_search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch search for k nearest neighbors.

        Args:
            queries: Query vectors (n_queries, dimension)
            k: Number of nearest neighbors to return

        Returns:
            Tuple of (distances, indices)
        """
        if not self.index_built:
            raise RuntimeError("Index has not been built yet.")
            
        if queries.dtype != np.float32 or not queries.flags["C_CONTIGUOUS"]:
            queries = np.ascontiguousarray(queries, dtype=np.float32)
        return self.index.search(queries, k)
