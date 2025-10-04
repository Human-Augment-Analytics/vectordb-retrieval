try:  # optional dependency
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # type: ignore
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
        if faiss is None:
            # Store friendly string until build time
            self.metric = 'l2' if metric == 'l2' else 'ip'  # type: ignore
        else:
            self.metric = faiss.METRIC_L2 if metric == 'l2' else faiss.METRIC_INNER_PRODUCT
        self.index = None

    def build_index(self, vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Build the Faiss index.

        Args:
            vectors: Vectors to index (n_vectors, dimension)
            metadata: Optional metadata (not used by this algorithm)
        """
        if faiss is None:
            raise ImportError("faiss is required for ExactSearch (faiss-based) but is not installed")
        self.vectors = vectors.astype(np.float32)
        # Convert metric string to FAISS metric if necessary
        metric_kind = self.metric
        if isinstance(metric_kind, str):
            metric_kind = faiss.METRIC_L2 if metric_kind == 'l2' else faiss.METRIC_INNER_PRODUCT
        self.index = faiss.IndexFlat(self.dimension, metric_kind)
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
        queries = queries.astype(np.float32)
        # For flat indexes, v2v ops = nq * ntotal exactly
        nq = int(queries.shape[0])
        ntotal = int(self.index.ntotal) if hasattr(self.index, "ntotal") else 0
        self._last_v2v_ops = nq * ntotal
        self._last_code_distance_ops = None
        return self.index.search(queries, k)
