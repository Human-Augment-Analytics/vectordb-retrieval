import numpy as np
try:  # optional dependency
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # type: ignore
from typing import List, Dict, Any, Tuple, Optional
from .base_algorithm import BaseAlgorithm

class HNSW(BaseAlgorithm):
    """
    Hierarchical Navigable Small World (HNSW) algorithm implementation.
    HNSW is a graph-based algorithm that creates a multi-layer structure for efficient search.
    """

    def __init__(self, name: str, dimension: int, M: int = 16, efConstruction: int = 200, 
                 efSearch: int = 100, metric: str = "l2", **kwargs):
        """
        Initialize the HNSW algorithm.

        Args:
            name: Name of the algorithm instance
            dimension: Dimensionality of the vectors
            M: Maximum number of connections per element (default: 16)
            efConstruction: Controls index build quality/time (default: 200)
            efSearch: Controls search accuracy/time (default: 100)
            metric: Distance metric to use ('l2', 'cosine', 'dot')
            **kwargs: Additional parameters
        """
        super().__init__(name, dimension, **kwargs)
        self.M = M
        self.efConstruction = efConstruction
        self.efSearch = efSearch
        self.metric = metric
        self.index = None
        
        # Manually add HNSW specific parameters to the config
        self.config.update({
            'M': self.M,
            'efConstruction': self.efConstruction,
            'efSearch': self.efSearch,
            'metric': self.metric
        })

    def build_index(self, vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Build the HNSW index for the given vectors.

        Args:
            vectors: Vectors to index (n_vectors, dimension)
            metadata: Optional metadata for each vector
        """
        if faiss is None:
            raise ImportError("faiss is required for HNSW but is not installed")
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected vectors of dimension {self.dimension}, got {vectors.shape[1]}")

        self.vectors = vectors
        self.metadata = metadata

        # Determine the metric type
        if self.metric == "cosine":
            # Normalize vectors for cosine similarity
            normalized_vectors = vectors.astype(np.float32, copy=True)
            norms = np.linalg.norm(normalized_vectors, axis=1, keepdims=True)
            normalized_vectors = np.divide(
                normalized_vectors,
                norms,
                out=np.zeros_like(normalized_vectors),
                where=norms > 0,
            )
            metric_type = faiss.METRIC_INNER_PRODUCT
            vectors_to_index = normalized_vectors
        elif self.metric == "dot":
            metric_type = faiss.METRIC_INNER_PRODUCT
            vectors_to_index = vectors
        else:  # Default to L2
            metric_type = faiss.METRIC_L2
            vectors_to_index = vectors

        # Create the HNSW index
        self.index = faiss.IndexHNSWFlat(self.dimension, self.M, metric_type)
        self.index.hnsw.efConstruction = self.efConstruction
        self.index.hnsw.efSearch = self.efSearch

        # Add vectors to the index
        self.index.add(vectors_to_index)
        self.index_built = True

    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the index for the k nearest neighbors of the query vector.

        Args:
            query: Query vector (dimension,)
            k: Number of nearest neighbors to return

        Returns:
            Tuple of (distances, indices)
        """
        if not self.index_built:
            raise RuntimeError("Index not built. Call build_index() first.")
        if faiss is None:
            raise ImportError("faiss is required for HNSW but is not installed")

        # Reshape query to (1, dimension) if needed
        if len(query.shape) == 1:
            query = query.reshape(1, -1)

        # Normalize query for cosine similarity
        if self.metric == "cosine":
            query = query.astype(np.float32, copy=True)
            query_norm = np.linalg.norm(query, axis=1, keepdims=True)
            query = np.divide(query, query_norm, out=np.zeros_like(query), where=query_norm > 0)

        # Perform the search
        distances, indices = self.index.search(query, k)

        return distances[0], indices[0]

    def batch_search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the index for the k nearest neighbors of multiple query vectors.

        Args:
            queries: Query vectors (n_queries, dimension)
            k: Number of nearest neighbors to return

        Returns:
            Tuple of (distances, indices)
        """
        if not self.index_built:
            raise RuntimeError("Index not built. Call build_index() first.")
        if faiss is None:
            raise ImportError("faiss is required for HNSW but is not installed")

        # Normalize queries for cosine similarity
        if self.metric == "cosine":
            queries_copy = queries.astype(np.float32, copy=True)
            norms = np.linalg.norm(queries_copy, axis=1, keepdims=True)
            queries_copy = np.divide(
                queries_copy,
                norms,
                out=np.zeros_like(queries_copy),
                where=norms > 0,
            )
            # Reset FAISS HNSW counters and search
            try:
                from ..utils.faiss_stats import reset_runtime_stats, read_distance_counts
            except Exception:  # pragma: no cover
                reset_runtime_stats = read_distance_counts = None  # type: ignore
            if reset_runtime_stats is not None:
                reset_runtime_stats(self.index)
            distances, indices = self.index.search(queries_copy, k)
        else:
            try:
                from ..utils.faiss_stats import reset_runtime_stats, read_distance_counts
            except Exception:  # pragma: no cover
                reset_runtime_stats = read_distance_counts = None  # type: ignore
            if reset_runtime_stats is not None:
                reset_runtime_stats(self.index)
            distances, indices = self.index.search(queries, k)

        # Expose vector op counts when available
        try:
            from ..utils.faiss_stats import counters_are_v2v, read_distance_counts
        except Exception:  # pragma: no cover
            counters_are_v2v = read_distance_counts = None  # type: ignore
        if read_distance_counts is not None:
            _ivf, hnsw_ndis = read_distance_counts(self.index)
            total_ndis = (hnsw_ndis or 0)
            v2v_flag = (counters_are_v2v(self.index) if counters_are_v2v is not None else True)
            self._last_v2v_ops = int(total_ndis) if v2v_flag else None
            self._last_code_distance_ops = int(total_ndis) if not v2v_flag else None

        return distances, indices
