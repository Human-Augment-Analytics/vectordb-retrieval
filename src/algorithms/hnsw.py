import numpy as np
import faiss
from typing import Any, Dict, List, Optional, Tuple
from .base_algorithm import BaseAlgorithm


class HNSW(BaseAlgorithm):
    """
    Hierarchical Navigable Small World (HNSW) algorithm implementation.
    HNSW is a graph-based algorithm that creates a multi-layer structure for efficient search.
    """

    def __init__(self, name: str, dimension: int, M: int = 16, efConstruction: int = 200,
                 efSearch: int = 100, metric: str = "l2", **kwargs):
        super().__init__(name, dimension, **kwargs)
        self.M = M
        self.efConstruction = efConstruction
        self.efSearch = efSearch
        self.metric = metric
        self.index = None

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

    def _reset_hnsw_stats(self) -> None:
        cvar = getattr(faiss, "cvar", None)
        if cvar is None:
            return
        hnsw_stats = getattr(cvar, "hnsw_stats", None)
        if hnsw_stats is not None:
            hnsw_stats.reset()

    def _collect_hnsw_stats(self) -> None:
        cvar = getattr(faiss, "cvar", None)
        if cvar is None:
            return
        hnsw_stats = getattr(cvar, "hnsw_stats", None)
        if hnsw_stats is not None and getattr(hnsw_stats, "ndis", 0) > 0:
            self.record_operation("ndis", float(hnsw_stats.ndis))
            for attr, key in (("n1", "hnsw_n1"), ("n2", "hnsw_n2"), ("n3", "hnsw_n3")):
                val = getattr(hnsw_stats, attr, None)
                if val is not None:
                    self.record_operation(key, float(val))

    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if not self.index_built:
            raise RuntimeError("Index not built. Call build_index() first.")

        if len(query.shape) == 1:
            query = query.reshape(1, -1)

        if self.metric == "cosine":
            query = query.astype(np.float32, copy=True)
            query_norm = np.linalg.norm(query, axis=1, keepdims=True)
            query = np.divide(query, query_norm, out=np.zeros_like(query), where=query_norm > 0)

        self._reset_hnsw_stats()
        distances, indices = self.index.search(query, k)
        self._collect_hnsw_stats()

        return distances[0], indices[0]

    def batch_search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if not self.index_built:
            raise RuntimeError("Index not built. Call build_index() first.")

        if self.metric == "cosine":
            queries = queries.astype(np.float32, copy=True)
            norms = np.linalg.norm(queries, axis=1, keepdims=True)
            queries = np.divide(queries, norms, out=np.zeros_like(queries), where=norms > 0)

        self._reset_hnsw_stats()
        distances, indices = self.index.search(queries, k)
        self._collect_hnsw_stats()

        return distances, indices
