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
        super().__init__(name, dimension, **kwargs)
        self.metric_name = metric
        self.faiss_metric = faiss.METRIC_L2 if metric == 'l2' else faiss.METRIC_INNER_PRODUCT
        self.index = None

    def build_index(self, vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        if vectors.dtype == np.float32 and vectors.flags["C_CONTIGUOUS"]:
            self.vectors = vectors
        else:
            self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.index = faiss.IndexFlat(self.dimension, self.faiss_metric)
        self.index.add(self.vectors)
        self.index_built = True

    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if not self.index_built:
            raise RuntimeError("Index has not been built yet.")

        query_vector = np.array([query], dtype=np.float32)
        distances, indices = self.index.search(query_vector, k)

        n_vectors = self.index.ntotal
        self.record_operation("ndis", float(n_vectors))

        return distances[0], indices[0]

    def batch_search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if not self.index_built:
            raise RuntimeError("Index has not been built yet.")

        if queries.dtype != np.float32 or not queries.flags["C_CONTIGUOUS"]:
            queries = np.ascontiguousarray(queries, dtype=np.float32)

        distances, indices = self.index.search(queries, k)

        # IndexFlat is a brute-force scan: every query touches every vector.
        n_vectors = self.index.ntotal
        n_queries = queries.shape[0]
        self.record_operation("ndis", float(n_vectors * n_queries))

        return distances, indices
