import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from .base_algorithm import BaseAlgorithm
import diskannpy
import time
import os
import tempfile
import shutil

class DiskANN(BaseAlgorithm):
    def __init__(self, name: str, dimension: int, **kwargs):
        super().__init__(name, dimension, **kwargs)
        self.metric = self.config.get("metric", "l2")
        self.vector_dtype_str = self.config.get("vector_dtype", "float32")
        self.vector_dtype = getattr(np, self.vector_dtype_str)

        # Build parameters
        self.max_degree = self.config.get("max_degree", 64)
        self.build_complexity = self.config.get("build_complexity", 750)
        self.build_alpha = self.config.get("build_alpha", 1.2)

        # Search parameters
        self.search_complexity = self.config.get("search_complexity", 100)

        self.num_threads = self.config.get("num_threads", 16)

        self._index_dir = tempfile.mkdtemp()
        self._index_prefix = "diskann_index"

        self.index = None

    def build_index(self, vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        if vectors.dtype != self.vector_dtype:
            vectors = vectors.astype(self.vector_dtype)

        start_time = time.time()

        diskannpy.build_fresh_index(
            data=vectors,
            distance_metric=self.metric,
            vector_dtype=self.vector_dtype,
            index_directory=self._index_dir,
            index_prefix=self._index_prefix,
            complexity=self.build_complexity,
            graph_degree=self.max_degree,
            alpha=self.build_alpha,
            num_threads=self.num_threads,
            use_pq_build=False,
            num_pq_bytes=0,
            use_opq=False
        )

        self.build_time = time.time() - start_time

        index_path = os.path.join(self._index_dir, self._index_prefix)
        self.index = diskannpy.StaticDiskIndex(
            distance_metric=self.metric,
            vector_dtype=self.vector_dtype,
            index_path=index_path,
            num_threads=self.num_threads,
            initial_search_complexity=self.search_complexity
        )

        index_size_bytes = 0
        for filename in os.listdir(self._index_dir):
            if filename.startswith(self._index_prefix):
                index_size_bytes += os.path.getsize(os.path.join(self._index_dir, filename))
        self.index_memory_usage = index_size_bytes / (1024 * 1024)

        self.index_built = True

    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if not self.index_built or self.index is None:
            raise RuntimeError("Index is not built yet.")

        if query.ndim == 1:
            query = np.expand_dims(query, axis=0)

        if query.dtype != self.vector_dtype:
            query = query.astype(self.vector_dtype)

        indices, distances = self.index.search(
            query=query[0],
            k_neighbors=k,
            complexity=self.search_complexity
        )
        return np.array(distances), np.array(indices)

    def batch_search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if not self.index_built or self.index is None:
            raise RuntimeError("Index is not built yet.")

        if queries.dtype != self.vector_dtype:
            queries = queries.astype(self.vector_dtype)

        indices, distances = self.index.batch_search(
            queries=queries,
            k_neighbors=k,
            complexity=self.search_complexity,
            num_threads=self.num_threads
        )
        return distances, indices

    def __del__(self):
        if hasattr(self, '_index_dir') and os.path.exists(self._index_dir):
            shutil.rmtree(self._index_dir)
