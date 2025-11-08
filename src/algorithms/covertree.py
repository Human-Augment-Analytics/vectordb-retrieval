from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from itertools import count
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .base_algorithm import BaseAlgorithm


@dataclass(slots=True)
class _CoverTreeNode:
    """Node within the cover tree structure."""

    index: int
    level: int
    children: List["_CoverTreeNode"] = field(default_factory=list)


class CoverTree(BaseAlgorithm):
    """
    Lightweight Cover Tree implementation compatible with the benchmarking stack.

    The implementation keeps the original toy insertion logic but adds:
      * Support for the BaseAlgorithm interface (build/search/batch_search).
      * Configurable metrics (L2, cosine, inner product).
      * Candidate pooling to keep search time reasonable for smoke benchmarks.
    """

    def __init__(
        self,
        name: str,
        dimension: int,
        metric: str = "l2",
        candidate_pool_size: int = 256,
        max_visit_nodes: Optional[int] = None,
        visit_multiplier: int = 16,
        **kwargs,
    ) -> None:
        super().__init__(name, dimension, **kwargs)
        self.metric_name = metric.lower()
        self.candidate_pool_size = max(int(candidate_pool_size), 1)
        self.max_visit_nodes = (
            int(max_visit_nodes) if max_visit_nodes is not None else self.candidate_pool_size * visit_multiplier
        )
        self.visit_multiplier = max(visit_multiplier, 1)
        self.root: Optional[_CoverTreeNode] = None
        self.max_level = 0
        self._vectors: Optional[np.ndarray] = None
        self._working_vectors: Optional[np.ndarray] = None

        self.config.update(
            {
                "metric": self.metric_name,
                "candidate_pool_size": self.candidate_pool_size,
                "max_visit_nodes": self.max_visit_nodes,
                "visit_multiplier": self.visit_multiplier,
            }
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_index(
        self,
        vectors: np.ndarray,
        metadata: Optional[Sequence[dict]] = None,
    ) -> None:
        """
        Build the cover tree index over the provided vectors.
        """
        processed = self._prepare_vectors(vectors)
        self.vectors = processed
        self.metadata = list(metadata) if metadata is not None else None

        if self.metric_name == "cosine":
            self._working_vectors = self._normalize_vectors(processed)
        else:
            self._working_vectors = processed

        self.root = None
        self.max_level = 0

        for idx in range(processed.shape[0]):
            self._insert_index(idx)

        self.index_built = True

    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve the k nearest neighbours for a single query vector.
        """
        if not self.index_built or self.root is None or self._working_vectors is None:
            raise RuntimeError("Index has not been built yet.")

        prepared = self._prepare_query(query)
        candidate_indices = self._collect_candidates(prepared, max(k, self.candidate_pool_size))

        if len(candidate_indices) < k:
            candidate_indices = list(range(self._working_vectors.shape[0]))

        distances, indices = self._rank_candidates(prepared, candidate_indices, k)
        return distances, indices

    def batch_search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batched version of the CoverTree search API.
        """
        if not self.index_built or self.root is None or self._working_vectors is None:
            raise RuntimeError("Index has not been built yet.")

        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self.dimension:
            raise ValueError(f"Expected queries of shape (n, {self.dimension})")

        distance_results = np.full((queries.shape[0], k), np.inf, dtype=np.float32)
        index_results = np.full((queries.shape[0], k), -1, dtype=np.int64)

        for row, query in enumerate(queries):
            distances, indices = self.search(query, k=k)
            limit = min(k, len(distances))
            distance_results[row, :limit] = distances[:limit]
            index_results[row, :limit] = indices[:limit]

        return distance_results, index_results

    # ------------------------------------------------------------------
    # Tree construction helpers
    # ------------------------------------------------------------------
    def _prepare_vectors(self, vectors: np.ndarray) -> np.ndarray:
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("Input vectors must be a 2D array")
        if arr.shape[1] != self.dimension:
            raise ValueError(f"Expected vectors with dimension {self.dimension}, got {arr.shape[1]}")
        return np.ascontiguousarray(arr)

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms

    def _prepare_query(self, query: np.ndarray) -> np.ndarray:
        arr = np.asarray(query, dtype=np.float32).reshape(-1)
        if arr.size != self.dimension:
            raise ValueError(f"Query vector must have dimension {self.dimension}")
        if self.metric_name == "cosine":
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
        return arr

    def _insert_index(self, idx: int) -> None:
        if self.root is None:
            self.root = _CoverTreeNode(index=idx, level=0)
            self.max_level = 0
            return

        queue = [self.root]
        level = self.max_level

        while True:
            if self._insert(idx, queue, level):
                return

            new_root = _CoverTreeNode(index=self.root.index, level=level + 1)
            new_root.children.append(self.root)
            self.root = new_root
            level += 1
            self.max_level = level
            queue = [self.root]

    def _insert(self, idx: int, queue: Iterable[_CoverTreeNode], level: int) -> bool:
        assert self._working_vectors is not None
        distance_threshold = 2.0 ** level
        queue = list(queue)

        children: List[_CoverTreeNode] = []
        for node in queue:
            children.extend(node.children)

        filtered = [
            child
            for child in children
            if self._distance_indices(idx, child.index) <= distance_threshold
        ]

        if filtered and self._insert(idx, filtered, level - 1):
            return True

        for node in queue:
            if self._distance_indices(idx, node.index) <= distance_threshold:
                new_node = _CoverTreeNode(index=idx, level=level - 1)
                node.children.append(new_node)
                return True

        return False

    # ------------------------------------------------------------------
    # Search helpers
    # ------------------------------------------------------------------
    def _collect_candidates(self, query: np.ndarray, max_candidates: int) -> List[int]:
        assert self.root is not None
        assert self._working_vectors is not None

        max_candidates = min(max_candidates, self._working_vectors.shape[0])

        heap: List[Tuple[float, int, _CoverTreeNode]] = []
        cache: dict[int, float] = {}
        counter = count()

        def push(node: _CoverTreeNode, distance: Optional[float] = None) -> None:
            if distance is None:
                distance = self._distance_to_query(query, node.index)
            cache[id(node)] = distance
            radius = 2.0 ** node.level
            lower_bound = max(0.0, distance - radius)
            heapq.heappush(heap, (lower_bound, next(counter), node))

        push(self.root)
        visited = 0
        candidates: List[int] = []
        seen_indices: set[int] = set()
        visit_budget = min(self.max_visit_nodes, self._working_vectors.shape[0])

        while heap and visited < visit_budget and len(candidates) < max_candidates:
            _, _, node = heapq.heappop(heap)
            visited += 1

            node_distance = cache.pop(id(node), None)
            if node_distance is None:
                node_distance = self._distance_to_query(query, node.index)

            if node.index not in seen_indices:
                seen_indices.add(node.index)
                candidates.append(node.index)

            for child in node.children:
                push(child)

        if not candidates:
            return list(range(self._working_vectors.shape[0]))
        return candidates

    def _rank_candidates(
        self,
        query: np.ndarray,
        candidate_indices: Sequence[int],
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert self._working_vectors is not None
        candidate_array = np.asarray(candidate_indices, dtype=np.int64)
        if candidate_array.size == 0:
            return (
                np.full(k, np.inf, dtype=np.float32),
                np.full(k, -1, dtype=np.int64),
            )

        distances = self._compute_distances(query, candidate_array)

        order = np.argsort(distances)
        limit = min(k, order.size)

        top_distances = np.full(k, np.inf, dtype=np.float32)
        top_indices = np.full(k, -1, dtype=np.int64)

        top_distances[:limit] = distances[order[:limit]]
        top_indices[:limit] = candidate_array[order[:limit]]
        return top_distances, top_indices

    def _compute_distances(self, query: np.ndarray, candidate_array: np.ndarray) -> np.ndarray:
        assert self._working_vectors is not None
        candidates = self._working_vectors[candidate_array]

        if self.metric_name in ("l2", "euclidean"):
            diffs = candidates - query
            distances = np.linalg.norm(diffs, axis=1)
        elif self.metric_name == "cosine":
            distances = 1.0 - np.dot(candidates, query)
        elif self.metric_name in ("dot", "ip", "inner_product"):
            distances = -np.dot(candidates, query)
        else:
            distances = np.array(
                [self._metric_fn(query, vec) for vec in candidates],
                dtype=np.float32,
            )
        return distances.astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------
    def _distance_indices(self, idx_a: int, idx_b: int) -> float:
        assert self._working_vectors is not None
        return float(self._metric_fn(self._working_vectors[idx_a], self._working_vectors[idx_b]))

    def _distance_to_query(self, query: np.ndarray, idx: int) -> float:
        assert self._working_vectors is not None
        return float(self._metric_fn(query, self._working_vectors[idx]))

    def _metric_fn(self, a: np.ndarray, b: np.ndarray) -> float:  # type: ignore[override]
        if self.metric_name in ("l2", "euclidean"):
            return float(np.linalg.norm(a - b))
        if self.metric_name == "cosine":
            return float(1.0 - np.dot(a, b))
        if self.metric_name in ("dot", "ip", "inner_product"):
            return float(-np.dot(a, b))
        raise ValueError(f"Unsupported metric: {self.metric_name}")
