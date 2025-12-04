from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .base_algorithm import BaseAlgorithm


@dataclass(slots=True)
class _CoverTreeV2Node:
    """Node in the CoverTreeV2 structure."""

    index: int
    level: int
    vector: np.ndarray
    children: List["_CoverTreeV2Node"] = field(default_factory=list)


class CoverTreeV2(BaseAlgorithm):
    """
    Cover Tree variant that mirrors the breadth-first search procedure from the
    `feature/covertree` branch while remaining compatible with the benchmarking stack.

    The tree construction mirrors Algorithm 2 (Insert) from Beygelzimer et al. (2006).
    Search follows the textbook cover-set traversal and then ranks every visited point
    exactly, guaranteeing perfect recall even when we request large `k`.
    """

    def __init__(self, name: str, dimension: int, metric: str = "l2", **kwargs) -> None:
        super().__init__(name, dimension, **kwargs)
        self.metric_name = metric.lower()
        self.root: Optional[_CoverTreeV2Node] = None
        self.max_level = 0
        self._vectors: Optional[np.ndarray] = None
        self._working_vectors: Optional[np.ndarray] = None

        self.config.update({"metric": self.metric_name})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_index(
        self,
        vectors: np.ndarray,
        metadata: Optional[Sequence[dict]] = None,
    ) -> None:
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
        if not self.index_built or self.root is None or self._working_vectors is None:
            raise RuntimeError("Index has not been built yet.")

        prepared = self._prepare_query(query)
        candidate_indices = self._cover_search(prepared)
        return self._rank_candidates(prepared, candidate_indices, k)

    def batch_search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
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

    def _make_node(self, idx: int, level: int) -> _CoverTreeV2Node:
        assert self._working_vectors is not None
        return _CoverTreeV2Node(index=idx, level=level, vector=self._working_vectors[idx])

    def _insert_index(self, idx: int) -> None:
        if self.root is None:
            self.root = self._make_node(idx, level=0)
            self.max_level = 0
            return

        queue = [self.root]
        level = self.max_level

        while True:
            if self._insert(idx, queue, level):
                return

            new_root = self._make_node(self.root.index, level=level + 1)
            new_root.children.append(self.root)
            self.root = new_root
            level += 1
            self.max_level = level
            queue = [self.root]

    def _insert(self, idx: int, queue: Iterable[_CoverTreeV2Node], level: int) -> bool:
        assert self._working_vectors is not None
        distance_threshold = 2.0 ** level
        queue = list(queue)

        children: List[_CoverTreeV2Node] = []
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
                new_node = self._make_node(idx, level=level - 1)
                node.children.append(new_node)
                return True

        return False

    # ------------------------------------------------------------------
    # Search helpers
    # ------------------------------------------------------------------
    def _cover_search(self, query: np.ndarray) -> List[int]:
        assert self.root is not None
        assert self._working_vectors is not None

        candidate_indices: List[int] = []
        visited: set[int] = set()

        def record(node: _CoverTreeV2Node) -> None:
            if node.index not in visited:
                visited.add(node.index)
                candidate_indices.append(node.index)

        Q_i: List[_CoverTreeV2Node] = [self.root]
        i = self.max_level
        record(self.root)

        while Q_i and i >= -self.dimension:  # fall back to negative levels if needed
            Q_children: List[_CoverTreeV2Node] = []
            for node in Q_i:
                if node.children:
                    Q_children.extend(node.children)

            if not Q_children:
                break

            child_distances = [self._distance_to_query(query, child.index) for child in Q_children]
            min_child_dist = float(np.min(child_distances))
            threshold = min_child_dist + (2.0 ** i)

            for child in Q_children:
                record(child)

            Q_next: List[_CoverTreeV2Node] = []
            for child, dist in zip(Q_children, child_distances):
                if dist <= threshold:
                    Q_next.append(child)

            Q_i = Q_next
            i -= 1

        total_points = self._working_vectors.shape[0]
        if len(candidate_indices) < total_points:
            candidate_indices.extend(
                idx for idx in range(total_points) if idx not in visited
            )

        return candidate_indices

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

    def _metric_fn(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.metric_name in ("l2", "euclidean"):
            return float(np.linalg.norm(a - b))
        if self.metric_name == "cosine":
            return float(1.0 - np.dot(a, b))
        if self.metric_name in ("dot", "ip", "inner_product"):
            return float(-np.dot(a, b))
        raise ValueError(f"Unsupported metric: {self.metric_name}")
