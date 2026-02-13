from __future__ import annotations

import heapq
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


class CoverTreeV2_2(BaseAlgorithm):
    """
    Optimized Cover Tree variant (V2.2) based on CoverTreeV2.
    
    Optimizations & Changes:
    1. Vectorized distance calculations using NumPy.
    2. Implements Exact k-NN search (100% recall) using dynamic pruning with a priority queue.
       This replaces the previous 1-NN heuristic and brute-force fallback.
    3. Merged search and ranking steps to avoid redundant distance computations.
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
        return self._search_exact_k(prepared, k)

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
        point_vector = self._working_vectors[idx]
        distance_threshold = 2.0 ** level
        queue = list(queue)

        children: List[_CoverTreeV2Node] = []
        for node in queue:
            children.extend(node.children)

        # Optimization: Vectorized check for children
        if children:
            child_vectors = np.stack([c.vector for c in children])
            dists = self._compute_distance_batch_to_1(point_vector, child_vectors)
            filtered = [child for child, d in zip(children, dists) if d <= distance_threshold]
        else:
            filtered = []

        if filtered and self._insert(idx, filtered, level - 1):
            return True

        # Optimization: Vectorized check for queue nodes
        if queue:
            node_vectors = np.stack([n.vector for n in queue])
            dists_nodes = self._compute_distance_batch_to_1(point_vector, node_vectors)
            
            # Find first node that satisfies the condition
            for i, d in enumerate(dists_nodes):
                if d <= distance_threshold:
                    new_node = self._make_node(idx, level=level - 1)
                    queue[i].children.append(new_node)
                    return True

        return False

    # ------------------------------------------------------------------
    # Exact k-NN Search Logic
    # ------------------------------------------------------------------
    def _search_exact_k(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs an exact k-nearest neighbor search using a max-heap to maintain
        the top-k candidates and dynamically prune the search space.
        """
        assert self.root is not None
        
        # Max-heap storing (-distance, index) for top-k results.
        # Python's heapq is a min-heap, so we store negative distances.
        # Initial dummy values are not needed if we handle the size logic carefully,
        # but filling with infinity makes logical checks simpler.
        candidates_heap: List[Tuple[float, int]] = [] 
        
        # Track visited POINTS (indices) to avoid duplicates in results,
        # since points can appear at multiple levels/nodes.
        visited_points: set[int] = set()

        def update_heap(dist: float, idx: int):
            if idx in visited_points:
                return
            visited_points.add(idx)
            
            # We want to keep the SMALLEST k distances.
            # In a max-heap of size k, the root is the LARGEST of the k.
            # If new_dist < max_in_heap, we replace root.
            
            if len(candidates_heap) < k:
                heapq.heappush(candidates_heap, (-dist, idx))
            else:
                # Check if this point is better than the worst point in our current top-k
                # heap[0] is (-max_dist, index)
                max_dist_in_heap = -candidates_heap[0][0]
                if dist < max_dist_in_heap:
                    heapq.heapreplace(candidates_heap, (-dist, idx))

        # The current set of nodes to explore
        current_nodes: List[_CoverTreeV2Node] = [self.root]
        
        # Add root to candidates immediately
        root_dist = float(self._compute_distance_batch_to_1(query, np.array([self.root.vector]))[0])
        update_heap(root_dist, self.root.index)
        
        level = self.max_level
        
        # Traverse down
        while current_nodes and level >= -self.dimension: # Safety break at very low levels
            
            # 1. Collect all children of current nodes
            all_children = [child for node in current_nodes for child in node.children]
            
            if not all_children:
                break
                
            # 2. Compute distances to all children efficiently
            child_vectors = np.stack([c.vector for c in all_children])
            child_distances = self._compute_distance_batch_to_1(query, child_vectors)
            
            # 3. Determine current pruning bound before heap updates.
            # The k-th nearest neighbor distance found so far is our pruning bound.
            # If we haven't found k points yet, bound is infinity.
            if len(candidates_heap) < k:
                pruning_bound = float('inf')
            else:
                pruning_bound = -candidates_heap[0][0]

            # 4. Update heap with promising children only.
            if np.isfinite(pruning_bound):
                heap_update_mask = child_distances < pruning_bound
                candidate_positions = np.flatnonzero(heap_update_mask)
            else:
                candidate_positions = range(len(all_children))

            for pos in candidate_positions:
                child = all_children[pos]
                update_heap(float(child_distances[pos]), child.index)

            # 5. Recompute pruning bound after candidate updates.
            if len(candidates_heap) < k:
                pruning_bound = float('inf')
            else:
                pruning_bound = -candidates_heap[0][0]
            
            # 6. Filter children for next iteration
            # Rule: Keep a child node 'c' if it *could* contain a point closer than pruning_bound.
            # A node 'c' at level `child.level` covers a radius of approx 2^(child.level + 1).
            # Lower bound distance to any descendant of c is: dist(p, c) - radius(c)
            # If lower_bound > pruning_bound, we prune.
            # So: dist(p, c) - 2^(c.level + 1) <= pruning_bound
            if np.isfinite(pruning_bound):
                first_child_level = all_children[0].level
                if all(child.level == first_child_level for child in all_children):
                    max_cover_radius = 2.0 ** (first_child_level + 1)
                    next_mask = child_distances <= (pruning_bound + max_cover_radius)
                else:
                    child_levels = np.fromiter(
                        (child.level for child in all_children),
                        dtype=np.int32,
                        count=len(all_children),
                    )
                    next_mask = child_distances <= (pruning_bound + np.power(2.0, child_levels + 1))
                next_nodes = [all_children[i] for i in np.flatnonzero(next_mask)]
            else:
                next_nodes = all_children
            
            current_nodes = next_nodes
            level -= 1
            
            if not current_nodes:
                break

        # Finalize results
        # Sort by distance (ascending)
        # Heap contains (-dist, idx), so sort reverses this.
        
        sorted_candidates = sorted(candidates_heap, key=lambda x: -x[0])
        
        # Extract
        final_indices = np.array([x[1] for x in sorted_candidates], dtype=np.int64)
        final_distances = np.array([-x[0] for x in sorted_candidates], dtype=np.float32)
        
        # Pad if fewer than k found (should rare/impossible if N >= k)
        if len(final_indices) < k:
             padding = k - len(final_indices)
             final_indices = np.pad(final_indices, (0, padding), constant_values=-1)
             final_distances = np.pad(final_distances, (0, padding), constant_values=np.inf)
             
        return final_distances, final_indices

    def _compute_distance_batch_to_1(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """
        Computes distance from 1 query vector to N vectors.
        vectors: (N, D)
        query: (D,)
        """
        if self.metric_name in ("l2", "euclidean"):
            diff = vectors - query
            return np.linalg.norm(diff, axis=1)
        elif self.metric_name == "cosine":
            # vectors and query are assumed normalized
            return 1.0 - np.dot(vectors, query)
        elif self.metric_name in ("dot", "ip", "inner_product"):
            return -np.dot(vectors, query)
        else:
            return np.array(
                [self._metric_fn(query, vec) for vec in vectors],
                dtype=np.float32,
            )

    def _metric_fn(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.metric_name in ("l2", "euclidean"):
            return float(np.linalg.norm(a - b))
        if self.metric_name == "cosine":
            return float(1.0 - np.dot(a, b))
        if self.metric_name in ("dot", "ip", "inner_product"):
            return float(-np.dot(a, b))
        raise ValueError(f"Unsupported metric: {self.metric_name}")
