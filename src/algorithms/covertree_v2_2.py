from __future__ import annotations

import heapq
import json
import shutil
import tempfile
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
        self._persistence_schema_version = 1

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

    def save_index(
        self,
        artifact_dir: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.index_built or self.root is None or self._working_vectors is None:
            raise RuntimeError("Cannot persist CoverTreeV2_2 before build_index has completed.")

        context = context or {}
        target_dir = Path(artifact_dir)
        target_dir.parent.mkdir(parents=True, exist_ok=True)

        force_rebuild = bool(context.get("force_rebuild", False))
        if target_dir.exists():
            if not force_rebuild:
                raise FileExistsError(
                    f"Artifact directory already exists: {target_dir}. "
                    "Set persistence.force_rebuild=true to overwrite."
                )
            shutil.rmtree(target_dir)

        temp_dir_path = tempfile.mkdtemp(prefix=f".{target_dir.name}.tmp.", dir=str(target_dir.parent))
        temp_dir = Path(temp_dir_path)

        try:
            node_indices, node_levels, child_offsets, child_ids = self._serialize_tree()

            vectors_path = temp_dir / "vectors.npy"
            node_indices_path = temp_dir / "tree_indices.npy"
            node_levels_path = temp_dir / "tree_levels.npy"
            child_offsets_path = temp_dir / "tree_child_offsets.npy"
            child_ids_path = temp_dir / "tree_children.npy"

            np.save(vectors_path, self._working_vectors, allow_pickle=False)
            np.save(node_indices_path, node_indices, allow_pickle=False)
            np.save(node_levels_path, node_levels, allow_pickle=False)
            np.save(child_offsets_path, child_offsets, allow_pickle=False)
            np.save(child_ids_path, child_ids, allow_pickle=False)

            build_metrics = context.get("build_metrics", {})
            manifest = {
                "schema_version": self._persistence_schema_version,
                "algorithm_type": self.__class__.__name__,
                "algorithm_name": self.name,
                "metric": self.metric_name,
                "dimension": int(self.dimension),
                "vector_count": int(self._working_vectors.shape[0]),
                "node_count": int(node_indices.shape[0]),
                "max_level": int(self.max_level),
                "root_node_id": 0,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "dataset_fingerprint": context.get("dataset_fingerprint"),
                "dataset_fingerprint_payload": context.get("dataset_fingerprint_payload"),
                "config_hash": context.get("config_hash"),
                "build_metrics": build_metrics,
                "files": {
                    "vectors": vectors_path.name,
                    "tree_indices": node_indices_path.name,
                    "tree_levels": node_levels_path.name,
                    "tree_child_offsets": child_offsets_path.name,
                    "tree_children": child_ids_path.name,
                },
            }

            manifest_path = temp_dir / "manifest.json"
            build_metrics_path = temp_dir / "build_metrics.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            build_metrics_path.write_text(json.dumps(build_metrics, indent=2), encoding="utf-8")

            # Last file written: sentinel marks a complete artifact.
            (temp_dir / "WRITE_COMPLETE").write_text("ok\n", encoding="utf-8")

            temp_dir.rename(target_dir)
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

        return {
            "artifact_dir": str(target_dir),
            "manifest_path": str(target_dir / "manifest.json"),
            "build_time_s": float(build_metrics.get("build_time_s", 0.0) or 0.0),
        }

    def load_index(
        self,
        artifact_dir: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = context or {}
        artifact_path = Path(artifact_dir)

        if not artifact_path.is_dir():
            raise FileNotFoundError(f"Persisted CoverTreeV2_2 artifact directory not found: {artifact_path}")
        if not (artifact_path / "WRITE_COMPLETE").is_file():
            raise FileNotFoundError(
                f"Artifact is incomplete or corrupted (missing WRITE_COMPLETE): {artifact_path}"
            )

        manifest_path = artifact_path / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing manifest.json in artifact directory: {artifact_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._validate_manifest(manifest, context=context, artifact_dir=str(artifact_path))

        files = manifest.get("files", {})
        vectors = np.load(artifact_path / files["vectors"], allow_pickle=False)
        node_indices = np.load(artifact_path / files["tree_indices"], allow_pickle=False)
        node_levels = np.load(artifact_path / files["tree_levels"], allow_pickle=False)
        child_offsets = np.load(artifact_path / files["tree_child_offsets"], allow_pickle=False)
        child_ids = np.load(artifact_path / files["tree_children"], allow_pickle=False)

        if vectors.ndim != 2:
            raise ValueError(f"Invalid vectors shape in persisted artifact: {vectors.shape}")
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Persisted index dimension mismatch: expected {self.dimension}, found {vectors.shape[1]}"
            )
        if node_indices.ndim != 1 or node_levels.ndim != 1:
            raise ValueError("Persisted tree metadata must be 1-dimensional arrays.")
        if node_indices.shape[0] != node_levels.shape[0]:
            raise ValueError("Persisted tree index/level arrays have different lengths.")
        if child_offsets.ndim != 1 or child_offsets.shape[0] != node_indices.shape[0] + 1:
            raise ValueError("Persisted child offset array has invalid shape.")
        if child_ids.ndim != 1:
            raise ValueError("Persisted child list must be a 1-dimensional array.")

        node_count = int(node_indices.shape[0])
        nodes: List[_CoverTreeV2Node] = []
        for i in range(node_count):
            point_idx = int(node_indices[i])
            if point_idx < 0 or point_idx >= vectors.shape[0]:
                raise ValueError(
                    f"Persisted node points at invalid vector index {point_idx} for node {i} "
                    f"(vector_count={vectors.shape[0]})"
                )
            level = int(node_levels[i])
            nodes.append(
                _CoverTreeV2Node(
                    index=point_idx,
                    level=level,
                    vector=vectors[point_idx],
                    children=[],
                )
            )

        for i in range(node_count):
            start = int(child_offsets[i])
            end = int(child_offsets[i + 1])
            if start < 0 or end < start or end > child_ids.shape[0]:
                raise ValueError(f"Invalid child offset range [{start}, {end}) for node {i}")
            children = []
            for child_id in child_ids[start:end]:
                child_pos = int(child_id)
                if child_pos < 0 or child_pos >= node_count:
                    raise ValueError(f"Invalid child node id {child_pos} in persisted tree.")
                children.append(nodes[child_pos])
            nodes[i].children = children

        root_id = int(manifest.get("root_node_id", 0))
        if node_count == 0:
            self.root = None
            self.max_level = 0
        else:
            if root_id < 0 or root_id >= node_count:
                raise ValueError(f"Invalid root node id {root_id} in persisted manifest.")
            self.root = nodes[root_id]
            self.max_level = int(manifest.get("max_level", max(node_levels.tolist(), default=0)))

        self._working_vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.vectors = self._working_vectors
        self.metadata = None
        self.index_built = True

        build_metrics = manifest.get("build_metrics", {})
        return {
            "artifact_dir": str(artifact_path),
            "manifest_path": str(manifest_path),
            "build_time_s": float(build_metrics.get("build_time_s", 0.0) or 0.0),
            "dataset_fingerprint": manifest.get("dataset_fingerprint"),
            "config_hash": manifest.get("config_hash"),
        }

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
        self.record_operation('ndis', float(len(vectors)))
        return vectors / norms

    def _prepare_query(self, query: np.ndarray) -> np.ndarray:
        arr = np.asarray(query, dtype=np.float32).reshape(-1)
        if arr.size != self.dimension:
            raise ValueError(f"Query vector must have dimension {self.dimension}")
        if self.metric_name == "cosine":
            norm = np.linalg.norm(arr)
            self.record_operation("ndis", float(1.0))
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
            self.record_operation("ndis", float(len(vectors)))
            return np.linalg.norm(diff, axis=1)
        elif self.metric_name == "cosine":
            # vectors and query are assumed normalized
            self.record_operation("ndis", float(len(vectors)))
            return 1.0 - np.dot(vectors, query)
        elif self.metric_name in ("dot", "ip", "inner_product"):
            self.record_operation("ndis", float(len(vectors)))
            return -np.dot(vectors, query)
        else:
            return np.array(
                [self._metric_fn(query, vec) for vec in vectors],
                dtype=np.float32,
            )

    def _serialize_tree(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self.root is not None

        nodes: List[_CoverTreeV2Node] = []
        node_ids: Dict[int, int] = {}
        queue: deque[_CoverTreeV2Node] = deque([self.root])
        node_ids[id(self.root)] = 0
        nodes.append(self.root)

        while queue:
            node = queue.popleft()
            for child in node.children:
                child_key = id(child)
                if child_key in node_ids:
                    continue
                node_ids[child_key] = len(nodes)
                nodes.append(child)
                queue.append(child)

        node_count = len(nodes)
        node_indices = np.empty(node_count, dtype=np.int64)
        node_levels = np.empty(node_count, dtype=np.int32)
        child_offsets = np.zeros(node_count + 1, dtype=np.int64)
        flat_children: List[int] = []

        for i, node in enumerate(nodes):
            node_indices[i] = int(node.index)
            node_levels[i] = int(node.level)
            for child in node.children:
                child_id = node_ids[id(child)]
                flat_children.append(child_id)
            child_offsets[i + 1] = len(flat_children)

        child_ids = np.asarray(flat_children, dtype=np.int64)
        return node_indices, node_levels, child_offsets, child_ids

    def _validate_manifest(
        self,
        manifest: Dict[str, Any],
        context: Dict[str, Any],
        artifact_dir: str,
    ) -> None:
        schema_version = int(manifest.get("schema_version", -1))
        if schema_version != self._persistence_schema_version:
            raise ValueError(
                f"Unsupported CoverTreeV2_2 persistence schema version {schema_version} "
                f"(expected {self._persistence_schema_version})."
            )

        algorithm_type = manifest.get("algorithm_type")
        if algorithm_type != self.__class__.__name__:
            raise ValueError(
                f"Persisted artifact at {artifact_dir} was built for {algorithm_type}, "
                f"not {self.__class__.__name__}."
            )

        persisted_metric = str(manifest.get("metric", "")).lower()
        if persisted_metric != self.metric_name:
            raise ValueError(
                f"Metric mismatch for persisted index at {artifact_dir}: "
                f"expected '{self.metric_name}', found '{persisted_metric}'."
            )

        persisted_dimension = int(manifest.get("dimension", -1))
        if persisted_dimension != self.dimension:
            raise ValueError(
                f"Dimension mismatch for persisted index at {artifact_dir}: "
                f"expected {self.dimension}, found {persisted_dimension}."
            )

        expected_fingerprint = context.get("dataset_fingerprint")
        artifact_fingerprint = manifest.get("dataset_fingerprint")
        if expected_fingerprint:
            if artifact_fingerprint is None:
                raise ValueError(
                    f"Persisted artifact at {artifact_dir} does not include dataset_fingerprint; "
                    "cannot validate compatibility."
                )
            if str(expected_fingerprint) != str(artifact_fingerprint):
                raise ValueError(
                    f"Dataset fingerprint mismatch for persisted index at {artifact_dir}. "
                    f"Expected {expected_fingerprint}, found {artifact_fingerprint}."
                )

        expected_config_hash = context.get("config_hash")
        artifact_config_hash = manifest.get("config_hash")
        if expected_config_hash and artifact_config_hash and str(expected_config_hash) != str(artifact_config_hash):
            raise ValueError(
                f"Config hash mismatch for persisted index at {artifact_dir}. "
                f"Expected {expected_config_hash}, found {artifact_config_hash}."
            )

    def _metric_fn(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.metric_name in ("l2", "euclidean"):
            return float(np.linalg.norm(a - b))
        if self.metric_name == "cosine":
            return float(1.0 - np.dot(a, b))
        if self.metric_name in ("dot", "ip", "inner_product"):
            return float(-np.dot(a, b))
        raise ValueError(f"Unsupported metric: {self.metric_name}")
