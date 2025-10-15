from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .base_algorithm import BaseAlgorithm
from .modular import BaseIndexer, BaseSearcher, IndexArtifact, register_indexer, register_searcher


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Return a row-normalised copy of the provided matrix."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Return a normalised copy of the provided vector."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return np.zeros_like(vector)
    return vector / norm


class LSHIndexer(BaseIndexer):
    """
    Random-projection locality sensitive hashing (LSH) indexer.

    Guarantee: for unit-normalised (cosine) data, points whose angular distance is at most θ
    collide in at least one of the `num_tables` hash tables with probability
    `1 - (1 - (1 - θ/π) ** hash_size) ** num_tables`. For Euclidean queries, we apply the standard
    E2-LSH scheme guaranteeing that pairs within radius r collide with probability
    `1 - (1 - p1 ** hash_size) ** num_tables`, where `p1` depends on `bucket_width` (see Datar et al. 2004).
    Assumptions: vectors follow the declared `metric` (`cosine` implies unit sphere), projections are
    i.i.d. Gaussian, and queries share the same distribution as the indexed data.
    Reproduction: add the `lsh` algorithm in `configs/benchmark_config.yaml` or run
    `python scripts/run_full_benchmark.py --config configs/lsh_example.yaml`.
    """

    SUPPORTED_METRICS = {"cosine", "l2"}

    def __init__(
        self,
        name: str,
        dimension: int,
        metric: str = "cosine",
        num_tables: int = 8,
        hash_size: int = 16,
        bucket_width: float = 4.0,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, dimension, metric, num_tables=num_tables, hash_size=hash_size, bucket_width=bucket_width, seed=seed, **kwargs)
        if metric not in self.SUPPORTED_METRICS:
            raise ValueError(f"LSHIndexer supports metrics {self.SUPPORTED_METRICS}, received '{metric}'")
        if hash_size <= 0:
            raise ValueError("hash_size must be positive")
        if num_tables <= 0:
            raise ValueError("num_tables must be positive")
        if metric == "l2" and bucket_width <= 0:
            raise ValueError("bucket_width must be positive for L2 LSH")

        self.num_tables = num_tables
        self.hash_size = hash_size
        self.bucket_width = bucket_width
        self.seed = seed

    def _sample_projections(self, rng: np.random.RandomState) -> np.ndarray:
        return rng.normal(size=(self.num_tables, self.hash_size, self.dimension)).astype(np.float32)

    def _sample_offsets(self, rng: np.random.RandomState) -> Optional[np.ndarray]:
        if self.metric != "l2":
            return None
        return rng.uniform(0.0, self.bucket_width, size=(self.num_tables, self.hash_size)).astype(np.float32)

    def _hash_cosine(self, vector: np.ndarray, projections: np.ndarray, bit_weights: np.ndarray) -> np.ndarray:
        signs = (projections @ vector >= 0).astype(np.uint8)
        return (signs.astype(np.uint64) * bit_weights[None, :]).sum(axis=1)

    def _hash_l2(self, vector: np.ndarray, projections: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        shifted = (projections @ vector + offsets) / self.bucket_width
        return np.floor(shifted).astype(np.int32)

    def _insert_into_tables(
        self,
        tables: List[Dict[Any, List[int]]],
        hashes: Sequence[Any],
        vector_index: int,
    ) -> None:
        for table_idx, hash_key in enumerate(hashes):
            tables[table_idx][hash_key].append(vector_index)

    def build(self, vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> IndexArtifact:
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected vectors with dimension {self.dimension}, received {vectors.shape[1]}")

        rng = np.random.RandomState(self.seed)
        projections = self._sample_projections(rng)
        offsets = self._sample_offsets(rng)
        bit_weights = (1 << np.arange(self.hash_size, dtype=np.uint64))

        processed_vectors = vectors.astype(np.float32, copy=True)
        if self.metric == "cosine":
            processed_vectors = _normalize_rows(processed_vectors)

        tables: List[Dict[Any, List[int]]] = [defaultdict(list) for _ in range(self.num_tables)]

        for idx, vector in enumerate(processed_vectors):
            if self.metric == "cosine":
                hash_codes = self._hash_cosine(vector, projections, bit_weights)
                keys: Iterable[Any] = hash_codes.tolist()
            else:
                hash_codes = self._hash_l2(vector, projections, offsets)  # type: ignore[arg-type]
                keys = [tuple(hash_codes_row.tolist()) for hash_codes_row in hash_codes]
            self._insert_into_tables(tables, keys, idx)

        artifact_metadata: Dict[str, Any] = {
            "metric": self.metric,
            "num_tables": self.num_tables,
            "hash_size": self.hash_size,
        }
        if self.metric == "cosine":
            artifact_metadata["normalize_queries"] = True
        else:
            artifact_metadata["bucket_width"] = self.bucket_width

        artifact_data: Dict[str, Any] = {
            "tables": [dict(table) for table in tables],
            "projections": projections,
            "vector_store": processed_vectors,
            "bit_weights": bit_weights,
        }
        if offsets is not None:
            artifact_data["offsets"] = offsets

        return IndexArtifact(kind="lsh", data=artifact_data, metadata=artifact_metadata)


register_indexer("LSHIndexer", LSHIndexer)


class LSHSearcher(BaseSearcher):
    """
    Candidate generator and reranker for the LSH index.

    The searcher gathers candidates from the hash collisions and optionally falls back to
    brute-force scoring if no bucket has been hit. While the lookup is approximate, scoring
    is exact on the retrieved candidate set.
    """

    def __init__(
        self,
        name: str,
        dimension: int,
        metric: str = "cosine",
        candidate_multiplier: float = 4.0,
        max_candidates: Optional[int] = None,
        fallback_to_bruteforce: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, dimension, metric, candidate_multiplier=candidate_multiplier, max_candidates=max_candidates, fallback_to_bruteforce=fallback_to_bruteforce, **kwargs)
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be positive")
        self.candidate_multiplier = candidate_multiplier
        self.max_candidates = max_candidates
        self.fallback_to_bruteforce = fallback_to_bruteforce

    def attach(self, artifact: IndexArtifact, vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        if artifact.kind != "lsh":
            raise ValueError("LSHSearcher can only attach to artifacts produced by LSHIndexer")

        data = artifact.data
        self.tables: List[Dict[Any, List[int]]] = data["tables"]
        self.projections: np.ndarray = data["projections"]
        self.vector_store: np.ndarray = data["vector_store"]
        self.bit_weights: np.ndarray = data["bit_weights"]
        self.offsets: Optional[np.ndarray] = data.get("offsets")

        self.metric = artifact.metadata.get("metric", self.metric)
        self.normalize_queries = artifact.metadata.get("normalize_queries", False)
        self.hash_size = artifact.metadata.get("hash_size", self.vector_store.shape[1])
        self.num_tables = artifact.metadata.get("num_tables", len(self.tables))
        self.bucket_width = artifact.metadata.get("bucket_width", None)
        self._prepared = True

    def _prepare_query(self, query: np.ndarray) -> np.ndarray:
        if query.ndim != 1:
            query = query.reshape(-1)
        query = query.astype(np.float32, copy=True)
        if self.normalize_queries:
            query = _normalize_vector(query)
        return query

    def _hash_cosine(self, query: np.ndarray) -> List[int]:
        projections = self.projections.reshape(self.num_tables, self.hash_size, self.dimension)
        hashes = []
        for table_idx in range(self.num_tables):
            table_proj = projections[table_idx]
            signs = (table_proj @ query >= 0).astype(np.uint8)
            hash_value = int((signs.astype(np.uint64) * self.bit_weights[: self.hash_size]).sum())
            hashes.append(hash_value)
        return hashes

    def _hash_l2(self, query: np.ndarray) -> List[Tuple[int, ...]]:
        if self.offsets is None or self.bucket_width is None:
            raise RuntimeError("L2 hashing requires offsets and bucket_width")
        projections = self.projections.reshape(self.num_tables, self.hash_size, self.dimension)
        hashes: List[Tuple[int, ...]] = []
        for table_idx in range(self.num_tables):
            table_proj = projections[table_idx]
            table_offsets = self.offsets[table_idx]
            values = (table_proj @ query + table_offsets) / self.bucket_width
            codes = tuple(np.floor(values).astype(np.int32).tolist())
            hashes.append(codes)
        return hashes

    def _gather_candidates(self, hashes: Sequence[Any]) -> List[int]:
        vote_counter: Counter[int] = Counter()
        for table_idx, hash_key in enumerate(hashes):
            bucket = self.tables[table_idx].get(hash_key)
            if bucket:
                vote_counter.update(bucket)

        if vote_counter:
            ordered_candidates = [idx for idx, _ in vote_counter.most_common()]
            return ordered_candidates
        return []

    def _select_candidates(self, k: int, ordered_candidates: List[int]) -> List[int]:
        if not ordered_candidates and self.fallback_to_bruteforce:
            return list(range(self.vector_store.shape[0]))
        if not ordered_candidates:
            return []

        candidate_cap = self.max_candidates
        if candidate_cap is None:
            candidate_cap = max(k, int(math.ceil(self.candidate_multiplier * k)))
        return ordered_candidates[:candidate_cap]

    def _compute_distances(self, query: np.ndarray, candidate_indices: np.ndarray) -> np.ndarray:
        candidate_vectors = self.vector_store[candidate_indices]
        if self.metric == "cosine":
            scores = candidate_vectors @ query
            return (1.0 - scores).astype(np.float32)
        if self.metric == "l2":
            diffs = candidate_vectors - query[None, :]
            return np.linalg.norm(diffs, axis=1).astype(np.float32)
        raise ValueError(f"Unsupported metric '{self.metric}' for LSHSearcher")

    def _search_single(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if not self._prepared:
            raise RuntimeError("LSHSearcher not attached to an index")

        if self.metric == "cosine":
            hashes = self._hash_cosine(query)
        elif self.metric == "l2":
            hashes = self._hash_l2(query)
        else:
            raise ValueError(f"Unsupported metric '{self.metric}'")

        ordered_candidates = self._gather_candidates(hashes)
        candidate_list = self._select_candidates(k, ordered_candidates)

        if not candidate_list:
            distances = np.full(k, np.inf, dtype=np.float32)
            indices = np.full(k, -1, dtype=np.int64)
            return distances, indices

        candidate_indices = np.array(candidate_list, dtype=np.int64)
        distances = self._compute_distances(query, candidate_indices)

        order = np.argsort(distances)
        limit = min(k, order.size)
        top_indices = candidate_indices[order[:limit]]
        top_distances = distances[order[:limit]]

        if limit < k:
            top_distances = np.pad(top_distances, (0, k - limit), constant_values=np.inf)
            top_indices = np.pad(top_indices, (0, k - limit), constant_values=-1)

        return top_distances.astype(np.float32), top_indices.astype(np.int64)

    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        prepared = self._prepare_query(query)
        return self._search_single(prepared, k)

    def batch_search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        distances_list = []
        indices_list = []
        for query in queries:
            distances, indices = self.search(query, k)
            distances_list.append(distances)
            indices_list.append(indices)
        return np.stack(distances_list, axis=0), np.stack(indices_list, axis=0)


register_searcher("LSHSearcher", LSHSearcher)


class LSH(BaseAlgorithm):
    """
    Convenience wrapper exposing the LSH indexer/searcher pair as a standalone algorithm.

    Use this class when wiring algorithms programmatically; for configuration-driven runs prefer
    the modular `indexer_ref` + `searcher_ref` approach demonstrated in `configs/benchmark_config.yaml`.
    """

    def __init__(
        self,
        name: str,
        dimension: int,
        metric: str = "cosine",
        num_tables: int = 8,
        hash_size: int = 16,
        bucket_width: float = 4.0,
        candidate_multiplier: float = 4.0,
        max_candidates: Optional[int] = None,
        fallback_to_bruteforce: bool = True,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, dimension, metric=metric, num_tables=num_tables, hash_size=hash_size, bucket_width=bucket_width, candidate_multiplier=candidate_multiplier, max_candidates=max_candidates, fallback_to_bruteforce=fallback_to_bruteforce, seed=seed, **kwargs)
        self.metric = metric
        self.indexer = LSHIndexer(
            name=f"{name}_indexer",
            dimension=dimension,
            metric=metric,
            num_tables=num_tables,
            hash_size=hash_size,
            bucket_width=bucket_width,
            seed=seed,
        )
        self.searcher = LSHSearcher(
            name=f"{name}_searcher",
            dimension=dimension,
            metric=metric,
            candidate_multiplier=candidate_multiplier,
            max_candidates=max_candidates,
            fallback_to_bruteforce=fallback_to_bruteforce,
        )

    def build_index(self, vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        artifact = self.indexer.build(vectors, metadata)
        self.searcher.attach(artifact, vectors, metadata)
        self.index_built = True

    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if not self.index_built:
            raise RuntimeError("Index has not been built for LSH algorithm")
        return self.searcher.search(query, k)

    def batch_search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if not self.index_built:
            raise RuntimeError("Index has not been built for LSH algorithm")
        return self.searcher.batch_search(queries, k)


__all__ = ["LSHIndexer", "LSHSearcher", "LSH"]
