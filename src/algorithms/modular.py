"""Modular indexing/search components for composing retrieval algorithms."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type
from abc import ABC, abstractmethod

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - backend optional at import time
    faiss = None

from .base_algorithm import BaseAlgorithm


@dataclass
class IndexArtifact:
    """Container returned by an indexer describing the built index."""

    kind: str
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseIndexer(ABC):
    """Base class for indexing strategies."""

    def __init__(self, name: str, dimension: int, metric: str = "l2", **kwargs: Any) -> None:
        self.name = name
        self.dimension = dimension
        self.metric = metric
        self.params = kwargs

    @abstractmethod
    def build(self, vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> IndexArtifact:
        """Build an index artifact from the provided vectors."""

    def describe(self) -> Dict[str, Any]:
        description = {
            "name": self.name,
            "type": self.__class__.__name__,
            "metric": self.metric,
        }
        if self.params:
            description["params"] = copy.deepcopy(self.params)
        return description


class BaseSearcher(ABC):
    """Base class for search strategies operating over an index artifact."""

    def __init__(self, name: str, dimension: int, metric: str = "l2", **kwargs: Any) -> None:
        self.name = name
        self.dimension = dimension
        self.metric = metric
        self.params = kwargs
        self._prepared = False

    @abstractmethod
    def attach(self, artifact: IndexArtifact, vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """Attach to an index artifact prior to servicing queries."""

    @abstractmethod
    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Search the attached index for a single query."""

    @abstractmethod
    def batch_search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Search the attached index for a batch of queries."""

    def describe(self) -> Dict[str, Any]:
        description = {
            "name": self.name,
            "type": self.__class__.__name__,
            "metric": self.metric,
        }
        if self.params:
            description["params"] = copy.deepcopy(self.params)
        return description


INDEXER_REGISTRY: Dict[str, Type[BaseIndexer]] = {}
SEARCHER_REGISTRY: Dict[str, Type[BaseSearcher]] = {}


def register_indexer(name: str, cls: Type[BaseIndexer]) -> None:
    INDEXER_REGISTRY[name] = cls


def register_searcher(name: str, cls: Type[BaseSearcher]) -> None:
    SEARCHER_REGISTRY[name] = cls


def get_indexer_class(name: str) -> Type[BaseIndexer]:
    if name not in INDEXER_REGISTRY:
        raise ValueError(f"Unknown indexer type '{name}'. Available: {list(INDEXER_REGISTRY.keys())}")
    return INDEXER_REGISTRY[name]


def get_searcher_class(name: str) -> Type[BaseSearcher]:
    if name not in SEARCHER_REGISTRY:
        raise ValueError(f"Unknown searcher type '{name}'. Available: {list(SEARCHER_REGISTRY.keys())}")
    return SEARCHER_REGISTRY[name]


def _safe_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def _ensure_float32(vectors: np.ndarray) -> np.ndarray:
    """Ensure vectors are float32 and C-contiguous without unnecessary copying."""
    if vectors.dtype == np.float32 and vectors.flags["C_CONTIGUOUS"]:
        return vectors
    return np.ascontiguousarray(vectors, dtype=np.float32)


class BruteForceIndexer(BaseIndexer):
    """Indexer that simply stores raw vectors in memory."""

    def build(self, vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> IndexArtifact:
        vector_store = _ensure_float32(vectors)
        artifact_metadata = {
            "metric": self.metric,
            "normalize_vectors": self.metric == "cosine",
        }
        return IndexArtifact(kind="raw_vectors", data=vector_store, metadata=artifact_metadata)


register_indexer("BruteForceIndexer", BruteForceIndexer)


class HNSWIndexer(BaseIndexer):
    """FAISS-backed HNSW indexer."""

    def __init__(self, name: str, dimension: int, metric: str = "l2", M: int = 16, efConstruction: int = 200, **kwargs: Any) -> None:
        if faiss is None:
            raise ImportError("faiss is required for HNSWIndexer")
        super().__init__(name, dimension, metric, M=M, efConstruction=efConstruction, **kwargs)
        self.M = M
        self.efConstruction = efConstruction
        self.efSearch = kwargs.get("efSearch", 100)

    def build(self, vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> IndexArtifact:
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected dimension {self.dimension}, got {vectors.shape[1]}")

        metric_type = faiss.METRIC_L2
        data = _ensure_float32(vectors)
        artefact_metadata: Dict[str, Any] = {
            "metric": self.metric,
            "faiss_metric": "l2",
            "efSearch": self.efSearch,
        }

        if self.metric == "cosine":
            data = _safe_normalize(data)
            metric_type = faiss.METRIC_INNER_PRODUCT
            artefact_metadata.update({
                "faiss_metric": "ip",
                "normalize_queries": True,
                "normalize_vectors": True,
            })
        elif self.metric == "ip":
            metric_type = faiss.METRIC_INNER_PRODUCT
            artefact_metadata["faiss_metric"] = "ip"

        index = faiss.IndexHNSWFlat(self.dimension, self.M, metric_type)
        index.hnsw.efConstruction = self.efConstruction
        index.hnsw.efSearch = self.efSearch
        index.add(data)

        return IndexArtifact(kind="faiss", data=index, metadata=artefact_metadata)


register_indexer("HNSWIndexer", HNSWIndexer)


class FaissLSHIndexer(BaseIndexer):
    """FAISS-backed LSH indexer using random-hyperplane hashes."""

    SUPPORTED_METRICS = {"l2", "cosine", "ip"}

    def __init__(self, name: str, dimension: int, metric: str = "l2", num_bits: int = 256, **kwargs: Any) -> None:
        if faiss is None:
            raise ImportError("faiss is required for FaissLSHIndexer")
        if metric not in self.SUPPORTED_METRICS:
            raise ValueError(f"FaissLSHIndexer supports metrics {self.SUPPORTED_METRICS}, received '{metric}'")
        if num_bits <= 0:
            raise ValueError("num_bits must be positive")

        super().__init__(name, dimension, metric, num_bits=num_bits, **kwargs)
        self.num_bits = num_bits

    def build(self, vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> IndexArtifact:
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected dimension {self.dimension}, got {vectors.shape[1]}")

        data = _ensure_float32(vectors)
        artefact_metadata: Dict[str, Any] = {
            "metric": self.metric,
            "num_bits": self.num_bits,
            "faiss_index_kind": "lsh",
        }

        if self.metric == "cosine":
            data = _safe_normalize(data)
            artefact_metadata["normalize_queries"] = True
        elif self.metric == "ip":
            artefact_metadata["faiss_metric"] = "ip"

        index = faiss.IndexLSH(self.dimension, self.num_bits)
        index.add(data)

        return IndexArtifact(kind="faiss", data=index, metadata=artefact_metadata)


register_indexer("FaissLSHIndexer", FaissLSHIndexer)


class FaissFactoryIndexer(BaseIndexer):
    """Generic FAISS indexer driven by the index_factory grammar."""

    _RESERVED_PARAM_KEYS = {"index_key", "index_type"}

    def __init__(
        self,
        name: str,
        dimension: int,
        metric: str = "l2",
        index_key: str = "Flat",
        **kwargs: Any,
    ) -> None:
        if faiss is None:
            raise ImportError("faiss is required for FaissFactoryIndexer")
        self.index_key = index_key
        params = dict(kwargs)
        params.setdefault("index_key", index_key)
        super().__init__(name, dimension, metric, **params)

    def _prepare_data(self, vectors: np.ndarray) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        data = _ensure_float32(vectors)
        metric_kind = faiss.METRIC_L2
        artefact_metadata: Dict[str, Any] = {
            "metric": self.metric,
            "index_key": self.index_key,
            "faiss_metric": "l2",
        }

        if self.metric == "cosine":
            data = _safe_normalize(data)
            metric_kind = faiss.METRIC_INNER_PRODUCT
            artefact_metadata.update(
                {
                    "faiss_metric": "ip",
                    "normalize_queries": True,
                    "normalize_vectors": True,
                }
            )
        elif self.metric == "ip":
            metric_kind = faiss.METRIC_INNER_PRODUCT
            artefact_metadata["faiss_metric"] = "ip"

        return data, metric_kind, artefact_metadata

    def _apply_runtime_params(self, index: "faiss.Index", artefact_metadata: Dict[str, Any]) -> None:
        for key, value in self.params.items():
            if key in self._RESERVED_PARAM_KEYS:
                continue
            if hasattr(index, key):
                setattr(index, key, value)
                artefact_metadata[key] = value

    def build(self, vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> IndexArtifact:
        data, metric_kind, artefact_metadata = self._prepare_data(vectors)
        index = faiss.index_factory(self.dimension, self.index_key, metric_kind)

        if not index.is_trained:
            index.train(data)
        index.add(data)

        self._apply_runtime_params(index, artefact_metadata)
        return IndexArtifact(kind="faiss", data=index, metadata=artefact_metadata)


register_indexer("FaissFactoryIndexer", FaissFactoryIndexer)


class FaissIVFIndexer(FaissFactoryIndexer):
    """Backward-compatible wrapper around FaissFactoryIndexer for IVF-based indexes."""

    def __init__(
        self,
        name: str,
        dimension: int,
        metric: str = "l2",
        index_type: str = "IVF100,Flat",
        **kwargs: Any,
    ) -> None:
        params = dict(kwargs)
        params.setdefault("index_type", index_type)
        super().__init__(name, dimension, metric, index_key=index_type, **params)
        self.index_type = index_type


register_indexer("FaissIVFIndexer", FaissIVFIndexer)


class LinearSearcher(BaseSearcher):
    """Linear scanning searcher over raw vectors."""

    def attach(self, artifact: IndexArtifact, vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        if artifact.kind != "raw_vectors":
            raise ValueError("LinearSearcher requires 'raw_vectors' artifact")
        self._vectors = artifact.data
        if self._vectors.shape[1] != self.dimension:
            raise ValueError("Vector dimension mismatch in LinearSearcher")

        self._normalized_vectors = None
        if self.metric == "cosine":
            self._normalized_vectors = _safe_normalize(self._vectors)
        self._prepared = True

    def _prepare_query(self, query: np.ndarray) -> np.ndarray:
        if query.ndim == 1:
            query = query.reshape(1, -1)
        return query.astype(np.float32, copy=True)

    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        distances, indices = self.batch_search(self._prepare_query(query), k)
        return distances[0], indices[0]

    def batch_search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if not self._prepared:
            raise RuntimeError("LinearSearcher not attached to an index")
        queries = self._prepare_query(queries)

        if self.metric == "l2":
            # Compute squared L2 distances and take sqrt for final distances
            diffs = self._vectors[None, :, :] - queries[:, None, :]
            sq_dists = np.sum(diffs ** 2, axis=2)
            if sq_dists.shape[1] == 0:
                raise RuntimeError("LinearSearcher cannot operate on empty index")
            limit = min(k, sq_dists.shape[1])
            kth = max(limit - 1, 0)
            topk_idx = np.argpartition(sq_dists, kth=kth, axis=1)[:, :limit]
            row_indices = np.arange(queries.shape[0])[:, None]
            topk_sq = sq_dists[row_indices, topk_idx]
            order = np.argsort(topk_sq, axis=1)
            sorted_idx = topk_idx[row_indices, order]
            sorted_sq = topk_sq[row_indices, order]
            distances = np.sqrt(sorted_sq)
            # Pad if limit < k for consistency
            if limit < k:
                distances = np.pad(distances, ((0, 0), (0, k - limit)), constant_values=np.inf)
                sorted_idx = np.pad(sorted_idx, ((0, 0), (0, k - limit)), constant_values=-1)
            return distances.astype(np.float32), sorted_idx.astype(np.int64)

        # Cosine similarity / inner product share similar logic
        if self.metric in {"cosine", "ip"}:
            if self.metric == "cosine":
                norm_queries = _safe_normalize(queries)
                scores = norm_queries @ self._normalized_vectors.T
            else:  # inner product
                scores = queries @ self._vectors.T

            # Higher scores are better. Convert to distances by negating.
            if scores.shape[1] == 0:
                raise RuntimeError("LinearSearcher cannot operate on empty index")
            limit = min(k, scores.shape[1])
            kth = max(limit - 1, 0)
            topk_idx = np.argpartition(-scores, kth=kth, axis=1)[:, :limit]
            row_indices = np.arange(queries.shape[0])[:, None]
            topk_scores = scores[row_indices, topk_idx]
            order = np.argsort(-topk_scores, axis=1)
            sorted_idx = topk_idx[row_indices, order]
            sorted_scores = topk_scores[row_indices, order]
            distances = -sorted_scores
            if limit < k:
                distances = np.pad(distances, ((0, 0), (0, k - limit)), constant_values=np.inf)
                sorted_idx = np.pad(sorted_idx, ((0, 0), (0, k - limit)), constant_values=-1)
            return distances.astype(np.float32), sorted_idx.astype(np.int64)

        raise ValueError(f"Unsupported metric '{self.metric}' for LinearSearcher")


register_searcher("LinearSearcher", LinearSearcher)


class FaissSearcher(BaseSearcher):
    """Searcher that delegates to a FAISS index.

    For standard FAISS indexes, this is a thin wrapper around ``index.search``.
    When attached to an ``IndexLSH`` artifact (marked via ``faiss_index_kind='lsh'``),
    it optionally performs a lightweight reranking step: it asks FAISS for an
    expanded candidate set, then re-scores those candidates against the original
    vectors using the configured metric. This significantly improves recall for
    LSH while keeping QPS competitive.
    """

    def __init__(self, name: str, dimension: int, metric: str = "l2", **kwargs: Any) -> None:
        super().__init__(name, dimension, metric, **kwargs)
        self.index: Any = None
        self.normalize_queries: bool = False
        self.index_kind: Optional[str] = None
        self._base_vectors: Optional[np.ndarray] = None
        self._normalized_vectors: Optional[np.ndarray] = None

        # LSH-specific knobs (ignored for non-LSH indexes).
        self._lsh_rerank: bool = bool(self.params.get("lsh_rerank", True))
        self._lsh_candidate_multiplier: float = float(self.params.get("lsh_candidate_multiplier", 8.0))
        max_candidates = self.params.get("lsh_max_candidates")
        self._lsh_max_candidates: Optional[int] = int(max_candidates) if max_candidates is not None else None

    def attach(self, artifact: IndexArtifact, vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        if artifact.kind != "faiss":
            raise ValueError("FaissSearcher requires 'faiss' artifact")
        if faiss is None:
            raise ImportError("faiss is required for FaissSearcher")
        self.index = artifact.data
        artefact_metadata = artifact.metadata or {}
        self.metric = artefact_metadata.get("metric", self.metric)
        self.normalize_queries = artefact_metadata.get("normalize_queries", False)
        self.index_kind = artefact_metadata.get("faiss_index_kind")
        self._prepared = True

        # For LSH, keep a reference to the original vectors so we can rerank
        # candidates by the true metric (e.g., L2 or cosine).
        if self.index_kind == "lsh" and self._lsh_rerank:
            self._base_vectors = _ensure_float32(vectors)
            if self.metric == "cosine":
                self._normalized_vectors = _safe_normalize(self._base_vectors)

        desired_nprobe = self.params.get("nprobe")
        if desired_nprobe is None:
            desired_nprobe = artefact_metadata.get("nprobe")
        if desired_nprobe is not None and hasattr(self.index, "nprobe"):
            self.index.nprobe = int(desired_nprobe)

    def _prepare_query(self, query: np.ndarray) -> np.ndarray:
        if query.ndim == 1:
            query = query.reshape(1, -1)
        query = query.astype(np.float32, copy=True)
        if self.normalize_queries:
            query = _safe_normalize(query)
        return query

    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        distances, indices = self.batch_search(self._prepare_query(query), k)
        return distances[0], indices[0]

    def _batch_search_lsh_rerank(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None or self._base_vectors is None:
            raise RuntimeError("LSH rerank path requires an attached index and base vectors")

        num_db = int(getattr(self.index, "ntotal", 0) or self._base_vectors.shape[0])
        if num_db <= 0:
            raise RuntimeError("LSH index has no vectors to search")

        candidate_k = max(k, 1)
        if self._lsh_candidate_multiplier > 1.0:
            candidate_k = int(max(candidate_k, k * self._lsh_candidate_multiplier))
        if self._lsh_max_candidates is not None:
            candidate_k = min(candidate_k, self._lsh_max_candidates)
        candidate_k = min(candidate_k, num_db)

        if candidate_k <= 0:
            # Fall back to the raw FAISS path in degenerate cases.
            distances, indices = self.index.search(queries, k)
            if self.metric in {"cosine", "ip"}:
                distances = -distances
            return distances.astype(np.float32), indices.astype(np.int64)

        _, candidate_indices = self.index.search(queries, candidate_k)

        n_queries = queries.shape[0]
        reranked_distances = np.full((n_queries, k), np.inf, dtype=np.float32)
        reranked_indices = np.full((n_queries, k), -1, dtype=np.int64)

        for i in range(n_queries):
            row_candidates = candidate_indices[i]
            valid = row_candidates[row_candidates >= 0]
            if valid.size == 0:
                # No candidates: rely on raw FAISS ordering for this query.
                raw_distances, raw_indices = self.index.search(queries[i : i + 1], k)
                if self.metric in {"cosine", "ip"}:
                    raw_distances = -raw_distances
                reranked_distances[i, :] = raw_distances[0].astype(np.float32)
                reranked_indices[i, :] = raw_indices[0].astype(np.int64)
                continue

            if self.metric == "l2":
                cand_vectors = self._base_vectors[valid]
                diffs = cand_vectors - queries[i : i + 1]
                sq_dists = np.sum(diffs ** 2, axis=1)
                limit = min(k, sq_dists.shape[0])
                kth = max(limit - 1, 0)
                topk_idx = np.argpartition(sq_dists, kth=kth)[:limit]
                topk_sq = sq_dists[topk_idx]
                order = np.argsort(topk_sq)
                sorted_sq = topk_sq[order]
                sorted_idx = valid[topk_idx[order]]
                reranked_distances[i, :limit] = np.sqrt(sorted_sq).astype(np.float32)
                reranked_indices[i, :limit] = sorted_idx.astype(np.int64)
            elif self.metric in {"cosine", "ip"}:
                # Queries are already normalized when metric == "cosine".
                if self.metric == "cosine":
                    if self._normalized_vectors is None:
                        raise RuntimeError("Normalized vectors are missing for cosine LSH rerank")
                    cand_vectors = self._normalized_vectors[valid]
                    query_vec = queries[i : i + 1]
                else:  # inner product
                    cand_vectors = self._base_vectors[valid]
                    query_vec = queries[i : i + 1]

                scores = (query_vec @ cand_vectors.T).ravel()
                limit = min(k, scores.shape[0])
                kth = max(limit - 1, 0)
                topk_idx = np.argpartition(-scores, kth=kth)[:limit]
                topk_scores = scores[topk_idx]
                order = np.argsort(-topk_scores)
                sorted_scores = topk_scores[order]
                sorted_idx = valid[topk_idx[order]]
                # Higher scores are better; convert to distances for consistency.
                distances = -sorted_scores
                reranked_distances[i, :limit] = distances.astype(np.float32)
                reranked_indices[i, :limit] = sorted_idx.astype(np.int64)
            else:
                raise ValueError(f"Unsupported metric '{self.metric}' for FaissSearcher LSH rerank")

        return reranked_distances, reranked_indices

    def batch_search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if not self._prepared:
            raise RuntimeError("FaissSearcher not attached to an index")
        queries = self._prepare_query(queries)

        if self.index_kind == "lsh" and self._lsh_rerank and self._base_vectors is not None:
            distances, indices = self._batch_search_lsh_rerank(queries, k)
        else:
            distances, indices = self.index.search(queries, k)
            if self.metric in {"cosine", "ip"}:
                distances = -distances

        return distances.astype(np.float32), indices.astype(np.int64)


register_searcher("FaissSearcher", FaissSearcher)


class CompositeAlgorithm(BaseAlgorithm):
    """Adapter exposing an indexer/searcher pair as a standard algorithm."""

    def __init__(
        self,
        name: str,
        dimension: int,
        indexer: Dict[str, Any],
        searcher: Dict[str, Any],
        metric: str = "l2",
        **kwargs: Any,
    ) -> None:
        super().__init__(name, dimension)
        self.metric = metric
        self.extra_params = kwargs
        self.index_artifact: Optional[IndexArtifact] = None

        if not indexer or not searcher:
            raise ValueError("Both indexer_config and searcher_config must be provided for CompositeAlgorithm")

        self.indexer_config = copy.deepcopy(indexer)
        self.searcher_config = copy.deepcopy(searcher)

        self.indexer = self._instantiate_indexer(self.indexer_config)
        self.searcher = self._instantiate_searcher(self.searcher_config)

        # Persist overall configuration for reporting
        self.config = {
            "metric": self.metric,
            "indexer": self.indexer.describe(),
            "searcher": self.searcher.describe(),
        }
        if self.extra_params:
            self.config["params"] = copy.deepcopy(self.extra_params)

    def _instantiate_indexer(self, cfg: Dict[str, Any]) -> BaseIndexer:
        cfg_copy = copy.deepcopy(cfg)
        idx_type = cfg_copy.pop("type", None)
        if idx_type is None:
            raise ValueError("Indexer configuration must include a 'type' field")
        idx_name = cfg_copy.pop("name", idx_type)
        metric = cfg_copy.pop("metric", self.metric)
        indexer_cls = get_indexer_class(idx_type)
        return indexer_cls(name=idx_name, dimension=self.dimension, metric=metric, **cfg_copy)

    def _instantiate_searcher(self, cfg: Dict[str, Any]) -> BaseSearcher:
        cfg_copy = copy.deepcopy(cfg)
        search_type = cfg_copy.pop("type", None)
        if search_type is None:
            raise ValueError("Searcher configuration must include a 'type' field")
        search_name = cfg_copy.pop("name", search_type)
        metric = cfg_copy.pop("metric", self.metric)
        searcher_cls = get_searcher_class(search_type)
        return searcher_cls(name=search_name, dimension=self.dimension, metric=metric, **cfg_copy)

    def build_index(self, vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        self.index_artifact = self.indexer.build(vectors, metadata)
        self.searcher.attach(self.index_artifact, vectors, metadata)
        self.index_built = True

    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if not self.index_built:
            raise RuntimeError("Index has not been built for this algorithm")
        return self.searcher.search(query, k)

    def batch_search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if not self.index_built:
            raise RuntimeError("Index has not been built for this algorithm")
        return self.searcher.batch_search(queries, k)


__all__ = [
    "BaseIndexer",
    "BaseSearcher",
    "CompositeAlgorithm",
    "IndexArtifact",
    "register_indexer",
    "register_searcher",
    "INDEXER_REGISTRY",
    "SEARCHER_REGISTRY",
    "FaissLSHIndexer",
]
