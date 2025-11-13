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
        self._operation_counts: Dict[str, Any] = {"search_ops": 0.0}

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

    def reset_operation_counts(self) -> None:
        self._operation_counts = {"search_ops": 0.0}

    def record_search_ops(self, count: float, source: Optional[str] = None) -> None:
        current = float(self._operation_counts.get("search_ops", 0.0))
        self._operation_counts["search_ops"] = current + float(count)
        if source:
            source_key = "search_ops_source"
            existing = self._operation_counts.get(source_key)
            if existing is None:
                self._operation_counts[source_key] = source
            elif existing != source:
                self._operation_counts[source_key] = "mixed"

    def get_operation_counts(self) -> Dict[str, Any]:
        return dict(self._operation_counts)


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


def _prepare_faiss_stats(index: Any) -> Tuple[Optional[Any], Optional[str]]:
    if faiss is None:
        return None, None

    stats_obj: Optional[Any] = None
    source: Optional[str] = None

    try:
        if hasattr(index, "nlist") and hasattr(index, "nprobe") and hasattr(faiss, "cvar") and hasattr(faiss.cvar, "indexIVF_stats"):
            stats_obj = faiss.cvar.indexIVF_stats
            if hasattr(stats_obj, "reset"):
                stats_obj.reset()
                source = "faiss.stats.ivf"

        if stats_obj is None and hasattr(index, "hnsw"):
            hnsw = index.hnsw
            if hasattr(hnsw, "reset_stats"):
                hnsw.reset_stats()
                stats_candidate = getattr(hnsw, "stats", None)
                if stats_candidate is not None:
                    stats_obj = stats_candidate
                    source = "faiss.stats.hnsw"

        if stats_obj is None and hasattr(faiss, "cvar") and hasattr(faiss.cvar, "indexHNSW_stats"):
            stats_obj = faiss.cvar.indexHNSW_stats
            if hasattr(stats_obj, "reset"):
                stats_obj.reset()
                source = source or "faiss.stats.hnsw_global"
    except Exception:
        stats_obj = None
        source = None

    return stats_obj, source


def _extract_faiss_ops(stats_obj: Optional[Any]) -> Optional[float]:
    if stats_obj is None:
        return None
    for attr in ("ndis", "nb_distance_computations", "n_dis"):
        if hasattr(stats_obj, attr):
            try:
                return float(getattr(stats_obj, attr))
            except Exception:
                continue
    return None


def _estimate_faiss_ops(index: Any, n_queries: int) -> Tuple[float, str]:
    ntotal = float(getattr(index, "ntotal", 0))
    if ntotal < 0:
        ntotal = 0.0

    nlist = float(getattr(index, "nlist", 0))
    nprobe = float(getattr(index, "nprobe", 0))
    if nlist > 0 and nprobe > 0 and ntotal > 0:
        avg_list_size = ntotal / max(nlist, 1.0)
        estimate = float(n_queries) * nprobe * avg_list_size
        return estimate, "estimate.ivf_lists"

    if hasattr(index, "hnsw"):
        ef_search = float(getattr(index.hnsw, "efSearch", getattr(index, "efSearch", 0)))
        if ef_search > 0 and ntotal > 0:
            log_factor = max(1.0, float(np.log2(max(ntotal, 2.0))))
            estimate = float(n_queries) * ef_search * log_factor
            return estimate, "estimate.hnsw_ef"

    fallback = float(n_queries) * ntotal
    return fallback, "estimate.full_scan"


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
            ops = float(queries.shape[0]) * float(self._vectors.shape[0])
            self.record_search_ops(ops, source="linear_scan")
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
            ops = float(queries.shape[0]) * float(self._vectors.shape[0])
            self.record_search_ops(ops, source="linear_scan")
            return distances.astype(np.float32), sorted_idx.astype(np.int64)

        raise ValueError(f"Unsupported metric '{self.metric}' for LinearSearcher")


register_searcher("LinearSearcher", LinearSearcher)


class FaissSearcher(BaseSearcher):
    """Searcher that delegates to a FAISS index."""

    def attach(self, artifact: IndexArtifact, vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        if artifact.kind != "faiss":
            raise ValueError("FaissSearcher requires 'faiss' artifact")
        if faiss is None:
            raise ImportError("faiss is required for FaissSearcher")
        self.index = artifact.data
        artefact_metadata = artifact.metadata or {}
        self.metric = artefact_metadata.get("metric", self.metric)
        self.normalize_queries = artefact_metadata.get("normalize_queries", False)
        self._prepared = True

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

    def batch_search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if not self._prepared:
            raise RuntimeError("FaissSearcher not attached to an index")
        queries = self._prepare_query(queries)
        stats_obj, stats_source = _prepare_faiss_stats(self.index)
        distances, indices = self.index.search(queries, k)
        ops = _extract_faiss_ops(stats_obj)
        source = stats_source
        if ops is None:
            ops, source = _estimate_faiss_ops(self.index, queries.shape[0])
        self.record_search_ops(ops, source=source)

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
        # Stash configs and placeholders before BaseAlgorithm initialises counters.
        if not indexer or not searcher:
            raise ValueError("Both indexer_config and searcher_config must be provided for CompositeAlgorithm")

        self.indexer_config = copy.deepcopy(indexer)
        self.searcher_config = copy.deepcopy(searcher)
        self.indexer: Optional[BaseIndexer] = None
        self.searcher: Optional[BaseSearcher] = None

        super().__init__(name, dimension)
        self.metric = metric
        self.extra_params = kwargs
        self.index_artifact: Optional[IndexArtifact] = None

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

    def reset_operation_counts(self) -> None:
        super().reset_operation_counts()
        if hasattr(self.indexer, "reset_operation_counts"):
            self.indexer.reset_operation_counts()
        if hasattr(self.searcher, "reset_operation_counts"):
            self.searcher.reset_operation_counts()

    def get_operation_counts(self) -> Dict[str, Any]:
        counts = super().get_operation_counts()
        if hasattr(self.searcher, "get_operation_counts"):
            search_counts = self.searcher.get_operation_counts()
            if "search_ops" in search_counts:
                counts["search_ops"] = float(search_counts.get("search_ops", 0.0))
            for key, value in search_counts.items():
                if key == "search_ops":
                    continue
                counts.setdefault(key, value)
        return counts


__all__ = [
    "BaseIndexer",
    "BaseSearcher",
    "CompositeAlgorithm",
    "IndexArtifact",
    "register_indexer",
    "register_searcher",
    "INDEXER_REGISTRY",
    "SEARCHER_REGISTRY",
]
