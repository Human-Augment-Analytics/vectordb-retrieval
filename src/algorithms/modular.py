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


class BruteForceIndexer(BaseIndexer):
    """Indexer that simply stores raw vectors in memory."""

    def build(self, vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> IndexArtifact:
        vector_store = vectors.astype(np.float32, copy=True)
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
        data = vectors.astype(np.float32, copy=True)
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


class FaissIVFIndexer(BaseIndexer):
    """Generic FAISS IVF-based indexer via index_factory."""

    def __init__(self, name: str, dimension: int, metric: str = "l2", index_type: str = "IVF100,Flat", **kwargs: Any) -> None:
        if faiss is None:
            raise ImportError("faiss is required for FaissIVFIndexer")
        super().__init__(name, dimension, metric, index_type=index_type, **kwargs)
        self.index_type = index_type
        self.training_params = kwargs

    def build(self, vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> IndexArtifact:
        metric_kind = faiss.METRIC_L2
        data = vectors.astype(np.float32, copy=True)
        artefact_metadata: Dict[str, Any] = {
            "metric": self.metric,
            "faiss_metric": "l2",
        }

        if self.metric == "cosine":
            data = _safe_normalize(data)
            metric_kind = faiss.METRIC_INNER_PRODUCT
            artefact_metadata.update({
                "faiss_metric": "ip",
                "normalize_queries": True,
                "normalize_vectors": True,
            })
        elif self.metric == "ip":
            metric_kind = faiss.METRIC_INNER_PRODUCT
            artefact_metadata["faiss_metric"] = "ip"

        index = faiss.index_factory(self.dimension, self.index_type, metric_kind)
        if not index.is_trained:
            index.train(data)
        index.add(data)

        if "nprobe" in self.params and hasattr(index, "nprobe"):
            index.nprobe = self.params["nprobe"]
            artefact_metadata["nprobe"] = self.params["nprobe"]

        return IndexArtifact(kind="faiss", data=index, metadata=artefact_metadata)


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
]
