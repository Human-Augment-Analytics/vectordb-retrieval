from __future__ import annotations

import numpy as np
import pytest

from src.algorithms.modular import CompositeAlgorithm


try:
    import faiss  # type: ignore  # noqa: F401

    FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without faiss
    FAISS_AVAILABLE = False


def brute_force_l2_neighbors(train: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Utility that returns expected neighbor indices using pure NumPy."""
    dists = np.linalg.norm(train[None, :, :] - queries[:, None, :], axis=2)
    indices = np.argsort(dists, axis=1)[:, :k]
    return indices


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def test_composite_bruteforce_linear_matches_numpy():
    train = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    queries = np.array(
        [
            [0.1, 0.1],
            [0.9, 0.2],
        ],
        dtype=np.float32,
    )

    algo = CompositeAlgorithm(
        name="bf_linear",
        dimension=train.shape[1],
        metric="l2",
        indexer={"type": "BruteForceIndexer", "metric": "l2"},
        searcher={"type": "LinearSearcher", "metric": "l2"},
    )
    algo.build_index(train)
    _, indices = algo.batch_search(queries, k=2)

    expected = brute_force_l2_neighbors(train, queries, k=2)
    np.testing.assert_array_equal(indices[:, :2], expected)


@pytest.mark.requires_faiss
@pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss is required for this test")
def test_composite_faiss_hnsw_runs_without_errors():
    train = np.random.RandomState(0).randn(64, 8).astype(np.float32)
    queries = np.random.RandomState(1).randn(5, 8).astype(np.float32)

    algo = CompositeAlgorithm(
        name="hnsw_faiss",
        dimension=train.shape[1],
        metric="l2",
        indexer={
            "type": "HNSWIndexer",
            "M": 8,
            "efConstruction": 40,
            "efSearch": 32,
            "metric": "l2",
        },
        searcher={"type": "FaissSearcher", "metric": "l2", "nprobe": 4},
    )

    algo.build_index(train)
    distances, indices = algo.batch_search(queries, k=3)

    assert distances.shape == (5, 3)
    assert indices.shape == (5, 3)


def test_composite_requires_indexer_and_searcher():
    train = np.random.RandomState(42).randn(10, 4).astype(np.float32)

    with pytest.raises(ValueError):
        CompositeAlgorithm(
            name="missing_indexer",
            dimension=train.shape[1],
            indexer={},
            searcher={"type": "LinearSearcher", "metric": "l2"},
        )

    with pytest.raises(ValueError):
        CompositeAlgorithm(
            name="missing_searcher",
            dimension=train.shape[1],
            indexer={"type": "BruteForceIndexer", "metric": "l2"},
            searcher={},
        )


def test_composite_lsh_cosine_recovers_identical_vectors():
    rng = np.random.RandomState(7)
    train = normalize_rows(rng.randn(128, 16).astype(np.float32))
    queries = train[:5].copy()

    algo = CompositeAlgorithm(
        name="lsh_cosine",
        dimension=train.shape[1],
        metric="cosine",
        indexer={
            "type": "LSHIndexer",
            "metric": "cosine",
            "num_tables": 12,
            "hash_size": 16,
            "seed": 7,
        },
        searcher={
            "type": "LSHSearcher",
            "metric": "cosine",
            "candidate_multiplier": 12.0,
            "fallback_to_bruteforce": True,
        },
    )
    algo.build_index(train)
    distances, indices = algo.batch_search(queries, k=1)

    np.testing.assert_allclose(distances[:, 0], 0.0, atol=1e-6)
    np.testing.assert_array_equal(indices[:, 0], np.arange(5))


def test_composite_lsh_l2_recovers_identical_vectors():
    rng = np.random.RandomState(11)
    train = rng.randn(160, 8).astype(np.float32)
    queries = train[10:20].copy()

    algo = CompositeAlgorithm(
        name="lsh_l2",
        dimension=train.shape[1],
        metric="l2",
        indexer={
            "type": "LSHIndexer",
            "metric": "l2",
            "num_tables": 10,
            "hash_size": 12,
            "bucket_width": 3.0,
            "seed": 11,
        },
        searcher={
            "type": "LSHSearcher",
            "metric": "l2",
            "candidate_multiplier": 10.0,
            "fallback_to_bruteforce": True,
        },
    )
    algo.build_index(train)
    distances, indices = algo.batch_search(queries, k=1)

    np.testing.assert_allclose(distances[:, 0], 0.0, atol=1e-6)
    np.testing.assert_array_equal(indices[:, 0], np.arange(10, 20))
