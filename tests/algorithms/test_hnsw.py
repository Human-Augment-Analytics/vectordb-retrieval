from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.requires_faiss
pytest.importorskip("faiss")

from src.algorithms.hnsw import HNSW


def _make_data(seed: int = 0, n_vectors: int = 128, n_queries: int = 6, dim: int = 8) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n_vectors, dim)).astype(np.float32)
    queries = rng.standard_normal((n_queries, dim)).astype(np.float32)
    return vectors, queries


def _build_hnsw(vectors: np.ndarray) -> HNSW:
    algo = HNSW(
        name="hnsw_test",
        dimension=vectors.shape[1],
        M=8,
        efConstruction=40,
        efSearch=32,
        metric="l2",
    )
    algo.build_index(vectors)
    return algo


def test_hnsw_ndis_counter_single():
    vectors, queries = _make_data(seed=1)
    algo = _build_hnsw(vectors)

    assert algo.operation_counter["ndis"] == pytest.approx(0.0)
    distances, indices = algo.search(queries[0], k=5)

    assert distances.shape == (5,)
    assert indices.shape == (5,)
    assert algo.operation_counter["ndis"] > 0.0
    assert algo.operation_counter["ndis"] == pytest.approx(float(getattr(algo.stats, "ndis", 0.0)))


def test_hnsw_ndis_counter_batch():
    vectors, queries = _make_data(seed=2)
    algo = _build_hnsw(vectors)

    distances, indices = algo.batch_search(queries, k=4)

    assert distances.shape == (queries.shape[0], 4)
    assert indices.shape == (queries.shape[0], 4)
    assert algo.operation_counter["ndis"] > 0.0
    assert algo.operation_counter["ndis"] == pytest.approx(float(getattr(algo.stats, "ndis", 0.0)))


def test_hnsw_ndis_counter_reset():
    vectors, queries = _make_data(seed=3)
    algo = _build_hnsw(vectors)

    algo.search(queries[0], k=3)
    ndis_before_reset = float(getattr(algo.stats, "ndis", 0.0))
    assert ndis_before_reset > 0.0
    assert algo.operation_counter["ndis"] == pytest.approx(ndis_before_reset)

    algo.reset_operation_counters()
    assert float(getattr(algo.stats, "ndis", 0.0)) == pytest.approx(0.0)
    assert algo.operation_counter["ndis"] == pytest.approx(0.0)

    # operation_counter mirrors FAISS stats on search calls
    algo.search(queries[1], k=3)
    assert algo.operation_counter["ndis"] > 0.0
    assert algo.operation_counter["ndis"] == pytest.approx(float(getattr(algo.stats, "ndis", 0.0)))


def test_hnsw_ndis_counter_reset_between_runs():
    vectors_a, queries_a = _make_data(seed=4)
    run_a = _build_hnsw(vectors_a)
    run_a.batch_search(queries_a, k=5)
    assert run_a.operation_counter["ndis"] > 0.0

    vectors_b, queries_b = _make_data(seed=5)
    run_b = _build_hnsw(vectors_b)
    assert run_b.operation_counter["ndis"] == pytest.approx(0.0)
    assert float(getattr(run_b.stats, "ndis", 0.0)) == pytest.approx(0.0)

    run_b.search(queries_b[0], k=5)
    assert run_b.operation_counter["ndis"] > 0.0


def test_hnsw_ndis_counter_accumulates_across_single_searches():
    vectors, queries = _make_data(seed=6)
    algo = _build_hnsw(vectors)

    algo.search(queries[0], k=5)
    after_first = algo.operation_counter["ndis"]
    algo.search(queries[1], k=5)
    after_second = algo.operation_counter["ndis"]

    assert after_second > after_first > 0.0


def test_hnsw_ndis_counter_accumulates_single_then_batch():
    vectors, queries = _make_data(seed=7)
    algo = _build_hnsw(vectors)

    algo.search(queries[0], k=5)
    after_single = algo.operation_counter["ndis"]
    algo.batch_search(queries[1:], k=5)
    after_batch = algo.operation_counter["ndis"]

    assert after_batch > after_single > 0.0
