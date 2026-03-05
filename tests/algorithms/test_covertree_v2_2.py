import numpy as np
import pytest

from src.algorithms.covertree_v2_2 import CoverTreeV2_2


def brute_force_neighbors(vectors: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    distances = np.linalg.norm(vectors - query, axis=1)
    return np.argsort(distances)[:k]


def test_covertree_v2_2_search_matches_brute_force() -> None:
    rng = np.random.default_rng(123)
    vectors = rng.standard_normal((256, 12)).astype(np.float32)
    query = rng.standard_normal(12).astype(np.float32)

    tree = CoverTreeV2_2(name="covertree_v2_2_test", dimension=12, metric="l2")
    tree.build_index(vectors)

    for k in (1, 5, 20):
        distances, indices = tree.search(query, k=k)
        assert distances.shape[0] == k
        assert indices.shape[0] == k

        expected = brute_force_neighbors(vectors, query, k)
        np.testing.assert_array_equal(indices, expected)


def test_covertree_v2_2_batch_search_shapes() -> None:
    rng = np.random.default_rng(321)
    vectors = rng.standard_normal((128, 4)).astype(np.float32)
    queries = rng.standard_normal((7, 4)).astype(np.float32)

    tree = CoverTreeV2_2(name="covertree_v2_2_batch", dimension=4, metric="l2")
    tree.build_index(vectors)

    distances, indices = tree.batch_search(queries, k=7)
    assert distances.shape == (7, 7)
    assert indices.shape == (7, 7)

    query = queries[3]
    expected = brute_force_neighbors(vectors, query, 7)
    np.testing.assert_array_equal(indices[3], expected)


def test_covertree_v2_2_records_distance_operations() -> None:
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((5, 3)).astype(np.float32)
    query = rng.standard_normal(3).astype(np.float32)

    tree = CoverTreeV2_2(name="covertree_v2_2_ops", dimension=3, metric="l2")

    tree._compute_distance_batch_to_1(query, vectors)
    np.testing.assert_allclose(tree.operation_counter["ndis"], 5)

    tree._compute_distance_batch_to_1(query, vectors[:2])
    np.testing.assert_allclose(tree.operation_counter["ndis"], 7)


def test_covertree_v2_2_persistence_roundtrip(tmp_path) -> None:
    rng = np.random.default_rng(2026)
    vectors = rng.standard_normal((96, 8)).astype(np.float32)
    queries = rng.standard_normal((5, 8)).astype(np.float32)
    artifact_dir = tmp_path / "covertree_artifact"

    original = CoverTreeV2_2(name="covertree_v2_2_original", dimension=8, metric="l2")
    original.build_index(vectors)
    original_results = [original.search(query, k=7) for query in queries]

    save_info = original.save_index(
        str(artifact_dir),
        context={
            "dataset_fingerprint": "fp-123",
            "build_metrics": {"build_time_s": 12.34},
        },
    )
    assert (artifact_dir / "WRITE_COMPLETE").exists()
    assert save_info["build_time_s"] == pytest.approx(12.34)

    restored = CoverTreeV2_2(name="covertree_v2_2_restored", dimension=8, metric="l2")
    load_info = restored.load_index(str(artifact_dir), context={"dataset_fingerprint": "fp-123"})
    assert load_info["build_time_s"] == pytest.approx(12.34)

    for query, (exp_distances, exp_indices) in zip(queries, original_results):
        got_distances, got_indices = restored.search(query, k=7)
        np.testing.assert_array_equal(got_indices, exp_indices)
        np.testing.assert_allclose(got_distances, exp_distances, rtol=0.0, atol=1e-7)


def test_covertree_v2_2_persistence_rejects_mismatch(tmp_path) -> None:
    rng = np.random.default_rng(2027)
    vectors = rng.standard_normal((48, 6)).astype(np.float32)
    artifact_dir = tmp_path / "covertree_artifact"

    tree = CoverTreeV2_2(name="covertree_v2_2_persist", dimension=6, metric="l2")
    tree.build_index(vectors)
    tree.save_index(str(artifact_dir), context={"dataset_fingerprint": "fp-good"})

    wrong_metric = CoverTreeV2_2(name="covertree_v2_2_wrong_metric", dimension=6, metric="cosine")
    with pytest.raises(ValueError, match="Metric mismatch"):
        wrong_metric.load_index(str(artifact_dir), context={"dataset_fingerprint": "fp-good"})

    wrong_fp = CoverTreeV2_2(name="covertree_v2_2_wrong_fp", dimension=6, metric="l2")
    with pytest.raises(ValueError, match="Dataset fingerprint mismatch"):
        wrong_fp.load_index(str(artifact_dir), context={"dataset_fingerprint": "fp-bad"})


def test_covertree_v2_2_persistence_requires_complete_artifact(tmp_path) -> None:
    rng = np.random.default_rng(2028)
    vectors = rng.standard_normal((32, 5)).astype(np.float32)
    artifact_dir = tmp_path / "covertree_artifact"

    tree = CoverTreeV2_2(name="covertree_v2_2_incomplete", dimension=5, metric="l2")
    tree.build_index(vectors)
    tree.save_index(str(artifact_dir))

    (artifact_dir / "WRITE_COMPLETE").unlink()
    restored = CoverTreeV2_2(name="covertree_v2_2_restore", dimension=5, metric="l2")
    with pytest.raises(FileNotFoundError, match="WRITE_COMPLETE"):
        restored.load_index(str(artifact_dir))
