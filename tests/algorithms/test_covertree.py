import numpy as np

from src.algorithms.covertree import CoverTree


def brute_force_neighbors(vectors: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    distances = np.linalg.norm(vectors - query, axis=1)
    return np.argsort(distances)[:k]


def test_covertree_search_matches_brute_force() -> None:
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((128, 8)).astype(np.float32)
    query = rng.standard_normal(8).astype(np.float32)

    tree = CoverTree(
        name="covertree_test",
        dimension=8,
        metric="l2",
        candidate_pool_size=vectors.shape[0],
        max_visit_nodes=vectors.shape[0] * 2,
    )
    tree.build_index(vectors)

    distances, indices = tree.search(query, k=5)
    assert distances.shape[0] == 5
    assert indices.shape[0] == 5

    expected = brute_force_neighbors(vectors, query, 5)
    np.testing.assert_array_equal(indices, expected)


def test_batch_search_shapes() -> None:
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((64, 4)).astype(np.float32)
    queries = rng.standard_normal((3, 4)).astype(np.float32)

    tree = CoverTree(
        name="covertree_batch",
        dimension=4,
        metric="l2",
        candidate_pool_size=128,
        max_visit_nodes=256,
    )
    tree.build_index(vectors)

    distances, indices = tree.batch_search(queries, k=3)

    assert distances.shape == (3, 3)
    assert indices.shape == (3, 3)
    assert np.all(indices >= -1)
