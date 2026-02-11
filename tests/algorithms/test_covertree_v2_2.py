import numpy as np

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
