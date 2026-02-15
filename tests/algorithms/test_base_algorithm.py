import numpy as np

from src.algorithms.base_algorithm import BaseAlgorithm


class _DummyAlgorithm(BaseAlgorithm):
    def build_index(self, vectors: np.ndarray, metadata=None) -> None:
        return None

    def search(self, query: np.ndarray, k: int = 10):
        return np.array([]), np.array([])

    def batch_search(self, queries: np.ndarray, k: int = 10):
        return np.array([[]]), np.array([[]])


def test_record_operation_updates_counter():
    algo = _DummyAlgorithm(name="dummy", dimension=2)

    algo.record_operation("distance", 1)
    algo.record_operation("distance", 2.5)
    algo.record_operation("insert", 3)

    np.testing.assert_allclose(algo.operation_counter["distance"], 3.5)
    np.testing.assert_allclose(algo.operation_counter["insert"], 3)


def test_get_operations_returns_copy():
    algo = _DummyAlgorithm(name="dummy", dimension=2)

    algo.record_operation("distance", 1)
    operations = algo.get_operations()
    operations["distance"] = 10

    np.testing.assert_allclose(algo.operation_counter["distance"], 1)
