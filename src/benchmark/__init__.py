from .dataset import Dataset
from .evaluation import Evaluator
from .metrics import recall_at_k, precision_at_k, mean_average_precision

__all__ = [
    "Dataset",
    "Evaluator",
    "BenchmarkRunner",
    "recall_at_k",
    "precision_at_k",
    "mean_average_precision",
]


def __getattr__(name):
    if name == "BenchmarkRunner":
        from .runner import BenchmarkRunner

        return BenchmarkRunner
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
