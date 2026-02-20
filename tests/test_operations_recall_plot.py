from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from src.benchmark.evaluation import Evaluator
from src.experiments.config import ExperimentConfig
from src.experiments.experiment_runner import ExperimentRunner


def test_plot_operations_vs_recall_writes_output(tmp_path):
    ground_truth = np.array(
        [
            np.arange(10, dtype=np.int64),
            np.arange(10, 20, dtype=np.int64),
        ]
    )
    evaluator = Evaluator(ground_truth)

    strong_predictions = ground_truth.copy()
    weaker_predictions = np.array(
        [
            np.array([0, 1, 2, 3, 4, 200, 201, 202, 203, 204], dtype=np.int64),
            np.array([10, 11, 12, 300, 301, 302, 303, 304, 305, 306], dtype=np.int64),
        ]
    )

    evaluator.evaluate(
        "strong_algo",
        strong_predictions,
        np.array([0.001, 0.0012], dtype=np.float64),
    )
    evaluator.evaluate(
        "weaker_algo",
        weaker_predictions,
        np.array([0.004, 0.0035], dtype=np.float64),
    )

    output_path = tmp_path / "operations_vs_recall.png"
    evaluator.plot_operations_vs_recall(output_file=str(output_path), title_suffix="glove50")

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_operations_vs_recall_falls_back_to_qps(tmp_path):
    evaluator = Evaluator(np.zeros((1, 10), dtype=np.int64))
    evaluator.results = {
        "algo_a": {"recall@10": 0.9, "qps": 1000.0},
        "algo_b": {"recall@10": 0.7, "qps": 100.0},
    }

    output_path = tmp_path / "operations_vs_recall_qps_fallback.png"
    evaluator.plot_operations_vs_recall(output_file=str(output_path), title_suffix="glove50")

    assert output_path.exists()
    assert output_path.stat().st_size > 0


class _DummyEvaluator:
    def __init__(self):
        self.recall_vs_qps_calls = []
        self.operations_vs_recall_calls = []

    def plot_recall_vs_qps(self, output_file=None, title_suffix=None):
        self.recall_vs_qps_calls.append((output_file, title_suffix))
        if output_file:
            Path(output_file).write_text("recall_vs_qps")

    def plot_operations_vs_recall(self, output_file=None, title_suffix=None):
        self.operations_vs_recall_calls.append((output_file, title_suffix))
        if output_file:
            Path(output_file).write_text("operations_vs_recall")


def test_experiment_runner_generates_glove_operations_plot(tmp_path):
    config = ExperimentConfig(dataset="glove50", output_prefix="test_glove")
    runner = ExperimentRunner(config, output_dir=str(tmp_path))
    runner.experiment_id = "plot_test"

    evaluator = _DummyEvaluator()
    runner._generate_plots(evaluator)

    plots_dir = tmp_path / "plots_plot_test"
    assert (plots_dir / "recall_vs_qps.png").exists()
    assert (plots_dir / "operations_vs_recall.png").exists()
    assert len(evaluator.recall_vs_qps_calls) == 1
    assert len(evaluator.operations_vs_recall_calls) == 1


def test_experiment_runner_skips_operations_plot_for_non_glove(tmp_path):
    config = ExperimentConfig(dataset="random", output_prefix="test_random")
    runner = ExperimentRunner(config, output_dir=str(tmp_path))
    runner.experiment_id = "plot_test_random"

    evaluator = _DummyEvaluator()
    runner._generate_plots(evaluator)

    plots_dir = tmp_path / "plots_plot_test_random"
    assert (plots_dir / "recall_vs_qps.png").exists()
    assert not (plots_dir / "operations_vs_recall.png").exists()
    assert len(evaluator.recall_vs_qps_calls) == 1
    assert len(evaluator.operations_vs_recall_calls) == 0
