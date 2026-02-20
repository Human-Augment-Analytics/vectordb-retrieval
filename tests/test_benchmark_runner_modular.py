from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.benchmark.runner import BenchmarkRunner


try:
    import faiss  # type: ignore  # noqa: F401

    FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without faiss
    FAISS_AVAILABLE = False


def test_benchmark_runner_resolves_modular_components(tmp_path):
    config = {
        "indexers": {
            "bf_l2": {"type": "BruteForceIndexer", "metric": "l2"},
        },
        "searchers": {
            "linear_l2": {"type": "LinearSearcher", "metric": "l2"},
        },
        "algorithms": {
            "bf_linear": {
                "indexer_ref": "bf_l2",
                "searcher_ref": "linear_l2",
                "metric": "l2",
            }
        },
        "datasets": [
            {
                "name": "random",
                "metric": "l2",
                "dataset_options": {
                    "train_size": 32,
                    "test_size": 6,
                    "ground_truth_k": 5,
                    "dimensions": 3,
                    "seed": 123,
                },
                "n_queries": 5,
                "topk": 5,
            }
        ],
        "n_queries": 5,
        "topk": 5,
        "repeat": 1,
        "output_dir": str(tmp_path / "benchmark_outputs"),
        "seed": 11,
    }

    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(config))

    runner = BenchmarkRunner(str(config_path), output_dir=str(tmp_path / "fallback_outputs"))
    results = runner.run()

    assert "random" in results
    dataset_results = results["random"]
    assert "bf_linear" in dataset_results
    algo_metrics = dataset_results["bf_linear"]
    assert "recall@1" in algo_metrics
    assert algo_metrics["n_train"] == 32
    assert algo_metrics["n_test"] == 5

    # Verify summary artefacts created in expected directory.
    summary_candidates = list(Path(runner.output_dir).glob("benchmark_summary.md"))
    assert summary_candidates, "Benchmark summary file was not generated"


@pytest.mark.requires_faiss
@pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss is required for this test")
def test_benchmark_runner_hnsw_reports_ndis(tmp_path):
    config = {
        "algorithms": {
            "hnsw": {
                "type": "HNSW",
                "M": 8,
                "efConstruction": 40,
                "efSearch": 32,
                "metric": "l2",
            }
        },
        "datasets": [
            {
                "name": "random",
                "metric": "l2",
                "dataset_options": {
                    "train_size": 96,
                    "test_size": 10,
                    "ground_truth_k": 5,
                    "dimensions": 8,
                    "seed": 99,
                },
                "n_queries": 6,
                "topk": 5,
            }
        ],
        "n_queries": 6,
        "topk": 5,
        "repeat": 1,
        "output_dir": str(tmp_path / "benchmark_outputs"),
        "seed": 11,
    }

    config_path = tmp_path / "config_hnsw.yaml"
    config_path.write_text(json.dumps(config))

    runner = BenchmarkRunner(str(config_path), output_dir=str(tmp_path / "fallback_outputs"))
    results = runner.run()

    assert "random" in results
    dataset_results = results["random"]
    assert "hnsw" in dataset_results


    metrics = dataset_results["hnsw"]
    print(f"{metrics['ndis']=}")

    assert metrics["n_test"] == 6
    assert metrics["ndis"] is not None
    assert metrics["ndis"] > 0
    assert metrics["ndis_per_query"] is not None
    assert metrics["ndis_per_query"] > 0.0
