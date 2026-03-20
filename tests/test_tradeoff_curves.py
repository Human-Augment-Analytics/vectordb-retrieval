from __future__ import annotations

import json
from pathlib import Path

from src.benchmark.runner import BenchmarkRunner


def _build_runner(tmp_path: Path, config: dict) -> BenchmarkRunner:
    config_path = tmp_path / "benchmark_tradeoff_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return BenchmarkRunner(str(config_path), output_dir=str(tmp_path / "fallback_output"))


def test_tradeoff_variant_expansion_and_deterministic_downsampling(tmp_path):
    config = {
        "algorithms": {},
        "datasets": ["random"],
        "output_dir": str(tmp_path / "outputs"),
    }
    runner = _build_runner(tmp_path, config)

    algorithms = {
        "algo_a": {
            "type": "Composite",
            "metric": "l2",
            "indexer": {"type": "DummyIndexer", "nprobe": 1},
            "searcher": {"type": "DummySearcher", "ef": 1},
        }
    }
    tradeoff_cfg = {
        "enabled": True,
        "max_points_per_algorithm": 3,
        "algorithms": {
            "algo_a": {
                "grid": {
                    "indexer.nprobe": [1, 2, 3, 4],
                    "searcher.ef": [10, 20],
                }
            }
        },
    }

    expanded, metadata = runner._expand_tradeoff_algorithms(
        dataset_name="random",
        algorithms_for_dataset=algorithms,
        tradeoff_cfg=tradeoff_cfg,
    )

    assert list(expanded.keys()) == ["algo_a__p01", "algo_a__p02", "algo_a__p03"]
    assert list(metadata.keys()) == ["algo_a__p01", "algo_a__p02", "algo_a__p03"]

    assert expanded["algo_a__p01"]["indexer"]["nprobe"] == 1
    assert expanded["algo_a__p01"]["searcher"]["ef"] == 10
    assert expanded["algo_a__p02"]["indexer"]["nprobe"] == 3
    assert expanded["algo_a__p02"]["searcher"]["ef"] == 10
    assert expanded["algo_a__p03"]["indexer"]["nprobe"] == 4
    assert expanded["algo_a__p03"]["searcher"]["ef"] == 20

    assert metadata["algo_a__p01"]["is_tradeoff_variant"] is True
    assert metadata["algo_a__p01"]["tradeoff_variant_id"] == "p01"
    assert metadata["algo_a__p01"]["base_algorithm"] == "algo_a"
    assert metadata["algo_a__p01"]["tradeoff_params"] == {
        "indexer.nprobe": 1,
        "searcher.ef": 10,
    }


def test_tradeoff_invalid_path_raises_clear_error(tmp_path):
    config = {
        "algorithms": {},
        "datasets": ["random"],
        "output_dir": str(tmp_path / "outputs"),
    }
    runner = _build_runner(tmp_path, config)

    algorithms = {
        "algo_a": {
            "type": "Composite",
            "metric": "l2",
            "indexer": {"type": "DummyIndexer", "nprobe": 1},
            "searcher": {"type": "DummySearcher", "ef": 1},
        }
    }
    tradeoff_cfg = {
        "enabled": True,
        "max_points_per_algorithm": 12,
        "algorithms": {
            "algo_a": {
                "grid": {
                    "indexer.missing_key": [1, 2],
                }
            }
        },
    }

    try:
        runner._expand_tradeoff_algorithms(
            dataset_name="random",
            algorithms_for_dataset=algorithms,
            tradeoff_cfg=tradeoff_cfg,
        )
    except ValueError as exc:
        message = str(exc)
        assert "Invalid tradeoff grid path" in message
        assert "indexer.missing_key" in message
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected ValueError for invalid tradeoff grid path")


def test_tradeoff_pareto_frontier_filters_dominated_points(tmp_path):
    config = {
        "algorithms": {},
        "datasets": ["random"],
        "output_dir": str(tmp_path / "outputs"),
    }
    runner = _build_runner(tmp_path, config)

    points = [
        {"variant_id": "p01", "algorithm_name": "algo_a__p01", "qps": 100.0, "recall": 0.95, "params": {}},
        {"variant_id": "p02", "algorithm_name": "algo_a__p02", "qps": 150.0, "recall": 0.92, "params": {}},
        {"variant_id": "p03", "algorithm_name": "algo_a__p03", "qps": 130.0, "recall": 0.90, "params": {}},
        {"variant_id": "p04", "algorithm_name": "algo_a__p04", "qps": 220.0, "recall": 0.80, "params": {}},
        {"variant_id": "p05", "algorithm_name": "algo_a__p05", "qps": 260.0, "recall": 0.78, "params": {}},
    ]

    frontier = runner._compute_tradeoff_pareto_frontier_points(points)
    frontier_ids = [point["variant_id"] for point in frontier]

    assert frontier_ids == ["p01", "p02", "p04", "p05"]


def test_tradeoff_curves_generated_and_embedded_in_summary(tmp_path):
    config = {
        "indexers": {
            "bf_l2": {"type": "BruteForceIndexer", "metric": "l2"},
            "lsh_idx_l2": {
                "type": "LSHIndexer",
                "metric": "l2",
                "num_tables": 4,
                "hash_size": 2,
                "bucket_width": 10.0,
                "seed": 42,
            },
        },
        "searchers": {
            "linear_l2": {"type": "LinearSearcher", "metric": "l2"},
            "lsh_search_l2": {
                "type": "LSHSearcher",
                "metric": "l2",
                "candidate_multiplier": 8.0,
                "fallback_to_bruteforce": False,
            },
        },
        "algorithms": {
            "exact": {
                "indexer_ref": "bf_l2",
                "searcher_ref": "linear_l2",
                "metric": "l2",
            },
            "lsh": {
                "indexer_ref": "lsh_idx_l2",
                "searcher_ref": "lsh_search_l2",
                "metric": "l2",
            },
        },
        "tradeoff_curves": {
            "enabled": True,
            "max_points_per_algorithm": 3,
            "algorithms": {
                "lsh": {
                    "grid": {
                        "indexer.hash_size": [2, 3, 4, 5],
                        "searcher.candidate_multiplier": [4.0, 8.0],
                    }
                }
            },
        },
        "datasets": [
            {
                "name": "random",
                "metric": "l2",
                "n_queries": 8,
                "topk": 5,
                "dataset_options": {
                    "dimensions": 8,
                    "train_size": 64,
                    "test_size": 16,
                    "ground_truth_k": 5,
                    "seed": 17,
                },
            }
        ],
        "n_queries": 8,
        "topk": 5,
        "repeat": 1,
        "seed": 7,
        "output_dir": str(tmp_path / "benchmark_outputs"),
    }

    runner = _build_runner(tmp_path, config)
    results = runner.run()
    dataset_results = results["random"]

    assert "exact" in dataset_results
    lsh_variants = sorted(name for name in dataset_results.keys() if name.startswith("lsh__p"))
    assert lsh_variants == ["lsh__p01", "lsh__p02", "lsh__p03"]

    exact_metrics = dataset_results["exact"]
    assert exact_metrics["base_algorithm"] == "exact"
    assert exact_metrics["tradeoff_variant_id"] == "base"
    assert exact_metrics["is_tradeoff_variant"] is False
    assert exact_metrics["tradeoff_params"] == {}

    for variant_name in lsh_variants:
        metrics = dataset_results[variant_name]
        assert metrics["base_algorithm"] == "lsh"
        assert metrics["is_tradeoff_variant"] is True
        assert metrics["tradeoff_variant_id"].startswith("p")
        assert "indexer.hash_size" in metrics["tradeoff_params"]
        assert "searcher.candidate_multiplier" in metrics["tradeoff_params"]

    tradeoff_dir = Path(runner.output_dir) / "random" / "tradeoff_curves"
    assert (tradeoff_dir / "tradeoff_random_exact_recall_qps.svg").exists()
    assert (tradeoff_dir / "tradeoff_random_lsh_recall_qps.svg").exists()
    assert (tradeoff_dir / "tradeoff_random_exact.json").exists()
    assert (tradeoff_dir / "tradeoff_random_lsh.json").exists()
    assert (tradeoff_dir / "tradeoff_random_exact_quadrant.mmd").exists()
    assert (tradeoff_dir / "tradeoff_random_lsh_quadrant.mmd").exists()
    lsh_mermaid_text = (tradeoff_dir / "tradeoff_random_lsh_quadrant.mmd").read_text(encoding="utf-8")
    assert "quadrantChart" in lsh_mermaid_text
    assert "p01:" in lsh_mermaid_text
    assert "_pareto:" in lsh_mermaid_text
    lsh_tradeoff_payload = json.loads((tradeoff_dir / "tradeoff_random_lsh.json").read_text(encoding="utf-8"))
    assert "pareto_frontier_points" in lsh_tradeoff_payload
    assert len(lsh_tradeoff_payload["pareto_frontier_points"]) >= 1
    assert "mermaid_quadrant_chart" in lsh_tradeoff_payload
    assert "quadrantChart" in lsh_tradeoff_payload["mermaid_quadrant_chart"]

    summary_text = (Path(runner.output_dir) / "one-page-summary.md").read_text(encoding="utf-8")
    assert "## Table of Contents" in summary_text
    assert "- [Dataset: random](#dataset-random)" in summary_text
    assert "- [random / Tradeoff Curves by Algorithm](#tradeoff-random)" in summary_text
    assert "- [random / Algorithm Implementation Details](#algo-details-random)" in summary_text
    assert "- [random / Dataset Details](#dataset-details-random)" in summary_text
    assert '<a id="dataset-random"></a>' in summary_text
    assert '<a id="tradeoff-random"></a>' in summary_text
    assert '<a id="algo-details-random"></a>' in summary_text
    assert '<a id="dataset-details-random"></a>' in summary_text
    assert "### Tradeoff Curves by Algorithm" in summary_text
    assert "##### Pareto Points (Non-dominated Frontier)" in summary_text
    assert "##### Mermaid Quadrant Chart" in summary_text
    assert "_Point IDs ending with `_pareto` mark Pareto points._" in summary_text
    assert "```mermaid" in summary_text
    assert "Mermaid source: `./random/tradeoff_curves/tradeoff_random_lsh_quadrant.mmd`" in summary_text
    assert "./random/tradeoff_curves/tradeoff_random_exact_recall_qps.svg" in summary_text
    assert "./random/tradeoff_curves/tradeoff_random_lsh_recall_qps.svg" in summary_text
    assert "./random/tradeoff_curves/tradeoff_random_exact.json" in summary_text
    assert "./random/tradeoff_curves/tradeoff_random_lsh.json" in summary_text
    assert "| Variant | QPS | Recall | Parameters |" in summary_text
    assert "| p01 |" in summary_text
    assert "searcher.candidate_multiplier=" in summary_text
