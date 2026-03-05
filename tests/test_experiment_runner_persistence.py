from __future__ import annotations

import copy

import pytest

from src.algorithms import get_algorithm_instance
from src.experiments.config import ExperimentConfig
from src.experiments.experiment_runner import ExperimentRunner


def _register_algorithms(runner: ExperimentRunner) -> None:
    runner.load_dataset()
    assert runner.dataset is not None
    dimension = runner.dataset.train_vectors.shape[1]
    for alg_name, alg_cfg_orig in runner.config.algorithms.items():
        alg_cfg = copy.deepcopy(alg_cfg_orig)
        alg_type = alg_cfg.pop("type")
        algorithm = get_algorithm_instance(alg_type, dimension, name=alg_name, **alg_cfg)
        runner.register_algorithm(algorithm, name=alg_name)


def _base_algorithms_config(mode: str, artifact_dir: str) -> dict:
    return {
        "covertree_v2_2": {
            "type": "CoverTreeV2_2",
            "metric": "l2",
            "persistence": {
                "enabled": True,
                "mode": mode,
                "artifact_dir": artifact_dir,
                "path_policy": "fixed",
                "force_rebuild": False,
                "fail_if_missing": True,
            },
        }
    }


def test_experiment_runner_build_only_then_retrieve_only(tmp_path) -> None:
    artifact_dir = tmp_path / "covertree_artifact"

    common_kwargs = {
        "dataset": "random",
        "data_dir": str(tmp_path / "data"),
        "dataset_options": {
            "dimensions": 6,
            "train_size": 96,
            "test_size": 24,
            "ground_truth_k": 10,
            "seed": 99,
        },
        "n_queries": 12,
        "topk": 5,
        "query_batch_size": 4,
        "seed": 123,
    }

    build_cfg = ExperimentConfig(
        **common_kwargs,
        output_prefix="persist_build",
        algorithms=_base_algorithms_config("build_only", str(artifact_dir)),
    )
    build_runner = ExperimentRunner(build_cfg, output_dir=str(tmp_path / "build_results"))
    _register_algorithms(build_runner)
    build_results = build_runner.run()
    build_metrics = build_results["covertree_v2_2"]

    assert build_metrics["status"] == "build_only"
    assert build_metrics["index_source"] == "built"
    assert build_metrics["build_time_s"] > 0
    assert (artifact_dir / "WRITE_COMPLETE").exists()

    retrieve_cfg = ExperimentConfig(
        **common_kwargs,
        output_prefix="persist_retrieve",
        algorithms=_base_algorithms_config("retrieve_only", str(artifact_dir)),
    )
    retrieve_runner = ExperimentRunner(retrieve_cfg, output_dir=str(tmp_path / "retrieve_results"))
    _register_algorithms(retrieve_runner)
    retrieve_results = retrieve_runner.run()
    retrieve_metrics = retrieve_results["covertree_v2_2"]

    assert retrieve_metrics["index_source"] == "loaded"
    assert retrieve_metrics["build_time_s"] > 0
    assert retrieve_metrics["index_load_time_s"] >= 0
    assert "recall@1" in retrieve_metrics
    assert retrieve_metrics.get("status") != "build_only"


def test_experiment_runner_retrieve_only_missing_artifact_fails(tmp_path) -> None:
    missing_artifact = tmp_path / "missing_artifact"
    cfg = ExperimentConfig(
        dataset="random",
        data_dir=str(tmp_path / "data"),
        dataset_options={
            "dimensions": 4,
            "train_size": 48,
            "test_size": 12,
            "ground_truth_k": 5,
            "seed": 77,
        },
        n_queries=8,
        topk=3,
        output_prefix="persist_missing",
        seed=7,
        algorithms=_base_algorithms_config("retrieve_only", str(missing_artifact)),
    )

    runner = ExperimentRunner(cfg, output_dir=str(tmp_path / "results"))
    _register_algorithms(runner)

    with pytest.raises(FileNotFoundError, match="Missing persisted index"):
        runner.run()
