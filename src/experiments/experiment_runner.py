"""Core experiment execution utilities."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import numpy as np
import yaml

from .config import ExperimentConfig
from ..algorithms.base_algorithm import BaseAlgorithm
from ..benchmark.dataset import Dataset
from ..benchmark.evaluation import Evaluator


class ExperimentRunner:
    """Execute vector-retrieval experiments for a dataset and algorithm set."""

    def __init__(self, config: ExperimentConfig, output_dir: str = "results") -> None:
        self.config = config
        self.output_dir = output_dir
        self.dataset: Optional[Dataset] = None
        self.algorithms: Dict[str, BaseAlgorithm] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = logging.getLogger("experiment_runner")

    # ------------------------------------------------------------------
    # Dataset & algorithm registration
    # ------------------------------------------------------------------
    def load_dataset(self) -> None:
        """Load the dataset defined in the configuration if not already loaded."""
        if self.dataset is not None:
            return

        self.logger.info(f"Loading dataset: {self.config.dataset}")
        dataset = Dataset(self.config.dataset, self.config.data_dir, options=self.config.dataset_options)
        dataset.load(force_download=self.config.force_download)
        self.dataset = dataset

    def register_algorithm(self, algorithm: BaseAlgorithm, name: Optional[str] = None) -> None:
        """Register an algorithm implementation for the current experiment."""
        if not isinstance(algorithm, BaseAlgorithm):
            raise TypeError("algorithm must inherit from BaseAlgorithm")

        algorithm_name = name or algorithm.get_name()
        if not algorithm_name:
            raise ValueError("Algorithm name must be provided")

        # Keep the algorithm's internal name synchronized with registration name.
        if getattr(algorithm, "name", None) != algorithm_name:
            setattr(algorithm, "name", algorithm_name)

        self.algorithms[algorithm_name] = algorithm
        self.logger.info(f"Registered algorithm: {algorithm_name}")

    # ------------------------------------------------------------------
    # Core execution flow
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Dict[str, Any]]:
        """Run all registered algorithms and return their evaluation metrics."""
        if not self.algorithms:
            raise RuntimeError("No algorithms registered for the experiment")

        self.load_dataset()
        assert self.dataset is not None  # mypy guard

        np.random.seed(self.config.seed)

        train_vectors = self.dataset.train_vectors
        test_queries = self.dataset.test_vectors
        ground_truth = self.dataset.ground_truth

        if train_vectors is None or test_queries is None:
            raise RuntimeError("Dataset did not provide train/test vectors")

        test_queries, ground_truth = self._select_query_subset(test_queries, ground_truth)

        algorithm_outputs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.results = {}

        for name, algorithm in self.algorithms.items():
            self.logger.info(f"Running experiment for algorithm: {name}")
            metrics, indices, query_times = self._run_single_algorithm(
                name, algorithm, train_vectors, test_queries
            )
            self.results[name] = metrics
            algorithm_outputs[name] = (indices, query_times)

        evaluator: Optional[Evaluator] = None
        if ground_truth is not None:
            evaluator = Evaluator(ground_truth)
            for name, (indices, query_times) in algorithm_outputs.items():
                metrics = evaluator.evaluate(name, indices, query_times)
                self.results[name].update(metrics)

                summary_k = min(100, self.config.topk)
                summary_key = f"recall@{summary_k}"

                if summary_key in metrics:
                    self.results[name]["recall"] = metrics[summary_key]
                elif metrics:
                    # Fallback to the largest available recall metric
                    available_recalls = sorted(
                        (key for key in metrics if key.startswith("recall@")),
                        key=lambda x: int(x.split("@")[-1])
                    )
                    if available_recalls:
                        self.results[name]["recall"] = metrics[available_recalls[-1]]

        for name in self.results:
            self._save_algorithm_results(name, self.results[name])

        self._save_combined_results()

        if evaluator is not None:
            evaluator.print_results()
            self._generate_plots(evaluator)

        self.logger.info("Experiment completed successfully.")
        return self.results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _select_query_subset(
        self, queries: np.ndarray, ground_truth: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply the n_queries limit while keeping ground-truth aligned."""
        n_available = len(queries)
        target = min(self.config.n_queries, n_available) if self.config.n_queries else n_available

        if target >= n_available:
            return queries, ground_truth

        indices = np.random.choice(n_available, target, replace=False)
        queries_subset = queries[indices]
        ground_truth_subset = ground_truth[indices] if ground_truth is not None else None

        self.logger.info(f"Using {target}/{n_available} queries for evaluation")
        return queries_subset, ground_truth_subset

    def _run_single_algorithm(
        self,
        name: str,
        algorithm: BaseAlgorithm,
        train_vectors: np.ndarray,
        test_queries: np.ndarray,
    ) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        """Train the algorithm, execute queries, and collect core metrics."""
        # Build phase
        build_start = time.time()
        if hasattr(algorithm, "reset_operation_counts"):
            algorithm.reset_operation_counts()
        algorithm.build_index(train_vectors)
        build_time = time.time() - build_start

        memory_usage_mb = self._estimate_memory_usage(algorithm, train_vectors)

        # Search phase
        k = self.config.topk
        indices = np.full((len(test_queries), k), -1, dtype=np.int64)
        query_times = np.zeros(len(test_queries), dtype=float)

        total_query_time = 0.0
        used_batch_api = False

        def _normalize_batch_indices(
            batch_result: Any, expected_rows: int, expected_k: int
        ) -> np.ndarray:
            if isinstance(batch_result, tuple):
                if len(batch_result) != 2:
                    raise ValueError("batch_search must return (distances, indices)")
                batch_result = batch_result[1]

            if isinstance(batch_result, list):
                rows = [np.asarray(row) for row in batch_result]
                output = np.full((expected_rows, expected_k), -1, dtype=np.int64)
                for row_idx, row in enumerate(rows[:expected_rows]):
                    limit = min(row.size, expected_k)
                    if limit > 0:
                        output[row_idx, :limit] = row[:limit]
                return output

            arr = np.asarray(batch_result)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.ndim != 2:
                raise ValueError("batch_search returned array with unexpected shape")
            if arr.shape[0] != expected_rows:
                if expected_rows == 1 and arr.shape[0] == expected_k:
                    arr = arr.reshape(1, -1)
                else:
                    raise ValueError(
                        f"batch_search returned {arr.shape[0]} rows, expected {expected_rows}"
                    )

            if arr.shape[1] < expected_k:
                padded = np.full((expected_rows, expected_k), -1, dtype=np.int64)
                padded[:, :arr.shape[1]] = arr
                arr = padded
            elif arr.shape[1] > expected_k:
                arr = arr[:, :expected_k]

            return arr.astype(np.int64, copy=False)

        if len(test_queries) > 0:
            batch_size_cfg = int(getattr(self.config, "query_batch_size", 0) or 0)
            if batch_size_cfg < 0:
                batch_size_cfg = 0
            batch_size = len(test_queries) if batch_size_cfg == 0 else min(batch_size_cfg, len(test_queries))

            try:
                cursor = 0
                while cursor < len(test_queries):
                    end = min(cursor + batch_size, len(test_queries))
                    batch_queries = test_queries[cursor:end]
                    batch_start = time.time()
                    batch_result = algorithm.batch_search(batch_queries, k=k)
                    elapsed = time.time() - batch_start
                    batch_indices = _normalize_batch_indices(batch_result, end - cursor, k)
                    indices[cursor:end] = batch_indices
                    per_query = elapsed / max((end - cursor), 1)
                    query_times[cursor:end] = per_query
                    total_query_time += elapsed
                    cursor = end

                used_batch_api = True
            except (AttributeError, NotImplementedError, TypeError, ValueError):
                # Reset to ensure clean fallback path
                indices.fill(-1)
                query_times.fill(0.0)
                total_query_time = 0.0

        if not used_batch_api:
            for idx, query in enumerate(test_queries):
                single_start = time.time()
                _, single_indices = algorithm.search(query, k=k)
                query_duration = time.time() - single_start
                query_times[idx] = query_duration
                indices[idx] = single_indices
                total_query_time += query_duration

        total_query_time = max(total_query_time, query_times.sum())
        mean_query_time_ms = (
            (total_query_time / max(len(test_queries), 1)) * 1000.0
            if len(test_queries) > 0
            else 0.0
        )

        qps = len(test_queries) / total_query_time if total_query_time > 0 else 0.0

        metrics: Dict[str, Any] = {
            "algorithm": name,
            "parameters": algorithm.get_parameters(),
            "dataset": self.config.dataset,
            "n_train": int(train_vectors.shape[0]),
            "n_test": int(len(test_queries)),
            "dimensions": int(train_vectors.shape[1]),
            "topk": self.config.topk,
            "build_time_s": float(build_time),
            "index_memory_mb": float(memory_usage_mb),
            "qps": float(qps),
            "mean_query_time_ms": float(mean_query_time_ms),
            "total_query_time_s": float(total_query_time),
            "timestamp": datetime.now().isoformat(),
        }

        op_counts = getattr(algorithm, "get_operation_counts", lambda: {})()
        if op_counts:
            search_ops = float(op_counts.get("search_ops", 0.0))
            metrics["vector_similarity_ops"] = search_ops
            per_query = search_ops / max(len(test_queries), 1)
            metrics["vector_similarity_ops_per_query"] = per_query
            source = op_counts.get("search_ops_source")
            if source:
                metrics["vector_similarity_ops_source"] = source

        return metrics, indices, query_times

    def _estimate_memory_usage(self, algorithm: BaseAlgorithm, train_vectors: np.ndarray) -> float:
        """Return algorithm-reported memory when available, otherwise estimate."""
        if hasattr(algorithm, "get_memory_usage"):
            try:
                usage = algorithm.get_memory_usage()
                if usage is not None:
                    return float(usage)
            except Exception:
                pass

        # Fallback: size of the raw training vectors in MB.
        return float(train_vectors.nbytes) / (1024.0 * 1024.0)

    def _save_algorithm_results(self, name: str, metrics: Dict[str, Any]) -> None:
        path = os.path.join(self.output_dir, f"{name}_results.json")
        with open(path, "w") as handle:
            json.dump(metrics, handle, indent=2)

    def _save_combined_results(self) -> None:
        combined_path = os.path.join(
            self.output_dir, f"{self.config.output_prefix}_all_results.json"
        )
        with open(combined_path, "w") as handle:
            json.dump(self.results, handle, indent=2)

        config_path = os.path.join(
            self.output_dir, f"{self.config.output_prefix}_{self.experiment_id}_config.yaml"
        )
        with open(config_path, "w") as handle:
            yaml.safe_dump(self.config.to_dict(), handle)

    def _generate_plots(self, evaluator: Evaluator) -> None:
        plots_dir = os.path.join(self.output_dir, f"plots_{self.experiment_id}")
        os.makedirs(plots_dir, exist_ok=True)

        plot_path = os.path.join(plots_dir, "recall_vs_qps.png")
        evaluator.plot_recall_vs_qps(
            output_file=plot_path,
            title_suffix=self.config.dataset,
        )
