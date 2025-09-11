import numpy as np
import time
import os
from typing import Dict, List, Any, Optional, Type
import matplotlib.pyplot as plt
from datetime import datetime
import yaml
import logging
from ..algorithms.base_algorithm import BaseAlgorithm
from ..benchmark.dataset import Dataset
from ..benchmark.evaluation import Evaluator
from .config import ExperimentConfig

class ExperimentRunner:
    """
    Class for running vector retrieval algorithm experiments.
    This is the core experimental loop for comparing different algorithms.
    """

    def __init__(self, config: ExperimentConfig, output_dir: str = "results"):
        """
        Initialize the experiment runner.

        Args:
            config: Configuration for the experiment
            output_dir: Directory to save results
        """
        self.config = config
        self.output_dir = output_dir
        self.dataset = None
        self.algorithms = {}
        self.results = {}
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Set up logging
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, f"experiment_{self.experiment_id}.log")
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                          handlers=[logging.FileHandler(log_file),
                                   logging.StreamHandler()])
        self.logger = logging.getLogger("ExperimentRunner")

        # Log configuration
        self.logger.info(f"Initializing experiment {self.experiment_id}")
        self.logger.info(f"Configuration: {config}")

    def load_dataset(self):
        """
        Load the dataset specified in the configuration.
        """
        self.logger.info(f"Loading dataset: {self.config.dataset}")
        self.dataset = Dataset(self.config.dataset, self.config.data_dir)
        self.dataset.load(force_download=self.config.force_download)

    def register_algorithm(self, algorithm: BaseAlgorithm):
        """
        Register an algorithm to be included in the experiment.

        Args:
            algorithm: Algorithm to register
        """
        name = algorithm.get_name()
        self.logger.info(f"Registering algorithm: {name}")
        self.algorithms[name] = algorithm

    def _build_indices(self, train_vectors: np.ndarray):
        """
        Build indices for all registered algorithms.

        Args:
            train_vectors: Training vectors to index
        """
        for name, algorithm in self.algorithms.items():
            self.logger.info(f"Building index for {name}...")
            start_time = time.time()
            algorithm.build_index(train_vectors)
            build_time = time.time() - start_time

            # Store build time
            algorithm.build_time = build_time
            self.logger.info(f"Index for {name} built in {build_time:.2f} seconds")

            # Estimate memory usage (baseline: size of raw vectors).
            # A more accurate measure would require algorithm-specific implementation.
            memory_usage_mb = train_vectors.nbytes / (1024 * 1024)
            algorithm.index_memory_usage = memory_usage_mb
            self.logger.info(f"Estimated base memory usage for {name} index: {memory_usage_mb:.2f} MB")

            # Store build time and memory usage in results
            if name not in self.results:
                self.results[name] = {}
            self.results[name]['build_time'] = build_time
            self.results[name]['index_memory_usage_mb'] = memory_usage_mb

    def _run_searches(self, test_vectors: np.ndarray, k: int):
        """
        Run search for all test vectors against all algorithms.

        Args:
            test_vectors: Test vectors to search for
            k: Number of nearest neighbors to retrieve

        Returns:
            Dictionary mapping algorithm names to (indices, query_times)
        """
        search_results = {}

        for name, algorithm in self.algorithms.items():
            self.logger.info(f"Running searches for {name}...")
            n_queries = len(test_vectors)
            indices = np.zeros((n_queries, k), dtype=np.int32)
            query_times = np.zeros(n_queries)

            for i in range(n_queries):
                start_time = time.time()
                _, idx = algorithm.search(test_vectors[i], k=k)
                query_times[i] = time.time() - start_time
                indices[i] = idx

                # Log progress
                if (i+1) % 100 == 0 or i+1 == n_queries:
                    self.logger.info(f"  {i+1}/{n_queries} queries processed")

            # Store search results
            search_results[name] = (indices, query_times)
            qps = 1.0 / np.mean(query_times)  # Queries per second
            self.logger.info(f"Search completed: {qps:.2f} queries per second")
import os
import time
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional

from .config import ExperimentConfig
from ..benchmark.dataset import Dataset
from ..algorithms.base import BaseAlgorithm

class ExperimentRunner:
    """
    Runs experiments for a specific dataset and multiple algorithms.
    """

    def __init__(self, config: ExperimentConfig, output_dir: str = "results"):
        """
        Initialize the experiment runner.

        Args:
            config: Experiment configuration
            output_dir: Directory to save results
        """
        self.config = config
        self.output_dir = output_dir
        self.dataset = None
        self.algorithms = {}
        self.results = {}
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = logging.getLogger("experiment_runner")

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def load_dataset(self) -> None:
        """
        Load the dataset based on configuration.
        """
        self.logger.info(f"Loading dataset: {self.config.dataset}")
        self.dataset = Dataset(self.config.dataset, self.config.data_dir)
        self.dataset.load(force_download=self.config.force_download)

    def register_algorithm(self, name: str, algorithm: BaseAlgorithm) -> None:
        """
        Register an algorithm for benchmarking.

        Args:
            name: Name of the algorithm
            algorithm: Algorithm instance
        """
        self.algorithms[name] = algorithm
        self.logger.info(f"Registered algorithm: {name}")

    def run(self) -> Dict[str, Dict[str, Any]]:
        """
        Run the experiment for all registered algorithms.

        Returns:
            Dictionary with results for each algorithm
        """
        if self.dataset is None:
            self.load_dataset()

        # Set random seed for reproducibility
        np.random.seed(self.config.seed)

        # Get train and test data
        train_data = self.dataset.train_vectors
        test_queries = self.dataset.test_vectors
        ground_truth = self.dataset.ground_truth

        # Limit number of queries if specified
        if self.config.n_queries and self.config.n_queries < len(test_queries):
            self.logger.info(f"Using {self.config.n_queries} out of {len(test_queries)} available queries")
            query_indices = np.random.choice(len(test_queries), self.config.n_queries, replace=False)
            test_queries = test_queries[query_indices]
            if ground_truth is not None:
                ground_truth = ground_truth[query_indices]

        # Run experiments for each algorithm
        results = {}
        for alg_name, algorithm in self.algorithms.items():
            self.logger.info(f"Running experiment for algorithm: {alg_name}")
            alg_results = self._run_algorithm_experiment(alg_name, algorithm, train_data, test_queries, ground_truth)
            results[alg_name] = alg_results

            # Save individual results
            results_file = os.path.join(self.output_dir, f"{alg_name}_results.json")
            with open(results_file, 'w') as f:
                json.dump(alg_results, f, indent=2)

        # Save combined results
        combined_results_file = os.path.join(self.output_dir, f"{self.config.output_prefix}_all_results.json")
        with open(combined_results_file, 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def _run_algorithm_experiment(self, 
                                  alg_name: str, 
                                  algorithm: BaseAlgorithm, 
                                  train_data: np.ndarray, 
                                  test_queries: np.ndarray, 
                                  ground_truth: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        Run experiment for a specific algorithm.

        Args:
            alg_name: Name of the algorithm
            algorithm: Algorithm instance
            train_data: Training data vectors
            test_queries: Test query vectors
            ground_truth: Ground truth nearest neighbors (optional)

        Returns:
            Dictionary with experiment results
        """
        # Initialize result dictionary
        results = {
            "algorithm": alg_name,
            "parameters": algorithm.get_parameters(),
            "dataset": self.config.dataset,
            "n_train": len(train_data),
            "n_test": len(test_queries),
            "dimensions": train_data.shape[1],
            "topk": self.config.topk
        }

        # Build index and measure time
        self.logger.info(f"Building index for {alg_name}")
        start_time = time.time()
        algorithm.build_index(train_data)
        build_time = time.time() - start_time
        results["build_time_s"] = build_time
        self.logger.info(f"Index built in {build_time:.2f} seconds")

        # Get index memory usage if available
        memory_usage = algorithm.get_memory_usage()
        results["index_memory_mb"] = memory_usage
        self.logger.info(f"Index memory usage: {memory_usage:.2f} MB")

        # Run search for each query
        self.logger.info(f"Running {len(test_queries)} queries with k={self.config.topk}")
        query_times = []
        # Batch search for faster execution if supported
        start_time = time.time()
        distances, indices = algorithm.batch_search(test_queries, self.config.topk)
        total_time = time.time() - start_time

        # Store individual results for analysis
        all_results = indices

        # Calculate metrics
        qps = len(test_queries) / total_time
        mean_query_time_ms = (total_time / len(test_queries)) * 1000

        results["qps"] = qps
        results["mean_query_time_ms"] = mean_query_time_ms
        results["total_query_time_s"] = total_time

        self.logger.info(f"Search completed: {qps:.2f} QPS, {mean_query_time_ms:.2f} ms per query")

        # Calculate recall if ground truth is available
        if ground_truth is not None:
            recall = self._calculate_recall(all_results, ground_truth)
            results["recall"] = recall
            self.logger.info(f"Recall@{self.config.topk}: {recall:.4f}")

        return results

    def _calculate_recall(self, results: List[np.ndarray], ground_truth: np.ndarray) -> float:
        """
        Calculate recall@k metric.

        Args:
            results: List of result indices for each query
            ground_truth: Ground truth indices

        Returns:
            Recall@k value
        """
        recall_sum = 0.0
        k = min(self.config.topk, ground_truth.shape[1])

        for i, (result, gt) in enumerate(zip(results, ground_truth)):
            # Convert result to set for faster intersection
            result_set = set(result[:k])
            gt_set = set(gt[:k])

            # Calculate recall for this query
            intersection = len(result_set.intersection(gt_set))
            recall = intersection / len(gt_set)
            recall_sum += recall

        # Average recall across all queries
        return recall_sum / len(results)

    def _build_indices(self, train_vectors: np.ndarray):
        """
        Build indices for all registered algorithms.

        Args:
            train_vectors: Training vectors to index
        """
        for name, algorithm in self.algorithms.items():
            self.logger.info(f"Building index for {name}...")
            start_time = time.time()
            algorithm.build_index(train_vectors)
            build_time = time.time() - start_time

            # Store build time
            algorithm.build_time = build_time
            self.logger.info(f"Index for {name} built in {build_time:.2f} seconds")

            # Estimate memory usage (baseline: size of raw vectors).
            # A more accurate measure would require algorithm-specific implementation.
            memory_usage_mb = train_vectors.nbytes / (1024 * 1024)
            algorithm.index_memory_usage = memory_usage_mb
            self.logger.info(f"Estimated base memory usage for {name} index: {memory_usage_mb:.2f} MB")

            # Store build time and memory usage in results
            if name not in self.results:
                self.results[name] = {}
            self.results[name]['build_time'] = build_time
            self.results[name]['index_memory_usage_mb'] = memory_usage_mb

    def _run_searches(self, test_vectors: np.ndarray, k: int):
        """
        Run search for all test vectors against all algorithms.

        Args:
            test_vectors: Test vectors to search for
            k: Number of nearest neighbors to retrieve

        Returns:
            Dictionary mapping algorithm names to (indices, query_times)
        """
        search_results = {}

        for name, algorithm in self.algorithms.items():
            self.logger.info(f"Running searches for {name}...")
            n_queries = len(test_vectors)
            indices = np.zeros((n_queries, k), dtype=np.int32)
            query_times = np.zeros(n_queries)

            for i in range(n_queries):
                start_time = time.time()
                _, idx = algorithm.search(test_vectors[i], k=k)
                query_times[i] = time.time() - start_time
                indices[i] = idx

                # Log progress
                if (i+1) % 100 == 0 or i+1 == n_queries:
                    self.logger.info(f"  {i+1}/{n_queries} queries processed")

            # Store search results
            search_results[name] = (indices, query_times)
            qps = 1.0 / np.mean(query_times)  # Queries per second
            self.logger.info(f"Search completed: {qps:.2f} queries per second")

        return search_results

    def run(self):
        """
        Run the experiment and evaluate all algorithms.
        """
        self.logger.info("Starting experiment...")

        # Load dataset if not already loaded
        if self.dataset is None:
            self.load_dataset()

        # Get train and test vectors
        train_vectors, test_vectors = self.dataset.get_train_test_split()
        ground_truth = self.dataset.get_ground_truth()

        # Build indices for all algorithms
        self._build_indices(train_vectors)

        # Run searches
        k = self.config.topk  # Number of results to retrieve
        search_results = self._run_searches(test_vectors[:self.config.n_queries], k)

        # Evaluate results
        evaluator = Evaluator(ground_truth[:self.config.n_queries])
        for name, (indices, query_times) in search_results.items():
            self.logger.info(f"Evaluating {name}...")
            metrics = evaluator.evaluate(name, indices, query_times)
            self.results[name].update(metrics)
            self.logger.info(f"Evaluation metrics: {metrics}")

        # Print summary of results
        evaluator.print_results()

        # Generate plots
        self._generate_plots(evaluator)

        # Save results
        self._save_results()

        self.logger.info("Experiment completed successfully.")
        return self.results

    def _generate_plots(self, evaluator: Evaluator):
        """
        Generate plots for the experiment results.

        Args:
            evaluator: Evaluator with results
        """
        plots_dir = os.path.join(self.output_dir, f"plots_{self.experiment_id}")
        os.makedirs(plots_dir, exist_ok=True)

        # Recall vs QPS plot
        recall_plot_file = os.path.join(plots_dir, "recall_vs_qps.png")
        evaluator.plot_recall_vs_qps(output_file=recall_plot_file)

        # Additional plots can be added here

    def _save_results(self):
        """
        Save experiment results to disk.
        """
        # Create results directory
        results_dir = os.path.join(self.output_dir, f"experiment_{self.experiment_id}")
        os.makedirs(results_dir, exist_ok=True)

        # Save results as YAML
        results_file = os.path.join(results_dir, "results.yaml")
        with open(results_file, 'w') as f:
            yaml.dump(self.results, f)

        # Save configuration
        config_file = os.path.join(results_dir, "config.yaml")
        with open(config_file, 'w') as f:
            yaml.dump(self.config.to_dict(), f)

        self.logger.info(f"Results saved to {results_dir}")
