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
            self.logger.info(f"Index built in {build_time:.2f} seconds")

            # Store build time in results
            if name not in self.results:
                self.results[name] = {}
            self.results[name]['build_time'] = build_time

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
