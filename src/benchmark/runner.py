import os
import json
import logging
import time
import datetime
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..experiments.config import ExperimentConfig
from ..experiments.experiment_runner import ExperimentRunner
from ..algorithms import get_algorithm_instance

class BenchmarkRunner:
    """
    Orchestrates running a full benchmark across multiple datasets and algorithms.
    """

    def __init__(self, config_file: str, output_dir: str = "benchmark_results"):
        """
        Initialize the benchmark runner.

        Args:
            config_file: Path to the benchmark configuration file
            output_dir: Directory to store benchmark results
        """
        self.config_file = config_file
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, f"benchmark_{self.timestamp}")

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Set up logging
        self.log_file = os.path.join(self.output_dir, "benchmark.log")
        self.logger = self._setup_logging()

        # Load configuration
        with open(config_file, 'r') as f:
            self.config = json.load(f) if config_file.endswith('.json') else yaml.safe_load(f)

        self.logger.info(f"Loaded benchmark configuration from {config_file}")
        self.logger.info(f"Output directory: {self.output_dir}")

        # Store all results
        self.all_results = {}

    def _setup_logging(self) -> logging.Logger:
        """
        Set up logging for the benchmark.

        Returns:
            Configured logger
        """
        logger = logging.getLogger("benchmark")
        logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)

        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)  # More detailed in file
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger

    def run(self) -> Dict[str, Any]:
        """
        Run the full benchmark across all datasets and algorithms.

        Returns:
            Dictionary with all benchmark results
        """
        self.logger.info("Starting benchmark run")
        start_time = time.time()

        # Run for each dataset
        for dataset_name in self.config.get('datasets', ['random']):
            self.logger.info(f"Running benchmark for dataset: {dataset_name}")

            # Create experiment config for this dataset
            experiment_config = ExperimentConfig(
                dataset=dataset_name,
                data_dir=self.config.get('data_dir', 'data'),
                force_download=self.config.get('force_download', False),
                n_queries=self.config.get('n_queries', 1000),
                topk=self.config.get('topk', 100),
                repeat=self.config.get('repeat', 1),
                algorithms=self.config.get('algorithms', {}),
                seed=self.config.get('seed', 42),
                output_prefix=f"{dataset_name}_{self.timestamp}"
            )

            # Run experiments for this dataset
            dataset_output_dir = os.path.join(self.output_dir, dataset_name)
            os.makedirs(dataset_output_dir, exist_ok=True)

            # Save the experiment config
            config_path = os.path.join(dataset_output_dir, f"{dataset_name}_config.yaml")
            experiment_config.save(config_path)

            # Create experiment runner
            runner = ExperimentRunner(experiment_config, output_dir=dataset_output_dir)

            try:
                # Load dataset
                self.logger.info(f"Loading dataset: {dataset_name}")
                runner.load_dataset()

                # Get vector dimension from dataset
                dimension = runner.dataset.train_vectors.shape[1]

                # Register algorithms
                for alg_name, alg_config in experiment_config.algorithms.items():
                    alg_type = alg_config.pop("type")
                    algorithm = get_algorithm_instance(alg_type, dimension, name=alg_name, **alg_config)
                    runner.register_algorithm(alg_name, algorithm)
                    alg_config["type"] = alg_type  # Restore the type for future reference

                # Run the experiment
                self.logger.info(f"Running experiments for dataset: {dataset_name}")
                results = runner.run()

                # Store results
                self.all_results[dataset_name] = results

                # Save results for this dataset
                results_path = os.path.join(dataset_output_dir, f"{dataset_name}_results.json")
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2)

                self.logger.info(f"Completed experiments for dataset: {dataset_name}")

            except Exception as e:
                self.logger.error(f"Error running experiments for dataset {dataset_name}: {str(e)}", exc_info=True)

        # Save all results
        all_results_path = os.path.join(self.output_dir, "all_results.json")
        with open(all_results_path, 'w') as f:
            json.dump(self.all_results, f, indent=2)

        # Generate summary report
        self._generate_summary_report()

        end_time = time.time()
        self.logger.info(f"Benchmark completed in {end_time - start_time:.2f} seconds")

        return self.all_results

    def _generate_summary_report(self) -> None:
        """
        Generate a markdown summary report of benchmark results.
        """
        self.logger.info("Generating summary report")

        report_path = os.path.join(self.output_dir, "benchmark_summary.md")

        with open(report_path, 'w') as f:
            f.write(f"# Vector Retrieval Benchmark Summary\n\n")
            f.write(f"*Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

            for dataset_name, results in self.all_results.items():
                f.write(f"## Dataset: {dataset_name}\n\n")

                # Algorithm performance table
                f.write(f"### Algorithm Performance\n\n")
                f.write(f"| Algorithm | Recall@{self.config.get('topk', 100)} | QPS | Mean Query Time (ms) | Build Time (s) | Index Memory (MB) |\n")
                f.write(f"|-----------|-----------|-----|----------------------|----------------|-------------------|\n")

                for alg_name, alg_results in results.items():
                    recall = alg_results.get('recall', 0)
                    qps = alg_results.get('qps', 0)
                    query_time = alg_results.get('mean_query_time_ms', 0)
                    build_time = alg_results.get('build_time_s', 0)
                    memory = alg_results.get('index_memory_mb', 0)

                    f.write(f"| {alg_name} | {recall:.4f} | {qps:.2f}| {query_time:.2f} | {build_time:.2f} | {memory:.2f} |\n")

                f.write(f"\n\n")

        self.logger.info(f"Summary report written to: {report_path}")
