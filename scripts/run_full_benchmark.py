#!/usr/bin/env python
"""
Full Benchmark Runner for Vector Retrieval Algorithms

This script provides a comprehensive experimental loop for evaluating
vector retrieval algorithms with retrieval guarantees across multiple
datasets and configurations.
"""

import os
import sys
import argparse
import logging
import yaml
import json
import time
import copy
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Ensure we are using appropriate numeric libraries for the current platform
from src.utils.compat import ensure_arm_compatible_blas

ensure_arm_compatible_blas()

# Import the BenchmarkRunner
from src.benchmark.runner import BenchmarkRunner


def _format_ops(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A" if value is None else str(value)

    if math.isnan(numeric):
        return "N/A"

    if abs(numeric) >= 1e9:
        return f"{numeric / 1e9:.2f}B"
    if abs(numeric) >= 1e6:
        return f"{numeric / 1e6:.2f}M"
    if abs(numeric) >= 1e3:
        return f"{numeric / 1e3:.1f}K"
    return f"{numeric:.0f}"

class FullBenchmarkRunner:
    """
    Comprehensive benchmark runner for vector retrieval algorithms.
    """
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """
        Initialize the benchmark runner.
        
        Args:
            output_dir: Directory to save all benchmark results
        """
        self.output_dir = output_dir
        self.benchmark_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(output_dir, f"benchmark_{self.benchmark_id}")
        
        # Create output directories
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Set up logging
        log_file = os.path.join(self.results_dir, "benchmark.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("FullBenchmark")
        
        self.all_results = {}
        
    def load_benchmark_config(self, config_file: str) -> Dict[str, Any]:
        """
        Load benchmark configuration from YAML file.
        
        Args:
            config_file: Path to benchmark configuration file
            
        Returns:
            Dictionary containing benchmark configuration
        """
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        self.logger.info(f"Loaded benchmark configuration from {config_file}")
        return config
    
    def run_single_experiment(self, dataset_name: str, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single experiment with given dataset and configuration.
        
        Args:
            dataset_name: Name of the dataset to use
            config_dict: Configuration dictionary for the experiment
            
        Returns:
            Dictionary containing experiment results
        """
        self.logger.info(f"Running experiment on dataset: {dataset_name}")
        
        # Create experiment configuration
        config_dict['dataset'] = dataset_name
        config = ExperimentConfig(**config_dict)
        
        # Create experiment-specific output directory
        exp_output_dir = os.path.join(self.results_dir, f"{dataset_name}")
        os.makedirs(exp_output_dir, exist_ok=True)
        
        # Create and run experiment
        runner = ExperimentRunner(config, output_dir=exp_output_dir)
        
        try:
            # Load dataset
            runner.load_dataset()
            
            # Get vector dimension
            dimension = runner.dataset.train_vectors.shape[1]
            
            # Register algorithms
            for alg_name, alg_config_orig in config.algorithms.items():
                alg_config = alg_config_orig.copy()
                alg_type = alg_config.pop("type")
                
                # Dynamically find algorithm class if needed, or use if/else
                if alg_type == "ExactSearch":
                    algorithm = ExactSearch(name=alg_name, dimension=dimension, **alg_config)
                elif alg_type == "ApproximateSearch":
                    algorithm = ApproximateSearch(name=alg_name, dimension=dimension, **alg_config)
                elif alg_type == "HNSW":
                    algorithm = HNSW(name=alg_name, dimension=dimension, **alg_config)
                else:
                    self.logger.warning(f"Unknown algorithm type: {alg_type}. Skipping.")
                    continue
                
                # Register algorithm
                runner.register_algorithm(algorithm)
            
            # Run experiment
            results = runner.run()
            
            self.logger.info(f"Experiment completed for {dataset_name}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error running experiment on {dataset_name}: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def run_benchmark_suite(self, config_file: str):
        """
        Run the complete benchmark suite.
        
        Args:
            config_file: Path to benchmark configuration file
        """
        self.logger.info(f"Starting full benchmark suite: {self.benchmark_id}")
        
        # Load configuration
        benchmark_config = self.load_benchmark_config(config_file)
        
        # Extract datasets and base configuration
        datasets = benchmark_config.get('datasets', ['random'])
        base_config = {k: v for k, v in benchmark_config.items() if k != 'datasets'}
        
        # Run experiments for each dataset
        for dataset_name in datasets:
            self.logger.info(f"=" * 60)
            self.logger.info(f"Running benchmark on dataset: {dataset_name}")
            self.logger.info(f"=" * 60)
            
            start_time = time.time()
            # Use deepcopy to prevent mutation of config between runs
            results = self.run_single_experiment(dataset_name, copy.deepcopy(base_config))
            end_time = time.time()
            
            # Store results with timing information
            self.all_results[dataset_name] = {
                'results': results,
                'experiment_time': end_time - start_time,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Dataset {dataset_name} completed in {end_time - start_time:.2f} seconds")
        
        # Generate summary report
        self.generate_summary_report()
        
        self.logger.info(f"Full benchmark suite completed: {self.benchmark_id}")
    
    def generate_summary_report(self):
        """
        Generate a comprehensive summary report of all benchmark results.
        """
        self.logger.info("Generating summary report...")
        
        # Save raw results
        results_file = os.path.join(self.results_dir, "all_results.json")
        with open(results_file, 'w') as f:
            json.dump(self.all_results, f, indent=2, default=str)
        
        # Generate summary statistics
        summary_file = os.path.join(self.results_dir, "benchmark_summary.md")
        with open(summary_file, 'w') as f:
            f.write(f"# Benchmark Summary Report\n\n")
            f.write(f"**Benchmark ID:** {self.benchmark_id}\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
            
            f.write("## Datasets Evaluated\n\n")
            for dataset_name, data in self.all_results.items():
                f.write(f"### {dataset_name}\n")
                f.write(f"- **Experiment Time:** {data['experiment_time']:.2f} seconds\n")
                f.write(f"- **Timestamp:** {data['timestamp']}\n")
                
                if 'error' in data['results']:
                    f.write(f"- **Status:** ERROR - {data['results']['error']}\n")
                else:
                    f.write(f"- **Status:** SUCCESS\n")
                    f.write(f"- **Algorithms Evaluated:** {len(data['results'])}\n")
                    
                    # Add algorithm performance summary
                    f.write("\n#### Algorithm Performance\n\n")
                    f.write("| Algorithm | Recall@10 | QPS | Mean Query Time (ms) | Build Time (s) | Index Memory (MB) | Vector Ops | Vector Ops / Query |\n")
                    f.write("|-----------|-----------|-----|----------------------|----------------|-------------------|------------|---------------------|")
                    
                    for alg_name, metrics in data['results'].items():
                        if isinstance(metrics, dict):
                            recall = metrics.get('recall@10', 'N/A')
                            qps = metrics.get('qps', 'N/A')
                            query_time = metrics.get('mean_query_time', 'N/A')
                            build_time = metrics.get('build_time', 'N/A')
                            memory_usage = metrics.get('index_memory_usage_mb', 'N/A')
                            vector_ops = metrics.get('vector_similarity_ops', 'N/A')
                            vector_ops_per_query = metrics.get('vector_similarity_ops_per_query', 'N/A')

                            # Formatting
                            recall_str = f"{recall:.4f}" if isinstance(recall, float) else str(recall)
                            qps_str = f"{qps:.2f}" if isinstance(qps, float) else str(qps)
                            query_time_str = f"{query_time:.2f}" if isinstance(query_time, float) else str(query_time)
                            build_time_str = f"{build_time:.2f}" if isinstance(build_time, float) else str(build_time)
                            memory_usage_str = f"{memory_usage:.2f}" if isinstance(memory_usage, float) else str(memory_usage)
                            vector_ops_str = _format_ops(vector_ops)
                            vector_ops_per_query_str = _format_ops(vector_ops_per_query)

                            f.write(
                                f"| {alg_name} | {recall_str} | {qps_str} | {query_time_str} | {build_time_str} | "
                                f"{memory_usage_str} | {vector_ops_str} | {vector_ops_per_query_str} |\n"
                            )
                
                f.write("\n")
        
        self.logger.info(f"Summary report saved to {summary_file}")

    @staticmethod
    def create_default_benchmark_config():
        """
        Create a default benchmark configuration file.
        """
        config = {
            'datasets': ['random', 'sift1m', 'glove50'],
            'n_queries': 1000,
            'topk': 100,
            'repeat': 1,
            'algorithms': {
                'exact': {
                    'type': 'ExactSearch',
                    'metric': 'l2'
                },
                'ivf_flat': {
                    'type': 'ApproximateSearch',
                    'index_type': 'IVF100,Flat',
                    'metric': 'l2',
                    'nprobe': 10
                },
                'hnsw': {
                    'type': 'HNSW',
                    'M': 16,
                    'efConstruction': 200,
                    'efSearch': 100,
                    'metric': 'l2'
                }
            },
            'seed': 42,
            'output_prefix': 'benchmark'
        }

        config_file = 'configs/benchmark_config.yaml'
        os.makedirs('configs', exist_ok=True)

        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"Default benchmark configuration created: {config_file}")
        return config_file

def main():
    """
    Main entry point for the benchmark runner.
    """
    parser = argparse.ArgumentParser(description="Run vector retrieval benchmark suite")
    parser.add_argument("--config", type=str, help="Path to benchmark configuration file")
    parser.add_argument("--create-config", action="store_true", help="Create default benchmark configuration")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Directory to save results")
    args = parser.parse_args()

    # Create default configuration if requested
    if args.create_config:
        config_file = FullBenchmarkRunner.create_default_benchmark_config()
        print(f"You can now edit {config_file} and run the benchmark with --config {config_file}")
        return 0

    # Ensure config file is provided
    if not args.config:
        print("Error: --config argument is required unless --create-config is specified")
        parser.print_help()
        return 1

    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return 1

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run the benchmark
    try:
        benchmark = BenchmarkRunner(args.config, args.output_dir)
        benchmark.run()
        print(f"Benchmark completed successfully. Results saved to {benchmark.output_dir}")
        return 0
    except Exception as e:
        print(f"Error running benchmark: {str(e)}")
        logging.error("Benchmark failed", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
