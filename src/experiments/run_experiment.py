#!/usr/bin/env python
import argparse
import logging
import os
import numpy as np
import sys
import yaml

from .config import ExperimentConfig
from .experiment_runner import ExperimentRunner
from ..algorithms import BaseAlgorithm, ExactSearch, ApproximateSearch, HNSW

def get_algorithm(algorithm_type: str, dimension: int, **params) -> BaseAlgorithm:
    """
    Create an algorithm instance based on type and parameters.

    Args:
        algorithm_type: Type of algorithm to create
        dimension: Dimensionality of vectors
        **params: Algorithm-specific parameters

    Returns:
        Algorithm instance
    """
    if algorithm_type == "ExactSearch":
        return ExactSearch(dimension=dimension, **params)
    elif algorithm_type == "ApproximateSearch":
        return ApproximateSearch(dimension=dimension, **params)
    elif algorithm_type == "HNSW":
        return HNSW(dimension=dimension, **params)
    else:
        raise ValueError(f"Unknown algorithm type: {algorithm_type}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run vector retrieval experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config file")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("run_experiment")

    # Load configuration
    try:
        config = ExperimentConfig.from_yaml(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        sys.exit(1)

    # Create experiment runner
    runner = ExperimentRunner(config, output_dir=args.output_dir)

    # Load dataset
    runner.load_dataset()

    # Get vector dimension from dataset
    dimension = runner.dataset.train_vectors.shape[1]

    # Register algorithms
    for alg_name, alg_config in config.algorithms.items():
        alg_type = alg_config.pop("type")
        algorithm = get_algorithm(alg_type, dimension, **alg_config)
        runner.register_algorithm(algorithm)
        alg_config["type"] = alg_type  # Restore the type for future reference

    # Run the experiment
    try:
        results = runner.run()
        logger.info("Experiment completed successfully")
    except Exception as e:
        logger.error(f"Error during experiment: {str(e)}", exc_info=True)
        sys.exit(1)

    return 0

if __name__ == "__main__":
    sys.exit(main())
