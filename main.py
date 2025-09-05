#!/usr/bin/env python
"""
Vector DB Retrieval Guarantee Research
Main script to run experiments comparing vector retrieval algorithms.
"""
import os
import argparse
import logging
import numpy as np
import yaml
from src.experiments import ExperimentRunner, ExperimentConfig
from src.algorithms import ExactSearch, ApproximateSearch, HNSW

def setup_logging(verbose: bool):
    """
    Set up logging configuration.

    Args:
        verbose: Whether to enable debug logging
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """
    Main entry point for running experiments.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Vector DB Retrieval Experiments")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to experiment configuration")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Directory to save results")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    args = parser.parse_args()

    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = ExperimentConfig.from_yaml(args.config)

    # Create experiment runner
    runner = ExperimentRunner(config, output_dir=args.output_dir)

    # Load dataset
    runner.load_dataset()

    # Get dimensions from dataset
    dimension = runner.dataset.train_vectors.shape[1]

    # Register algorithms based on configuration
    for name, alg_config in config.algorithms.items():
        alg_type = alg_config.pop("type")
        if alg_type == "ExactSearch":
            algorithm = ExactSearch(dimension=dimension, **alg_config)
        elif alg_type == "ApproximateSearch":
            algorithm = ApproximateSearch(dimension=dimension, **alg_config)
        elif alg_type == "HNSW":
            algorithm = HNSW(dimension=dimension, **alg_config)
        else:
            logger.warning(f"Unknown algorithm type: {alg_type}. Skipping.")
            continue

        # Restore the type to the config
        alg_config["type"] = alg_type

        # Register the algorithm
        runner.register_algorithm(algorithm)

    # Run the experiment
    logger.info("Starting experiment...")
    results = runner.run()
    logger.info("Experiment completed.")

    return 0

if __name__ == "__main__":
    main()
