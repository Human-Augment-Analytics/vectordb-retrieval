#!/usr/bin/env python
import argparse
import logging
import os
import sys
import yaml
import copy

from .config import ExperimentConfig
from .experiment_runner import ExperimentRunner
from ..algorithms import get_algorithm_instance

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
        alg_config_copy = copy.deepcopy(alg_config)
        alg_type = alg_config_copy.pop("type")
        algorithm = get_algorithm_instance(alg_type, dimension, name=alg_name, **alg_config_copy)
        runner.register_algorithm(algorithm, name=alg_name)

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
