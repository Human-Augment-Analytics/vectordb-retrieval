import os
import json
import logging
import time
import datetime
import copy
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

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

        # Load configuration early so we can honor repository-level paths.
        with open(config_file, 'r') as f:
            self.config = json.load(f) if config_file.endswith('.json') else yaml.safe_load(f)

        self.global_indexers = copy.deepcopy(self.config.get('indexers', {}))
        self.global_searchers = copy.deepcopy(self.config.get('searchers', {}))

        base_output_dir = self.config.get('output_dir', output_dir)
        self.output_dir = os.path.join(base_output_dir, f"benchmark_{self.timestamp}")

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Set up logging
        self.log_file = os.path.join(self.output_dir, "benchmark.log")
        self.logger = self._setup_logging()

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
        datasets_config = self.config.get('datasets', ['random'])

        for dataset_entry in datasets_config:
            dataset_name, dataset_options = self._normalize_dataset_entry(dataset_entry)
            self.logger.info(f"Running benchmark for dataset: {dataset_name}")

            dataset_metric = dataset_options.get('metric')
            dataset_algorithms_override = dataset_options.get('algorithms', {})
            dataset_specific_options = copy.deepcopy(
                dataset_options.get('dataset_options') or dataset_options.get('options') or {}
            )
            dataset_data_dir = dataset_options.get('data_dir', self.config.get('data_dir', 'data'))

            # Apply dataset-specific overrides while keeping base algorithm definitions intact.
            base_algorithms = copy.deepcopy(self.config.get('algorithms', {}))
            algorithms_for_dataset: Dict[str, Dict[str, Any]] = {}

            for alg_name, alg_config in base_algorithms.items():
                merged_config = copy.deepcopy(alg_config)
                override_config = dataset_algorithms_override.get(alg_name, {})
                merged_config.update(copy.deepcopy(override_config))

                if dataset_metric is not None:
                    merged_config['metric'] = dataset_metric

                self._resolve_modular_components(merged_config)

                algorithms_for_dataset[alg_name] = merged_config

            # Include overrides for algorithms defined only at the dataset level.
            for alg_name, override_config in dataset_algorithms_override.items():
                if alg_name not in algorithms_for_dataset:
                    merged_override = copy.deepcopy(override_config)
                    if dataset_metric is not None and 'metric' not in merged_override:
                        merged_override['metric'] = dataset_metric
                    self._resolve_modular_components(merged_override)
                    algorithms_for_dataset[alg_name] = merged_override

            experiment_kwargs = dict(
                dataset=dataset_name,
                data_dir=dataset_data_dir,
                force_download=self.config.get('force_download', False),
                n_queries=dataset_options.get('n_queries', self.config.get('n_queries', 1000)),
                topk=dataset_options.get('topk', self.config.get('topk', 100)),
                repeat=dataset_options.get('repeat', self.config.get('repeat', 1)),
                algorithms=algorithms_for_dataset,
                seed=dataset_options.get('seed', self.config.get('seed', 42)),
                output_prefix=dataset_options.get('output_prefix', f"{dataset_name}_{self.timestamp}")
            )

            query_batch_size = dataset_options.get('query_batch_size')
            if query_batch_size is None:
                query_batch_size = self.config.get('query_batch_size')
            if query_batch_size is not None:
                experiment_kwargs['query_batch_size'] = query_batch_size

            if dataset_metric is not None:
                experiment_kwargs['metric'] = dataset_metric
            if dataset_specific_options:
                experiment_kwargs['dataset_options'] = dataset_specific_options
            experiment_config = ExperimentConfig(**experiment_kwargs)

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
                    alg_config_copy = copy.deepcopy(alg_config)
                    alg_type = alg_config_copy.pop("type")
                    algorithm = get_algorithm_instance(alg_type, dimension, name=alg_name, **alg_config_copy)
                    runner.register_algorithm(algorithm, name=alg_name)

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

    @staticmethod
    def _deep_merge_dict(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge dictionary values without mutating inputs."""
        result = copy.deepcopy(base)
        for key, value in updates.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = BenchmarkRunner._deep_merge_dict(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    def _materialize_component(
        self,
        ref_name: Optional[str],
        inline_cfg: Optional[Any],
        registry: Dict[str, Dict[str, Any]],
        component_label: str,
    ) -> Optional[Dict[str, Any]]:
        """Resolve component configuration from references/overrides."""

        config: Optional[Dict[str, Any]] = None

        if ref_name:
            if ref_name not in registry:
                raise ValueError(
                    f"Unknown {component_label} reference '{ref_name}'. Available: {list(registry.keys())}"
                )
            config = copy.deepcopy(registry[ref_name])

        if inline_cfg is None:
            return config

        if isinstance(inline_cfg, str):
            # Treat direct string as a reference
            if inline_cfg not in registry:
                raise ValueError(
                    f"Unknown {component_label} reference '{inline_cfg}'. Available: {list(registry.keys())}"
                )
            inline_dict = copy.deepcopy(registry[inline_cfg])
        elif isinstance(inline_cfg, dict):
            inline_dict = copy.deepcopy(inline_cfg)
        else:
            raise TypeError(
                f"{component_label.capitalize()} configuration must be a dict or string reference, got {type(inline_cfg)}"
            )

        if config is None:
            config = inline_dict
        else:
            config = self._deep_merge_dict(config, inline_dict)

        return config

    def _resolve_modular_components(self, algorithm_config: Dict[str, Any]) -> None:
        """Inject resolved indexer/searcher configs for modular algorithms if present."""

        indexer_ref = algorithm_config.pop('indexer_ref', None)
        searcher_ref = algorithm_config.pop('searcher_ref', None)

        indexer_cfg = self._materialize_component(
            indexer_ref,
            algorithm_config.get('indexer'),
            self.global_indexers,
            'indexer'
        )
        searcher_cfg = self._materialize_component(
            searcher_ref,
            algorithm_config.get('searcher'),
            self.global_searchers,
            'searcher'
        )

        if indexer_cfg is not None:
            algorithm_config['indexer'] = indexer_cfg
        if searcher_cfg is not None:
            algorithm_config['searcher'] = searcher_cfg

        if indexer_cfg is not None or searcher_cfg is not None:
            algorithm_config.setdefault('type', 'Composite')

    def _normalize_dataset_entry(self, entry: Any) -> Tuple[str, Dict[str, Any]]:
        """Convert dataset configuration entries into a uniform structure."""
        if isinstance(entry, str):
            return entry, {}
        if isinstance(entry, dict):
            if 'name' not in entry:
                raise ValueError("Dataset configuration entries must include a 'name' key")
            name = entry['name']
            options = {k: v for k, v in entry.items() if k != 'name'}
            return name, options
        raise ValueError("Dataset configuration entries must be strings or dictionaries")

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
                f.write("| Algorithm | Recall | QPS | Mean Query Time (ms) | Build Time (s) | Index Memory (MB) |\n")
                f.write("|-----------|--------|-----|----------------------|----------------|-------------------|\n")

                for alg_name, alg_results in results.items():
                    recall_display = "0.0000"
                    recall_value = alg_results.get('recall')
                    recall_key = None

                    if recall_value is not None:
                        recall_key = 'summary'
                    else:
                        recall_metrics = [
                            key for key in alg_results.keys()
                            if key.startswith('recall@') and alg_results.get(key) is not None
                        ]
                        if recall_metrics:
                            recall_metrics.sort(key=lambda k: int(k.split('@')[-1]))
                            recall_key = recall_metrics[-1]
                            recall_value = alg_results[recall_key]

                    if recall_value is not None:
                        if recall_key and recall_key.startswith('recall@'):
                            cutoff = recall_key.split('@')[-1]
                            recall_display = f"{recall_value:.4f} (@{cutoff})"
                        else:
                            recall_display = f"{recall_value:.4f}"

                    qps = alg_results.get('qps', 0)
                    query_time = alg_results.get('mean_query_time_ms', 0)
                    build_time = alg_results.get('build_time_s', 0)
                    memory = alg_results.get('index_memory_mb', 0)

                    f.write(f"| {alg_name} | {recall_display} | {qps:.2f}| {query_time:.2f} | {build_time:.2f} | {memory:.2f} |\n")

                f.write(f"\n\n")

        self.logger.info(f"Summary report written to: {report_path}")
