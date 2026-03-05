import os
import json
import html
import logging
import math
import time
import datetime
import copy
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

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
        try:
            self._generate_one_page_summary()
        except Exception as exc:  # pragma: no cover - defensive; benchmark should not fail on reporting
            self.logger.error(f"Failed to generate one-page summary: {exc}", exc_info=True)

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

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """Convert scalar-like values to finite floats, returning None otherwise."""
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric):
            return None
        return numeric

    @staticmethod
    def _sanitize_slug(name: str) -> str:
        slug = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name.strip().lower())
        return slug or "dataset"

    @staticmethod
    def _md_escape(value: str) -> str:
        return value.replace("|", "\\|")

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        if isinstance(value, (str, int, bool)) or value is None:
            return str(value)
        try:
            rendered = json.dumps(value, sort_keys=True)
        except TypeError:
            rendered = str(value)
        if len(rendered) > 120:
            rendered = f"{rendered[:117]}..."
        return rendered

    def _extract_recall_metric(self, alg_results: Dict[str, Any]) -> Tuple[Optional[float], str]:
        """Resolve the recall value and source key for a result row."""
        summary_recall = self._safe_float(alg_results.get("recall"))
        if summary_recall is not None:
            return summary_recall, "recall"

        candidates: List[Tuple[int, float, str]] = []
        for key, raw_value in alg_results.items():
            if not key.startswith("recall@"):
                continue
            try:
                cutoff = int(key.split("@", 1)[1])
            except (TypeError, ValueError):
                continue
            value = self._safe_float(raw_value)
            if value is None:
                continue
            candidates.append((cutoff, value, key))

        if not candidates:
            return None, "recall"

        cutoff, recall_value, key = max(candidates, key=lambda item: item[0])
        return recall_value, key if cutoff > 0 else "recall"

    def _load_dataset_config(self, dataset_name: str) -> Tuple[Dict[str, Any], Optional[Path]]:
        cfg_path = Path(self.output_dir) / dataset_name / f"{dataset_name}_config.yaml"
        if not cfg_path.exists():
            return {}, None
        with open(cfg_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            return {}, cfg_path
        return loaded, cfg_path

    def _describe_component(self, component_cfg: Any) -> str:
        if not isinstance(component_cfg, dict):
            return "N/A"
        comp_type = str(component_cfg.get("type", "Unknown"))
        params: List[str] = []
        for key in sorted(component_cfg.keys()):
            if key == "type":
                continue
            value = component_cfg[key]
            if isinstance(value, (str, int, float, bool)) or value is None:
                params.append(f"{key}={self._format_value(value)}")
        if not params:
            return comp_type
        rendered = ", ".join(params[:4])
        if len(params) > 4:
            rendered += ", ..."
        return f"{comp_type} ({rendered})"

    def _build_qps_recall_svg(
        self,
        dataset_name: str,
        points: List[Dict[str, Any]],
        output_path: Path,
        recall_label: str,
    ) -> None:
        width, height = 920, 460
        margin = {"left": 78, "right": 260, "top": 42, "bottom": 60}
        plot_w = width - margin["left"] - margin["right"]
        plot_h = height - margin["top"] - margin["bottom"]

        x_vals = [point["qps"] for point in points]
        y_vals = [point["recall"] for point in points]

        x_min = max(min(x_vals) * 0.8, 1e-3)
        x_max = max(x_vals) * 1.2
        if x_max <= x_min:
            x_max = x_min * 10.0

        y_min = max(0.0, min(y_vals) - 0.05)
        y_max = min(1.05, max(y_vals) + 0.05)
        if y_max <= y_min:
            y_max = min(1.05, y_min + 0.1)

        lmin = math.log10(x_min)
        lmax = math.log10(x_max)
        if lmax <= lmin:
            lmax = lmin + 1.0

        def x_to_px(x_value: float) -> float:
            return margin["left"] + ((math.log10(x_value) - lmin) / (lmax - lmin)) * plot_w

        def y_to_px(y_value: float) -> float:
            return margin["top"] + (1.0 - ((y_value - y_min) / (y_max - y_min))) * plot_h

        sorted_points = sorted(points, key=lambda item: (-item["recall"], -item["qps"], item["algo"]))
        for idx, point in enumerate(sorted_points, start=1):
            point["idx"] = idx
            point["x"] = x_to_px(point["qps"])
            point["y"] = y_to_px(point["recall"])

        lines: List[str] = []
        lines.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        )
        lines.append(
            "<style>"
            "text{font-family:Arial,sans-serif;fill:#111}"
            ".axis{stroke:#111;stroke-width:1.1}"
            ".grid{stroke:#999;stroke-dasharray:3 3;opacity:.35}"
            ".label{font-size:11px}"
            ".title{font-size:14px;font-weight:600}"
            ".index{font-size:10px;font-weight:600}"
            "</style>"
        )
        lines.append('<rect x="0" y="0" width="100%" height="100%" fill="#fff"/>')

        x0 = margin["left"]
        y0 = margin["top"] + plot_h
        x1 = margin["left"] + plot_w
        y1 = margin["top"]

        lines.append(
            f'<text class="title" x="{x0}" y="24">QPS vs Recall — {html.escape(dataset_name)}</text>'
        )
        lines.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}"/>')
        lines.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}"/>')

        x_tick_start = int(math.floor(math.log10(x_min)))
        x_tick_end = int(math.ceil(math.log10(x_max)))
        for exp in range(x_tick_start, x_tick_end + 1):
            tick_value = 10 ** exp
            if tick_value < x_min or tick_value > x_max:
                continue
            tick_x = x_to_px(float(tick_value))
            lines.append(f'<line class="grid" x1="{tick_x:.2f}" y1="{y0}" x2="{tick_x:.2f}" y2="{y1}"/>')
            lines.append(
                f'<text class="label" x="{tick_x:.2f}" y="{y0 + 18}" text-anchor="middle">{tick_value:,}</text>'
            )

        y_ticks = 5
        for tick_idx in range(y_ticks):
            tick_ratio = tick_idx / (y_ticks - 1)
            tick_value = y_min + ((y_max - y_min) * tick_ratio)
            tick_y = y_to_px(tick_value)
            lines.append(f'<line class="grid" x1="{x0}" y1="{tick_y:.2f}" x2="{x1}" y2="{tick_y:.2f}"/>')
            lines.append(
                f'<text class="label" x="{x0 - 10}" y="{tick_y + 4:.2f}" text-anchor="end">{tick_value:.2f}</text>'
            )

        lines.append(
            f'<text class="label" x="{(x0 + x1) / 2:.2f}" y="{height - 18}" text-anchor="middle">QPS (log scale)</text>'
        )
        lines.append(
            f'<text class="label" x="18" y="{(y0 + y1) / 2:.2f}" '
            f'transform="rotate(-90 18 {(y0 + y1) / 2:.2f})" text-anchor="middle">{html.escape(recall_label)}</text>'
        )

        for point in sorted_points:
            lines.append(
                f'<circle cx="{point["x"]:.2f}" cy="{point["y"]:.2f}" r="4.4" fill="#d62728" stroke="#111" stroke-width="0.6"/>'
            )
            lines.append(
                f'<text class="index" x="{point["x"] + 6:.2f}" y="{point["y"] - 4:.2f}">{point["idx"]}</text>'
            )

        list_x = x1 + 16
        list_y = y1 + 8
        lines.append(f'<text class="label" x="{list_x}" y="{list_y}">Labels</text>')
        list_y += 14
        for point in sorted_points:
            lines.append(
                f'<text class="label" x="{list_x}" y="{list_y}">{point["idx"]}. {html.escape(point["algo"])}</text>'
            )
            list_y += 14

        lines.append("</svg>")
        output_path.write_text("\n".join(lines), encoding="utf-8")

    def _generate_one_page_summary(self) -> None:
        """Generate a compact one-page summary with QPS-vs-recall plots."""
        summary_path = Path(self.output_dir) / "one-page-summary.md"
        compatibility_path = Path(self.output_dir) / "qps_recall_summary.md"

        lines: List[str] = []
        lines.append("# One-Page Benchmark Summary (QPS vs Recall)")
        lines.append("")
        lines.append(f"*Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        lines.append(f"*Run directory: `{self.output_dir}`*")
        lines.append("")

        takeaway_rows: List[str] = []

        for dataset_name, dataset_results in self.all_results.items():
            lines.append(f"## Dataset: {dataset_name}")
            lines.append("")

            config_data, config_path = self._load_dataset_config(dataset_name)
            plot_points: List[Dict[str, Any]] = []
            score_rows: List[Dict[str, Any]] = []
            recall_label = "recall"

            for algorithm_name, metrics in dataset_results.items():
                if not isinstance(metrics, dict):
                    continue
                recall_value, recall_key = self._extract_recall_metric(metrics)
                if recall_key != "recall":
                    recall_label = recall_key
                qps = self._safe_float(metrics.get("qps"))
                query_time_ms = self._safe_float(metrics.get("mean_query_time_ms"))
                if query_time_ms is None:
                    query_time_ms = self._safe_float(metrics.get("mean_query_time"))
                build_time_s = self._safe_float(metrics.get("build_time_s"))
                status = str(metrics.get("status", "ok")).lower()

                score_rows.append(
                    {
                        "algorithm": algorithm_name,
                        "recall": recall_value,
                        "qps": qps,
                        "query_time_ms": query_time_ms,
                        "build_time_s": build_time_s,
                        "status": status,
                    }
                )

                if status != "build_only" and recall_value is not None and qps is not None and qps > 0:
                    plot_points.append(
                        {
                            "algo": algorithm_name,
                            "qps": qps,
                            "recall": recall_value,
                        }
                    )

            if plot_points:
                svg_name = f"qps_recall_{self._sanitize_slug(dataset_name)}.svg"
                svg_path = Path(self.output_dir) / svg_name
                self._build_qps_recall_svg(dataset_name, plot_points, svg_path, recall_label)
                lines.append(f"![QPS vs Recall — {dataset_name}](./{svg_name})")
                lines.append("")

                best_recall = max(plot_points, key=lambda item: (item["recall"], item["qps"]))
                best_qps = max(plot_points, key=lambda item: (item["qps"], item["recall"]))
                takeaway_rows.append(
                    f"- `{dataset_name}`: best recall `{best_recall['algo']}` ({best_recall['recall']:.4f}), "
                    f"best QPS `{best_qps['algo']}` ({best_qps['qps']:.2f})"
                )
            else:
                lines.append("_No QPS/recall plot data available for this dataset._")
                lines.append("")

            lines.append("| Algorithm | Recall | QPS | Mean Query Time (ms) | Build Time (s) | Status |")
            lines.append("|---|---:|---:|---:|---:|---|")
            for row in sorted(
                score_rows,
                key=lambda item: (
                    -(item["recall"] if item["recall"] is not None else -1.0),
                    -(item["qps"] if item["qps"] is not None else -1.0),
                    item["algorithm"],
                ),
            ):
                recall_display = f"{row['recall']:.4f}" if row["recall"] is not None else "N/A"
                qps_display = f"{row['qps']:.2f}" if row["qps"] is not None else "N/A"
                query_display = f"{row['query_time_ms']:.3f}" if row["query_time_ms"] is not None else "N/A"
                build_display = f"{row['build_time_s']:.2f}" if row["build_time_s"] is not None else "N/A"
                lines.append(
                    f"| {self._md_escape(row['algorithm'])} | {recall_display} | {qps_display} | "
                    f"{query_display} | {build_display} | {self._md_escape(row['status'])} |"
                )
            lines.append("")

            algorithms_cfg = config_data.get("algorithms", {}) if isinstance(config_data, dict) else {}
            if isinstance(algorithms_cfg, dict) and algorithms_cfg:
                lines.append("### Algorithm Implementation Details")
                lines.append("")
                lines.append("| Algorithm | Type | Metric | Indexer | Searcher |")
                lines.append("|---|---|---|---|---|")
                for algorithm_name in sorted(algorithms_cfg.keys()):
                    algorithm_cfg = algorithms_cfg[algorithm_name] if isinstance(algorithms_cfg[algorithm_name], dict) else {}
                    alg_type = str(algorithm_cfg.get("type", "Unknown")) if isinstance(algorithm_cfg, dict) else "Unknown"
                    alg_metric = str(algorithm_cfg.get("metric", "N/A")) if isinstance(algorithm_cfg, dict) else "N/A"
                    indexer_desc = self._describe_component(algorithm_cfg.get("indexer")) if isinstance(algorithm_cfg, dict) else "N/A"
                    searcher_desc = self._describe_component(algorithm_cfg.get("searcher")) if isinstance(algorithm_cfg, dict) else "N/A"
                    lines.append(
                        f"| {self._md_escape(algorithm_name)} | {self._md_escape(alg_type)} | "
                        f"{self._md_escape(alg_metric)} | {self._md_escape(indexer_desc)} | "
                        f"{self._md_escape(searcher_desc)} |"
                    )
                lines.append("")

            lines.append("### Dataset Details")
            lines.append("")
            if config_path is not None:
                lines.append(f"- Config: `{config_path}`")
            if isinstance(config_data, dict):
                for key in ("metric", "topk", "n_queries", "repeat", "seed"):
                    if key in config_data:
                        lines.append(f"- {key}: `{self._format_value(config_data[key])}`")
                dataset_options = config_data.get("dataset_options")
                if isinstance(dataset_options, dict) and dataset_options:
                    for option_key in sorted(dataset_options.keys()):
                        lines.append(
                            f"- dataset_options.{option_key}: "
                            f"`{self._format_value(dataset_options[option_key])}`"
                        )
            lines.append("")

        if takeaway_rows:
            lines.append("## Brief Takeaways")
            lines.append("")
            lines.extend(takeaway_rows)
            lines.append("")

        content = "\n".join(lines).rstrip() + "\n"
        summary_path.write_text(content, encoding="utf-8")
        compatibility_path.write_text(content, encoding="utf-8")
        self.logger.info(f"One-page summary written to: {summary_path}")
        self.logger.info(f"Compatibility summary written to: {compatibility_path}")

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
                    status = str(alg_results.get("status", "")).lower()
                    if status == "build_only":
                        build_time = float(alg_results.get("build_time_s", 0.0) or 0.0)
                        memory = float(alg_results.get("index_memory_mb", 0.0) or 0.0)
                        f.write(
                            f"| {alg_name} | BUILD_ONLY | N/A | N/A | {build_time:.2f} | {memory:.2f} |\n"
                        )
                        continue

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

                    qps = float(alg_results.get('qps', 0.0) or 0.0)
                    query_time = float(alg_results.get('mean_query_time_ms', 0.0) or 0.0)
                    build_time = float(alg_results.get('build_time_s', 0.0) or 0.0)
                    memory = float(alg_results.get('index_memory_mb', 0.0) or 0.0)

                    f.write(
                        f"| {alg_name} | {recall_display} | {qps:.2f}| {query_time:.2f} | {build_time:.2f} | {memory:.2f} |\n"
                    )

                f.write(f"\n\n")

        self.logger.info(f"Summary report written to: {report_path}")
