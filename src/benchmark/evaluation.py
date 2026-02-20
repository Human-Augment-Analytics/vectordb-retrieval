import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Iterable
from .metrics import recall_at_k, precision_at_k, mean_average_precision
import time
import pandas as pd
import matplotlib.pyplot as plt

class Evaluator:
    """
    Class for evaluating vector retrieval algorithms on benchmark datasets.
    """

    def __init__(self, ground_truth: np.ndarray, k_values: Optional[Iterable[int]] = None):
        """
        Initialize the evaluator.

        Args:
            ground_truth: Ground truth nearest neighbors for test queries
            k_values: Iterable of cut-offs to evaluate (defaults to {1, 10, 100})
        """
        self.ground_truth = ground_truth
        self.results = {}
        default_k_values = [1, 10, 100]
        if k_values is None:
            self.k_values = default_k_values
        else:
            merged = set(default_k_values)
            merged.update(k_values)
            self.k_values = sorted(merged)

    def evaluate(self, algorithm_name: str, predicted_indices: np.ndarray, 
                query_times: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the retrieval results against ground truth.

        Args:
            algorithm_name: Name of the algorithm
            predicted_indices: Predicted nearest neighbor indices
            query_times: Query times in seconds for each query

        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}

        # Calculate retrieval metrics at different k values
        for k in self.k_values:
            if k <= predicted_indices.shape[1]:
                metrics[f'recall@{k}'] = recall_at_k(self.ground_truth, predicted_indices, k)
                metrics[f'precision@{k}'] = precision_at_k(self.ground_truth, predicted_indices, k)

        # Calculate MAP if we have enough predictions
        if predicted_indices.shape[1] >= 10:
            metrics['map@10'] = mean_average_precision(self.ground_truth, predicted_indices, 10)

        # Query time statistics
        metrics['qps'] = 1.0 / np.mean(query_times)  # Queries per second
        metrics['mean_query_time'] = np.mean(query_times) * 1000  # Convert to ms
        metrics['median_query_time'] = np.median(query_times) * 1000  # Convert to ms
        metrics['min_query_time'] = np.min(query_times) * 1000  # Convert to ms
        metrics['max_query_time'] = np.max(query_times) * 1000  # Convert to ms

        # Store results
        self.results[algorithm_name] = metrics

        return metrics

    def _resolve_operations_metric(
        self,
        algorithms: List[str],
    ) -> Tuple[str, List[float], str]:
        """
        Resolve which metric to use as the X-axis for operations/cost plots.

        Returns:
            Tuple of (metric_key, metric_values, axis_label)
        """
        # Prefer explicit operation-count style metrics when available.
        candidates = [
            ("operations_per_query", "Operations / Query"),
            ("operation_count", "Operations"),
            ("distance_computations", "Distance Computations"),
            ("distance_operations", "Distance Operations"),
            ("mean_query_time_ms", "Mean Query Time (ms)"),
            ("mean_query_time", "Mean Query Time (ms)"),
            ("total_query_time_s", "Total Query Time (s)"),
        ]

        for metric_key, axis_label in candidates:
            values: List[float] = []
            for alg in algorithms:
                raw_value = self.results[alg].get(metric_key)
                if raw_value is None:
                    values = []
                    break
                try:
                    numeric_value = float(raw_value)
                except (TypeError, ValueError):
                    values = []
                    break
                if not np.isfinite(numeric_value):
                    values = []
                    break
                values.append(numeric_value)
            if values:
                return metric_key, values, axis_label

        # Last-resort fallback: derive per-query time (ms) from QPS.
        derived_values: List[float] = []
        for alg in algorithms:
            qps_value = self.results[alg].get("qps")
            try:
                qps = float(qps_value) if qps_value is not None else 0.0
            except (TypeError, ValueError):
                qps = 0.0
            if qps <= 0:
                return "qps", [], "Mean Query Time (ms, derived from QPS)"
            derived_values.append((1.0 / qps) * 1000.0)

        return "qps", derived_values, "Mean Query Time (ms, derived from QPS)"

    def print_results(self):
        """
        Print a summary of evaluation results for all algorithms.
        """
        if not self.results:
            print("No evaluation results available.")
            return

        # Create a DataFrame for easier comparison
        df = pd.DataFrame(self.results).T

        # Reorder columns for better readability
        recall_cols = [f'recall@{k}' for k in self.k_values]
        precision_cols = [f'precision@{k}' for k in self.k_values]
        metric_order = recall_cols + precision_cols + [
            'map@10',
            'qps', 'mean_query_time', 'median_query_time', 
            'min_query_time', 'max_query_time'
        ]

        available_metrics = [m for m in metric_order if m in df.columns]

        print("\nEvaluation Results:\n")
        print(df[available_metrics].round(4))

    def plot_recall_vs_qps(
        self,
        output_file: Optional[str] = None,
        title_suffix: Optional[str] = None,
    ) -> None:
        """
        Plot recall@k vs queries per second for all algorithms.

        Args:
            output_file: Optional file to save the plot
        """
        if not self.results:
            print("No evaluation results available for plotting.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract recall@10 and QPS for each algorithm
        algorithms = list(self.results.keys())
        target_k = min(10, max(self.k_values))
        recalls = [self.results[alg].get(f'recall@{target_k}', 0) for alg in algorithms]
        qps = [self.results[alg].get('qps', 0) for alg in algorithms]

        # Create scatter plot
        ax.scatter(qps, recalls, s=100)

        # Add labels for each point
        for i, alg in enumerate(algorithms):
            ax.annotate(alg, (qps[i], recalls[i]), fontsize=9,
                       xytext=(5, 5), textcoords='offset points')

        # Set axis labels and title
        ax.set_xlabel('Queries Per Second (QPS)')
        ax.set_ylabel('Recall@10')
        title = 'Retrieval Accuracy vs Speed'
        if title_suffix:
            title = f"{title} — {title_suffix}"
        ax.set_title(title)

        # Add grid and set axis limits
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlim(min(qps) * 0.8, max(qps) * 1.2)
        ax.set_ylim(0, 1.05)

        # Save or show the plot
        if output_file:
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_file}")
        else:
            plt.tight_layout()
            plt.show()

    def plot_operations_vs_recall(
        self,
        output_file: Optional[str] = None,
        title_suffix: Optional[str] = None,
    ) -> None:
        """
        Plot operations/cost vs recall@k for all algorithms.

        Args:
            output_file: Optional file to save the plot
            title_suffix: Optional dataset/context suffix appended to title
        """
        if not self.results:
            print("No evaluation results available for plotting.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        algorithms = list(self.results.keys())
        target_k = min(10, max(self.k_values))
        recalls = [self.results[alg].get(f"recall@{target_k}", 0) for alg in algorithms]
        metric_key, operations, x_label = self._resolve_operations_metric(algorithms)

        if not operations:
            print("No operations/cost metric available for plotting.")
            plt.close(fig)
            return

        ax.scatter(operations, recalls, s=100)

        for i, alg in enumerate(algorithms):
            ax.annotate(
                alg,
                (operations[i], recalls[i]),
                fontsize=9,
                xytext=(5, 5),
                textcoords="offset points",
            )

        if len(operations) >= 2:
            positive_ops = [val for val in operations if val > 0]
            if positive_ops and (max(positive_ops) / max(min(positive_ops), 1e-12) >= 20):
                ax.set_xscale("log")

            min_op = min(operations)
            max_op = max(operations)
            if max_op > min_op:
                pad = (max_op - min_op) * 0.2
                ax.set_xlim(min_op - pad, max_op + pad)
        ax.set_ylim(0, 1.05)

        ax.set_xlabel(x_label)
        ax.set_ylabel(f"Recall@{target_k}")
        title = "Operations vs Recall Trade-off"
        if title_suffix:
            title = f"{title} — {title_suffix}"
        ax.set_title(title)

        # Surface which metric powered the x-axis for traceability.
        ax.text(
            0.01,
            0.01,
            f"x-axis metric: {metric_key}",
            transform=ax.transAxes,
            fontsize=8,
            alpha=0.7,
        )

        ax.grid(True, linestyle="--", alpha=0.7)

        if output_file:
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {output_file}")
        else:
            plt.tight_layout()
            plt.show()
