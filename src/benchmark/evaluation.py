import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from .metrics import recall_at_k, precision_at_k, mean_average_precision
import time
import pandas as pd
import matplotlib.pyplot as plt

class Evaluator:
    """
    Class for evaluating vector retrieval algorithms on benchmark datasets.
    """

    def __init__(self, ground_truth: np.ndarray):
        """
        Initialize the evaluator.

        Args:
            ground_truth: Ground truth nearest neighbors for test queries
        """
        self.ground_truth = ground_truth
        self.results = {}

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
        k_values = [1, 10, 100]
        metrics = {}

        # Calculate retrieval metrics at different k values
        for k in k_values:
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
        metric_order = [
            'recall@1', 'recall@10', 'recall@100',
            'precision@1', 'precision@10', 'precision@100',
            'map@10',
            'qps', 'mean_query_time', 'median_query_time', 
            'min_query_time', 'max_query_time'
        ]

        # Filter to only include available metrics
        available_metrics = [m for m in metric_order if m in df.columns]

        print("\nEvaluation Results:\n")
        print(df[available_metrics].round(4))

    def plot_recall_vs_qps(self, output_file: Optional[str] = None):
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
        recalls = [self.results[alg].get('recall@10', 0) for alg in algorithms]
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
        ax.set_title('Retrieval Accuracy vs Speed')

        # Add grid and set axis limits
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlim(min(qps) * 0.8, max(qps) * 1.2)
        ax.set_ylim(0, 1.05)

        # Save or show the plot
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_file}")
        else:
            plt.tight_layout()
            plt.show()
