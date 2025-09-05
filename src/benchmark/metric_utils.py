import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Any
from .metrics import *

def evaluate_all_metrics(ground_truth: np.ndarray, predicted: np.ndarray, k_values: List[int]) -> Dict[str, Dict[int, float]]:
    """
    Evaluate all available metrics at different k values.

    Args:
        ground_truth: Ground truth indices for each query (n_queries, n_ground_truth)
        predicted: Predicted indices for each query (n_queries, n_predicted)
        k_values: List of k values to evaluate

    Returns:
        Dictionary of metric results at different k values
    """
    results = {
        'recall': {},
        'precision': {},
        'ndcg': {},
        'hit_rate': {},
        'mrr': {}
    }

    # MAP is evaluated once with the max k
    max_k = max(k_values)
    map_score = mean_average_precision(ground_truth, predicted, max_k)
    results['map'] = {max_k: map_score}

    # Evaluate each metric at each k value
    for k in k_values:
        results['recall'][k] = recall_at_k(ground_truth, predicted, k)
        results['precision'][k] = precision_at_k(ground_truth, predicted, k)
        results['ndcg'][k] = ndcg_at_k(ground_truth, predicted, k)
        results['hit_rate'][k] = hit_rate_at_k(ground_truth, predicted, k)

    # MRR is calculated once using the maximum k
    mrr_score = mean_reciprocal_rank(ground_truth, predicted, max_k)
    results['mrr'] = {max_k: mrr_score}

    return results

def plot_metrics_by_k(results: Dict[str, Dict[int, float]], title: str = "Metrics by k") -> plt.Figure:
    """
    Plot metrics against k values.

    Args:
        results: Dictionary of metric results as returned by evaluate_all_metrics
        title: Title for the plot

    Returns:
        The matplotlib Figure object
    """
    metrics_to_plot = ['recall', 'precision', 'ndcg', 'hit_rate']

    fig, ax = plt.subplots(figsize=(10, 6))

    for metric in metrics_to_plot:
        if metric in results:
            k_values = sorted(results[metric].keys())
            values = [results[metric][k] for k in k_values]
            ax.plot(k_values, values, marker='o', label=metric.capitalize())

    ax.set_xlabel('k')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    plt.tight_layout()
    return fig

def compare_algorithms(results_by_algo: Dict[str, Dict[str, Dict[int, float]]], 
                      metric: str, k_values: List[int]) -> plt.Figure:
    """
    Compare different algorithms by plotting a specified metric across k values.

    Args:
        results_by_algo: Dictionary mapping algorithm names to their metric results
        metric: The metric to compare ('recall', 'precision', etc.)
        k_values: List of k values to plot

    Returns:
        The matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo_name, results in results_by_algo.items():
        if metric in results:
            values = [results[metric].get(k, 0) for k in k_values]
            ax.plot(k_values, values, marker='o', label=algo_name)

    ax.set_xlabel('k')
    ax.set_ylabel(f'{metric.capitalize()} Score')
    ax.set_title(f'Comparison of {metric.capitalize()} across Algorithms')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    plt.tight_layout()
    return fig

def summarize_results(results: Dict[str, Dict[int, float]]) -> str:
    """
    Generate a text summary of metric results.

    Args:
        results: Dictionary of metric results

    Returns:
        A formatted string summarizing the results
    """
    summary = "===== Metric Results Summary =====\n"

    # Single-value metrics
    if 'map' in results:
        k = list(results['map'].keys())[0]
        summary += f"MAP@{k}: {results['map'][k]:.4f}\n"

    if 'mrr' in results:
        k = list(results['mrr'].keys())[0]
        summary += f"MRR@{k}: {results['mrr'][k]:.4f}\n"

    # Multi-k metrics
    multi_k_metrics = ['recall', 'precision', 'ndcg', 'hit_rate']
    for metric in multi_k_metrics:
        if metric in results:
            summary += f"\n{metric.capitalize()} at different k values:\n"
            for k in sorted(results[metric].keys()):
                summary += f"  {metric.capitalize()}@{k}: {results[metric][k]:.4f}\n"

    return summary
