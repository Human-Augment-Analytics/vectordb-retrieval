#!/usr/bin/env python
"""
Algorithm Comparison Script for Vector Retrieval

This script enables systematic comparison of different vector retrieval
algorithms with statistical analysis and comprehensive reporting.
"""

import os
import sys
import argparse
import logging
import yaml
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None
try:
    from scipy import stats
except ImportError:
    stats = None

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.experiments import ExperimentRunner, ExperimentConfig
from src.algorithms import ExactSearch, ApproximateSearch, HNSW

class AlgorithmComparator:
    """
    Comprehensive algorithm comparison with statistical analysis.
    """
    
    def __init__(self, output_dir: str = "algorithm_comparison_results"):
        """
        Initialize the algorithm comparator.
        
        Args:
            output_dir: Directory to save comparison results
        """
        self.output_dir = output_dir
        self.comparison_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(output_dir, f"comparison_{self.comparison_id}")
        
        # Create output directories
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Set up logging
        log_file = os.path.join(self.results_dir, "algorithm_comparison.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("AlgorithmComparator")
        
        self.comparison_results = {}
        
    def load_comparison_config(self, config_file: str) -> Dict[str, Any]:
        """
        Load algorithm comparison configuration from YAML file.
        
        Args:
            config_file: Path to comparison configuration file
            
        Returns:
            Dictionary containing comparison configuration
        """
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        self.logger.info(f"Loaded comparison configuration from {config_file}")
        return config
    
    def run_algorithm_experiments(self, base_config: Dict[str, Any], 
                                 algorithms_config: Dict[str, Dict[str, Any]], 
                                 num_runs: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run experiments for all algorithms with multiple runs for statistical analysis.
        
        Args:
            base_config: Base experiment configuration
            algorithms_config: Dictionary of algorithm configurations
            num_runs: Number of runs for each algorithm
            
        Returns:
            Dictionary mapping algorithm names to lists of results
        """
        all_results = {}
        
        for alg_name, alg_config in algorithms_config.items():
            self.logger.info(f"Running experiments for algorithm: {alg_name}")
            all_results[alg_name] = []
            
            for run_idx in range(num_runs):
                self.logger.info(f"  Run {run_idx + 1}/{num_runs}")
                
                # Create experiment configuration
                config_dict = base_config.copy()
                config_dict['algorithms'] = {alg_name: alg_config}
                config_dict['seed'] = base_config.get('seed', 42) + run_idx  # Different seed for each run
                
                config = ExperimentConfig(**config_dict)
                
                # Create run-specific output directory
                run_output_dir = os.path.join(self.results_dir, f"{alg_name}_run_{run_idx}")
                os.makedirs(run_output_dir, exist_ok=True)
                
                try:
                    # Create and run experiment
                    runner = ExperimentRunner(config, output_dir=run_output_dir)
                    runner.load_dataset()
                    
                    # Get vector dimension
                    dimension = runner.dataset.train_vectors.shape[1]
                    
                    # Create algorithm instance
                    alg_type = alg_config["type"]
                    alg_params = {k: v for k, v in alg_config.items() if k != "type"}
                    
                    if alg_type == "ExactSearch":
                        algorithm = ExactSearch(dimension=dimension, **alg_params)
                    elif alg_type == "ApproximateSearch":
                        algorithm = ApproximateSearch(dimension=dimension, **alg_params)
                    elif alg_type == "HNSW":
                        algorithm = HNSW(dimension=dimension, **alg_params)
                    else:
                        raise ValueError(f"Unknown algorithm type: {alg_type}")
                    
                    # Register and run
                    runner.register_algorithm(algorithm)
                    results = runner.run()
                    
                    # Extract results for this algorithm
                    alg_results = results.get(alg_name, {})
                    alg_results['run_index'] = run_idx
                    alg_results['success'] = True
                    
                    all_results[alg_name].append(alg_results)
                    
                    self.logger.info(f"    Recall@10: {alg_results.get('recall@10', 0):.4f}, "
                                   f"QPS: {alg_results.get('qps', 0):.2f}")
                    
                except Exception as e:
                    self.logger.error(f"Error in run {run_idx} for {alg_name}: {str(e)}")
                    all_results[alg_name].append({
                        'run_index': run_idx,
                        'error': str(e),
                        'success': False
                    })
        
        return all_results
    
    def perform_statistical_analysis(self, results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Perform statistical analysis on the comparison results.
        
        Args:
            results: Dictionary of algorithm results
            
        Returns:
            Dictionary containing statistical analysis results
        """
        self.logger.info("Performing statistical analysis...")
        
        # Metrics to analyze
        metrics = ['recall@1', 'recall@10', 'recall@100', 'qps', 'mean_query_time', 'build_time']
        
        # Prepare data for analysis
        analysis_data = {}
        algorithm_names = list(results.keys())
        
        for metric in metrics:
            analysis_data[metric] = {}
            
            for alg_name in algorithm_names:
                successful_runs = [r for r in results[alg_name] if r.get('success', False)]
                if successful_runs:
                    values = [r.get(metric, 0) for r in successful_runs]
                    analysis_data[metric][alg_name] = {
                        'values': values,
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'median': np.median(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'n_runs': len(values)
                    }
        
        # Perform pairwise statistical tests
        statistical_tests = {}
        if stats is not None:
            for metric in metrics:
                statistical_tests[metric] = {}
                
                for i, alg1 in enumerate(algorithm_names):
                    for j, alg2 in enumerate(algorithm_names[i+1:], i+1):
                        if (alg1 in analysis_data[metric] and alg2 in analysis_data[metric] and
                            len(analysis_data[metric][alg1]['values']) > 1 and
                            len(analysis_data[metric][alg2]['values']) > 1):
                            
                            values1 = analysis_data[metric][alg1]['values']
                            values2 = analysis_data[metric][alg2]['values']
                            
                            try:
                                # Perform t-test
                                t_stat, p_value = stats.ttest_ind(values1, values2)
                                
                                # Perform Mann-Whitney U test (non-parametric)
                                u_stat, u_p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')
                                
                                statistical_tests[metric][f"{alg1}_vs_{alg2}"] = {
                                    't_test': {'statistic': t_stat, 'p_value': p_value},
                                    'mann_whitney': {'statistic': u_stat, 'p_value': u_p_value},
                                    'significant_005': p_value < 0.05,
                                    'significant_001': p_value < 0.01
                                }
                            except Exception as e:
                                self.logger.warning(f"Statistical test failed for {alg1} vs {alg2} on {metric}: {str(e)}")
        else:
            self.logger.warning("scipy not available - skipping statistical tests")
        
        return {
            'descriptive_stats': analysis_data,
            'statistical_tests': statistical_tests
        }
    
    def run_comparison(self, config_file: str):
        """
        Run the complete algorithm comparison.
        
        Args:
            config_file: Path to comparison configuration file
        """
        self.logger.info(f"Starting algorithm comparison: {self.comparison_id}")
        
        # Load configuration
        comparison_config = self.load_comparison_config(config_file)
        
        # Extract configuration components
        base_config = comparison_config.get('base_config', {})
        algorithms_config = comparison_config.get('algorithms', {})
        num_runs = comparison_config.get('num_runs', 3)
        
        self.logger.info(f"Comparing {len(algorithms_config)} algorithms with {num_runs} runs each")
        
        # Run experiments
        start_time = time.time()
        results = self.run_algorithm_experiments(base_config, algorithms_config, num_runs)
        end_time = time.time()
        
        self.logger.info(f"All experiments completed in {end_time - start_time:.2f} seconds")
        
        # Store results
        self.comparison_results = {
            'raw_results': results,
            'experiment_time': end_time - start_time,
            'timestamp': datetime.now().isoformat(),
            'config': comparison_config
        }
        
        # Perform statistical analysis
        statistical_analysis = self.perform_statistical_analysis(results)
        self.comparison_results['statistical_analysis'] = statistical_analysis
        
        # Generate reports and visualizations
        self.generate_reports()
        self.generate_visualizations()
        
        self.logger.info(f"Algorithm comparison completed: {self.comparison_id}")
    
    def generate_reports(self):
        """
        Generate comprehensive comparison reports.
        """
        self.logger.info("Generating comparison reports...")
        
        # Save raw results
        results_file = os.path.join(self.results_dir, "comparison_results.json")
        with open(results_file, 'w') as f:
            json.dump(self.comparison_results, f, indent=2, default=str)
        
        # Generate summary report
        self.generate_summary_report()
        
        # Generate detailed statistical report
        self.generate_statistical_report()
        
        self.logger.info("Reports generated successfully")
    
    def generate_summary_report(self):
        """
        Generate a summary report of the algorithm comparison.
        """
        summary_file = os.path.join(self.results_dir, "comparison_summary.md")
        
        with open(summary_file, 'w') as f:
            f.write(f"# Algorithm Comparison Summary Report\n\n")
            f.write(f"**Comparison ID:** {self.comparison_id}\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
            
            # Overview
            algorithms = list(self.comparison_results['raw_results'].keys())
            f.write(f"## Overview\n\n")
            f.write(f"- **Algorithms Compared:** {len(algorithms)}\n")
            f.write(f"- **Total Experiment Time:** {self.comparison_results['experiment_time']:.2f} seconds\n\n")
            
            # Algorithm performance summary
            f.write(f"## Performance Summary\n\n")
            
            stats = self.comparison_results['statistical_analysis']['descriptive_stats']
            
            # Create performance table
            f.write("| Algorithm | Recall@10 | QPS | Mean Query Time (ms) | Build Time (s) |\n")
            f.write("|-----------|-----------|-----|---------------------|----------------|\n")
            
            for alg_name in algorithms:
                recall = stats.get('recall@10', {}).get(alg_name, {})
                qps = stats.get('qps', {}).get(alg_name, {})
                query_time = stats.get('mean_query_time', {}).get(alg_name, {})
                build_time = stats.get('build_time', {}).get(alg_name, {})
                
                f.write(f"| {alg_name} | ")
                f.write(f"{recall.get('mean', 0):.4f} ± {recall.get('std', 0):.4f} | ")
                f.write(f"{qps.get('mean', 0):.2f} ± {qps.get('std', 0):.2f} | ")
                f.write(f"{query_time.get('mean', 0):.2f} ± {query_time.get('std', 0):.2f} | ")
                f.write(f"{build_time.get('mean', 0):.2f} ± {build_time.get('std', 0):.2f} |\n")
            
            f.write("\n")
            
            # Statistical significance summary
            f.write(f"## Statistical Significance\n\n")
            f.write("Significant differences (p < 0.05) found in:\n\n")
            
            tests = self.comparison_results['statistical_analysis']['statistical_tests']
            for metric, metric_tests in tests.items():
                significant_pairs = [pair for pair, test_result in metric_tests.items() 
                                   if test_result['significant_005']]
                if significant_pairs:
                    f.write(f"**{metric}:**\n")
                    for pair in significant_pairs:
                        p_val = metric_tests[pair]['t_test']['p_value']
                        f.write(f"- {pair.replace('_vs_', ' vs ')}: p = {p_val:.4f}\n")
                    f.write("\n")
        
        self.logger.info(f"Summary report saved to {summary_file}")
    
    def generate_statistical_report(self):
        """
        Generate detailed statistical analysis report.
        """
        stats_file = os.path.join(self.results_dir, "statistical_analysis.md")
        
        with open(stats_file, 'w') as f:
            f.write(f"# Detailed Statistical Analysis\n\n")
            
            stats = self.comparison_results['statistical_analysis']['descriptive_stats']
            tests = self.comparison_results['statistical_analysis']['statistical_tests']
            
            # Descriptive statistics
            f.write(f"## Descriptive Statistics\n\n")
            
            for metric, metric_stats in stats.items():
                f.write(f"### {metric}\n\n")
                f.write("| Algorithm | Mean | Std | Median | Min | Max | N |\n")
                f.write("|-----------|------|-----|--------|-----|-----|---|\n")
                
                for alg_name, alg_stats in metric_stats.items():
                    f.write(f"| {alg_name} | ")
                    f.write(f"{alg_stats['mean']:.4f} | ")
                    f.write(f"{alg_stats['std']:.4f} | ")
                    f.write(f"{alg_stats['median']:.4f} | ")
                    f.write(f"{alg_stats['min']:.4f} | ")
                    f.write(f"{alg_stats['max']:.4f} | ")
                    f.write(f"{alg_stats['n_runs']} |\n")
                
                f.write("\n")
            
            # Statistical tests
            f.write(f"## Statistical Tests\n\n")
            
            for metric, metric_tests in tests.items():
                if metric_tests:
                    f.write(f"### {metric}\n\n")
                    f.write("| Comparison | t-test p-value | Mann-Whitney p-value | Significant (α=0.05) |\n")
                    f.write("|------------|----------------|----------------------|---------------------|\n")
                    
                    for pair, test_result in metric_tests.items():
                        t_p = test_result['t_test']['p_value']
                        mw_p = test_result['mann_whitney']['p_value']
                        sig = "Yes" if test_result['significant_005'] else "No"
                        
                        f.write(f"| {pair.replace('_vs_', ' vs ')} | ")
                        f.write(f"{t_p:.4f} | {mw_p:.4f} | {sig} |\n")
                    
                    f.write("\n")
        
        self.logger.info(f"Statistical analysis report saved to {stats_file}")
    
    def generate_visualizations(self):
        """
        Generate comparison visualizations.
        """
        self.logger.info("Generating visualizations...")
        
        plots_dir = os.path.join(self.results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        stats = self.comparison_results['statistical_analysis']['descriptive_stats']
        
        # Performance comparison bar plots
        metrics_to_plot = ['recall@10', 'qps', 'mean_query_time']
        
        for metric in metrics_to_plot:
            if metric in stats and stats[metric]:
                plt.figure(figsize=(10, 6))
                
                algorithms = list(stats[metric].keys())
                means = [stats[metric][alg]['mean'] for alg in algorithms]
                stds = [stats[metric][alg]['std'] for alg in algorithms]
                
                plt.bar(algorithms, means, yerr=stds, capsize=5, alpha=0.7)
                plt.title(f'Algorithm Comparison: {metric}')
                plt.ylabel(metric)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                plt.savefig(os.path.join(plots_dir, f'{metric}_comparison.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
        
        # Recall vs QPS scatter plot
        if 'recall@10' in stats and 'qps' in stats:
            plt.figure(figsize=(10, 6))
            
            for alg in stats['recall@10'].keys():
                if alg in stats['qps']:
                    recall_mean = stats['recall@10'][alg]['mean']
                    qps_mean = stats['qps'][alg]['mean']
                    recall_std = stats['recall@10'][alg]['std']
                    qps_std = stats['qps'][alg]['std']
                    
                    plt.errorbar(qps_mean, recall_mean, 
                               xerr=qps_std, yerr=recall_std,
                               marker='o', markersize=8, label=alg, capsize=5)
            
            plt.xlabel('Queries Per Second (QPS)')
            plt.ylabel('Recall@10')
            plt.title('Algorithm Performance: Recall vs Speed Trade-off')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(os.path.join(plots_dir, 'recall_vs_qps_comparison.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"Visualizations saved to {plots_dir}")

def create_default_comparison_config():
    """
    Create a default algorithm comparison configuration file.
    """
    config = {
        'base_config': {
            'dataset': 'random',
            'n_queries': 1000,
            'topk': 100,
            'seed': 42
        },
        'num_runs': 3,
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
        }
    }
    
    config_file = 'configs/comparison_config.yaml'
    os.makedirs('configs', exist_ok=True)
    
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Default comparison configuration created: {config_file}")
    return config_file

def main():
    parser = argparse.ArgumentParser(description="Compare vector retrieval algorithms")
    parser.add_argument("--config", type=str, help="Path to comparison configuration file")
    parser.add_argument("--output-dir", type=str, default="algorithm_comparison_results", 
                       help="Directory to save comparison results")
    parser.add_argument("--create-config", action="store_true", 
                       help="Create default comparison configuration file")
    
    args = parser.parse_args()
    
    if args.create_config:
        config_file = create_default_comparison_config()
        print(f"Use: python {__file__} --config {config_file}")
        return
    
    if not args.config:
        print("Error: --config is required. Use --create-config to generate a default configuration.")
        return
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        return
    
    # Run algorithm comparison
    comparator = AlgorithmComparator(output_dir=args.output_dir)
    comparator.run_comparison(args.config)

if __name__ == "__main__":
    main()