#!/usr/bin/env python
"""
Parameter Sweep Script for Vector Retrieval Algorithms

This script enables systematic parameter optimization for vector retrieval
algorithms to find optimal configurations and analyze retrieval guarantees.
"""

import os
import sys
import argparse
import logging
import yaml
import json
import itertools
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.experiments import ExperimentRunner, ExperimentConfig
from src.algorithms import ExactSearch, ApproximateSearch, HNSW

class ParameterSweepRunner:
    """
    Parameter sweep runner for systematic algorithm optimization.
    """
    
    def __init__(self, output_dir: str = "parameter_sweep_results"):
        """
        Initialize the parameter sweep runner.
        
        Args:
            output_dir: Directory to save sweep results
        """
        self.output_dir = output_dir
        self.sweep_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(output_dir, f"sweep_{self.sweep_id}")
        
        # Create output directories
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Set up logging
        log_file = os.path.join(self.results_dir, "parameter_sweep.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("ParameterSweep")
        
        self.sweep_results = []
        
    def load_sweep_config(self, config_file: str) -> Dict[str, Any]:
        """
        Load parameter sweep configuration from YAML file.
        
        Args:
            config_file: Path to sweep configuration file
            
        Returns:
            Dictionary containing sweep configuration
        """
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        self.logger.info(f"Loaded sweep configuration from {config_file}")
        return config
    
    def generate_parameter_combinations(self, param_ranges: Dict[str, List]) -> List[Dict[str, Any]]:
        """
        Generate all combinations of parameters for the sweep.
        
        Args:
            param_ranges: Dictionary mapping parameter names to lists of values
            
        Returns:
            List of parameter combinations
        """
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        combinations = []
        for combo in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)
        
        self.logger.info(f"Generated {len(combinations)} parameter combinations")
        return combinations
    
    def run_single_configuration(self, base_config: Dict[str, Any], 
                                algorithm_name: str, algorithm_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run experiment with a single parameter configuration.
        
        Args:
            base_config: Base experiment configuration
            algorithm_name: Name of the algorithm to test
            algorithm_params: Parameters for the algorithm
            
        Returns:
            Dictionary containing experiment results
        """
        # Create modified configuration
        config_dict = base_config.copy()
        config_dict['algorithms'] = {
            algorithm_name: algorithm_params
        }
        
        config = ExperimentConfig(**config_dict)
        
        # Create temporary output directory
        temp_output_dir = os.path.join(self.results_dir, "temp")
        os.makedirs(temp_output_dir, exist_ok=True)
        
        try:
            # Create and run experiment
            runner = ExperimentRunner(config, output_dir=temp_output_dir)
            runner.load_dataset()
            
            # Get vector dimension
            dimension = runner.dataset.train_vectors.shape[1]
            
            # Create algorithm instance
            alg_type = algorithm_params["type"]
            alg_params = {k: v for k, v in algorithm_params.items() if k != "type"}
            
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
            
            # Extract key metrics
            alg_results = results.get(algorithm_name, {})
            return {
                'parameters': algorithm_params,
                'recall@1': alg_results.get('recall@1', 0),
                'recall@10': alg_results.get('recall@10', 0),
                'recall@100': alg_results.get('recall@100', 0),
                'qps': alg_results.get('qps', 0),
                'mean_query_time': alg_results.get('mean_query_time', 0),
                'build_time': alg_results.get('build_time', 0),
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Error running configuration {algorithm_params}: {str(e)}")
            return {
                'parameters': algorithm_params,
                'error': str(e),
                'success': False
            }
    
    def run_parameter_sweep(self, config_file: str):
        """
        Run the complete parameter sweep.
        
        Args:
            config_file: Path to sweep configuration file
        """
        self.logger.info(f"Starting parameter sweep: {self.sweep_id}")
        
        # Load configuration
        sweep_config = self.load_sweep_config(config_file)
        
        # Extract base configuration and sweep parameters
        base_config = sweep_config.get('base_config', {})
        algorithm_name = sweep_config.get('algorithm_name', 'test_algorithm')
        algorithm_type = sweep_config.get('algorithm_type', 'HNSW')
        parameter_ranges = sweep_config.get('parameter_ranges', {})
        
        # Generate parameter combinations
        param_combinations = self.generate_parameter_combinations(parameter_ranges)
        
        self.logger.info(f"Running sweep for {algorithm_name} ({algorithm_type}) with {len(param_combinations)} configurations")
        
        # Run experiments for each parameter combination
        for i, params in enumerate(param_combinations):
            self.logger.info(f"Running configuration {i+1}/{len(param_combinations)}: {params}")
            
            # Add algorithm type to parameters
            algorithm_params = params.copy()
            algorithm_params['type'] = algorithm_type
            
            start_time = time.time()
            result = self.run_single_configuration(base_config, algorithm_name, algorithm_params)
            end_time = time.time()
            
            # Add timing and index information
            result['config_index'] = i
            result['experiment_time'] = end_time - start_time
            result['timestamp'] = datetime.now().isoformat()
            
            self.sweep_results.append(result)
            
            if result['success']:
                self.logger.info(f"  Recall@10: {result['recall@10']:.4f}, QPS: {result['qps']:.2f}")
            else:
                self.logger.warning(f"  Configuration failed: {result.get('error', 'Unknown error')}")
        
        # Generate analysis and reports
        self.analyze_results()
        self.generate_reports()
        
        self.logger.info(f"Parameter sweep completed: {self.sweep_id}")
    
    def analyze_results(self):
        """
        Analyze sweep results to find optimal configurations.
        """
        self.logger.info("Analyzing sweep results...")
        
        # Filter successful results
        successful_results = [r for r in self.sweep_results if r['success']]
        
        if not successful_results:
            self.logger.warning("No successful configurations found!")
            return
        
        # Convert to DataFrame for analysis
        df_data = []
        for result in successful_results:
            row = result['parameters'].copy()
            row.update({
                'recall@1': result['recall@1'],
                'recall@10': result['recall@10'],
                'recall@100': result['recall@100'],
                'qps': result['qps'],
                'mean_query_time': result['mean_query_time'],
                'build_time': result['build_time']
            })
            df_data.append(row)
        
        self.results_df = pd.DataFrame(df_data)
        
        # Find best configurations for different metrics
        self.best_configs = {
            'best_recall@10': self.results_df.loc[self.results_df['recall@10'].idxmax()],
            'best_qps': self.results_df.loc[self.results_df['qps'].idxmax()],
            'best_balanced': self.results_df.loc[(self.results_df['recall@10'] * self.results_df['qps']).idxmax()]
        }
        
        self.logger.info("Analysis completed")
    
    def generate_reports(self):
        """
        Generate comprehensive reports and visualizations.
        """
        self.logger.info("Generating reports...")
        
        # Save raw results
        results_file = os.path.join(self.results_dir, "sweep_results.json")
        with open(results_file, 'w') as f:
            json.dump(self.sweep_results, f, indent=2, default=str)
        
        # Save results DataFrame
        if hasattr(self, 'results_df'):
            csv_file = os.path.join(self.results_dir, "sweep_results.csv")
            self.results_df.to_csv(csv_file, index=False)
        
        # Generate summary report
        self.generate_summary_report()
        
        # Generate visualizations
        self.generate_visualizations()
        
        self.logger.info("Reports generated successfully")
    
    def generate_summary_report(self):
        """
        Generate a summary report of the parameter sweep.
        """
        summary_file = os.path.join(self.results_dir, "sweep_summary.md")
        
        with open(summary_file, 'w') as f:
            f.write(f"# Parameter Sweep Summary Report\n\n")
            f.write(f"**Sweep ID:** {self.sweep_id}\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
            
            # Overall statistics
            total_configs = len(self.sweep_results)
            successful_configs = len([r for r in self.sweep_results if r['success']])
            
            f.write(f"## Overview\n\n")
            f.write(f"- **Total Configurations:** {total_configs}\n")
            f.write(f"- **Successful Configurations:** {successful_configs}\n")
            f.write(f"- **Success Rate:** {successful_configs/total_configs*100:.1f}%\n\n")
            
            if hasattr(self, 'best_configs'):
                f.write(f"## Best Configurations\n\n")
                
                for metric_name, config in self.best_configs.items():
                    f.write(f"### {metric_name.replace('_', ' ').title()}\n\n")
                    f.write(f"**Performance:**\n")
                    f.write(f"- Recall@10: {config['recall@10']:.4f}\n")
                    f.write(f"- QPS: {config['qps']:.2f}\n")
                    f.write(f"- Mean Query Time: {config['mean_query_time']:.2f} ms\n")
                    f.write(f"- Build Time: {config['build_time']:.2f} s\n\n")
                    
                    f.write(f"**Parameters:**\n")
                    for param, value in config.items():
                        if param not in ['recall@1', 'recall@10', 'recall@100', 'qps', 'mean_query_time', 'build_time']:
                            f.write(f"- {param}: {value}\n")
                    f.write("\n")
        
        self.logger.info(f"Summary report saved to {summary_file}")
    
    def generate_visualizations(self):
        """
        Generate visualization plots for the parameter sweep results.
        """
        if not hasattr(self, 'results_df') or self.results_df.empty:
            self.logger.warning("No data available for visualization")
            return
        
        plots_dir = os.path.join(self.results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Recall vs QPS scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(self.results_df['qps'], self.results_df['recall@10'], alpha=0.6)
        plt.xlabel('Queries Per Second (QPS)')
        plt.ylabel('Recall@10')
        plt.title('Parameter Sweep: Recall vs Speed Trade-off')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, 'recall_vs_qps.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Parameter correlation heatmap (if multiple numeric parameters)
        numeric_cols = self.results_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 2:
            plt.figure(figsize=(12, 8))
            correlation_matrix = self.results_df[numeric_cols].corr()
            plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
            plt.colorbar()
            plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45)
            plt.yticks(range(len(numeric_cols)), numeric_cols)
            plt.title('Parameter Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'parameter_correlation.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"Visualizations saved to {plots_dir}")

def create_default_sweep_config():
    """
    Create a default parameter sweep configuration file.
    """
    config = {
        'algorithm_name': 'hnsw_sweep',
        'algorithm_type': 'HNSW',
        'base_config': {
            'dataset': 'random',
            'n_queries': 500,
            'topk': 100,
            'seed': 42
        },
        'parameter_ranges': {
            'M': [8, 16, 32],
            'efConstruction': [100, 200, 400],
            'efSearch': [50, 100, 200],
            'metric': ['l2']
        }
    }
    
    config_file = 'configs/sweep_config.yaml'
    os.makedirs('configs', exist_ok=True)
    
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Default sweep configuration created: {config_file}")
    return config_file

def main():
    parser = argparse.ArgumentParser(description="Run parameter sweep for vector retrieval algorithms")
    parser.add_argument("--config", type=str, help="Path to sweep configuration file")
    parser.add_argument("--output-dir", type=str, default="parameter_sweep_results", 
                       help="Directory to save sweep results")
    parser.add_argument("--create-config", action="store_true", 
                       help="Create default sweep configuration file")
    
    args = parser.parse_args()
    
    if args.create_config:
        config_file = create_default_sweep_config()
        print(f"Use: python {__file__} --config {config_file}")
        return
    
    if not args.config:
        print("Error: --config is required. Use --create-config to generate a default configuration.")
        return
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        return
    
    # Run parameter sweep
    runner = ParameterSweepRunner(output_dir=args.output_dir)
    runner.run_parameter_sweep(args.config)

if __name__ == "__main__":
    main()