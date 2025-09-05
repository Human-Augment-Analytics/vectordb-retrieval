# Vector Retrieval Experimental Loop - Complete Guide

## Overview

This repository now provides a comprehensive experimental loop for vector retrieval algorithms with retrieval guarantees. The enhanced system enables systematic evaluation, parameter optimization, and algorithm comparison for research purposes.

## Key Features

### ✅ Enhanced Experimental Infrastructure
- **Complete dataset support**: SIFT1M dataset download and processing
- **Multiple algorithm implementations**: ExactSearch, ApproximateSearch, HNSW
- **Comprehensive metrics**: Recall, Precision, MAP, QPS, timing statistics
- **Statistical analysis**: Multi-run experiments with significance testing
- **Automated reporting**: Markdown reports with visualizations

### ✅ Automation Scripts
1. **Full Benchmark Runner** (`scripts/run_full_benchmark.py`)
2. **Parameter Sweep** (`scripts/parameter_sweep.py`) 
3. **Algorithm Comparison** (`scripts/compare_algorithms.py`)

### ✅ Research-Ready Features
- **Reproducible experiments** with seed control
- **Multi-dataset evaluation** (random, SIFT1M, extensible)
- **Performance tracking** with detailed logging
- **Result visualization** and statistical analysis

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Full Benchmark Suite
```bash
# Create default configuration
python scripts/run_full_benchmark.py --create-config

# Run benchmark
python scripts/run_full_benchmark.py --config configs/benchmark_config.yaml
```

### 3. Parameter Optimization
```bash
# Create sweep configuration
python scripts/parameter_sweep.py --create-config

# Run parameter sweep
python scripts/parameter_sweep.py --config configs/sweep_config.yaml
```

### 4. Algorithm Comparison
```bash
# Create comparison configuration
python scripts/compare_algorithms.py --create-config

# Run statistical comparison
python scripts/compare_algorithms.py --config configs/comparison_config.yaml
```

## Detailed Usage

### Full Benchmark Runner

**Purpose**: Run comprehensive benchmarks across multiple datasets and algorithms.

**Features**:
- Multi-dataset evaluation
- Automated result aggregation
- Performance summary reports
- Configurable algorithm parameters

**Example Configuration** (`configs/benchmark_config.yaml`):
```yaml
datasets: ['random', 'sift1m']
n_queries: 1000
topk: 100
algorithms:
  exact:
    type: ExactSearch
    metric: l2
  ivf_flat:
    type: ApproximateSearch
    index_type: IVF100,Flat
    metric: l2
    nprobe: 10
  hnsw:
    type: HNSW
    M: 16
    efConstruction: 200
    efSearch: 100
    metric: l2
```

**Output**:
- `benchmark_results/benchmark_YYYYMMDD_HHMMSS/`
  - `benchmark_summary.md`: Performance summary
  - `all_results.json`: Raw results data
  - Individual dataset results

### Parameter Sweep

**Purpose**: Systematic parameter optimization for algorithms.

**Features**:
- Grid search over parameter ranges
- Performance optimization analysis
- Best configuration identification
- Trade-off visualization

**Example Configuration** (`configs/sweep_config.yaml`):
```yaml
algorithm_name: hnsw_sweep
algorithm_type: HNSW
base_config:
  dataset: random
  n_queries: 500
  topk: 100
parameter_ranges:
  M: [8, 16, 32]
  efConstruction: [100, 200, 400]
  efSearch: [50, 100, 200]
  metric: ['l2']
```

**Output**:
- `parameter_sweep_results/sweep_YYYYMMDD_HHMMSS/`
  - `sweep_summary.md`: Best configurations
  - `sweep_results.csv`: All results
  - `plots/`: Performance visualizations

### Algorithm Comparison

**Purpose**: Statistical comparison of multiple algorithms.

**Features**:
- Multi-run experiments for statistical validity
- T-tests and Mann-Whitney U tests
- Performance confidence intervals
- Significance testing

**Example Configuration** (`configs/comparison_config.yaml`):
```yaml
base_config:
  dataset: random
  n_queries: 1000
  topk: 100
num_runs: 3
algorithms:
  exact:
    type: ExactSearch
    metric: l2
  ivf_flat:
    type: ApproximateSearch
    index_type: IVF100,Flat
    metric: l2
    nprobe: 10
  hnsw:
    type: HNSW
    M: 16
    efConstruction: 200
    efSearch: 100
    metric: l2
```

**Output**:
- `algorithm_comparison_results/comparison_YYYYMMDD_HHMMSS/`
  - `comparison_summary.md`: Performance comparison
  - `statistical_analysis.md`: Detailed statistics
  - `plots/`: Comparison visualizations

## Dataset Support

### Random Dataset
- **Purpose**: Quick testing and development
- **Size**: 10K training vectors, 1K test vectors
- **Dimensions**: 128
- **Ground Truth**: Computed via brute force

### SIFT1M Dataset
- **Purpose**: Standard benchmark evaluation
- **Size**: 1M training vectors, 10K test vectors  
- **Dimensions**: 128
- **Ground Truth**: Provided with dataset
- **Auto-download**: Automatic download and processing

### Adding New Datasets
1. Add dataset info to `Dataset.AVAILABLE_DATASETS`
2. Implement download logic in `_download_[dataset_name]()`
3. Implement processing logic in `_process_[dataset_name]()`

## Metrics and Evaluation

### Retrieval Metrics
- **Recall@k**: Proportion of relevant items retrieved
- **Precision@k**: Proportion of retrieved items that are relevant
- **MAP@k**: Mean Average Precision
- **NDCG@k**: Normalized Discounted Cumulative Gain
- **Hit Rate@k**: Proportion of queries with at least one relevant result
- **MRR**: Mean Reciprocal Rank

### Performance Metrics
- **QPS**: Queries per second
- **Query Time**: Mean/median/percentile latencies
- **Build Time**: Index construction time
- **Memory Usage**: Index memory footprint (future)

### Statistical Analysis
- **Descriptive Statistics**: Mean, std, median, min, max
- **Significance Testing**: T-tests, Mann-Whitney U tests
- **Confidence Intervals**: Performance uncertainty quantification

## Research Applications

### 1. Algorithm Development
- Test new retrieval algorithms
- Compare against baselines
- Optimize parameters systematically

### 2. Performance Analysis
- Understand recall vs speed trade-offs
- Analyze scalability characteristics
- Identify optimal configurations

### 3. Reproducible Research
- Version-controlled experiments
- Standardized evaluation protocols
- Statistical significance validation

### 4. Retrieval Guarantees
- Track theoretical guarantees
- Validate approximation ratios
- Analyze worst-case performance

## Configuration Options

### Experiment Parameters
- `n_queries`: Number of test queries
- `topk`: Number of results to retrieve
- `repeat`: Number of experiment repetitions
- `seed`: Random seed for reproducibility

### Algorithm Parameters
- **ExactSearch**: `metric` (l2, cosine, dot)
- **ApproximateSearch**: `index_type`, `nprobe`, `metric`
- **HNSW**: `M`, `efConstruction`, `efSearch`, `metric`

### Output Options
- `output_prefix`: Prefix for result files
- Custom output directories
- Configurable logging levels

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce dataset size or `n_queries`
2. **Slow Performance**: Use smaller parameter ranges
3. **Import Errors**: Install missing dependencies
4. **Segmentation Faults**: Check algorithm implementations

### Performance Tips

1. **Start Small**: Use random dataset for initial testing
2. **Parallel Processing**: Run multiple sweeps in parallel
3. **Resource Monitoring**: Monitor memory and CPU usage
4. **Result Caching**: Leverage dataset caching

## Future Enhancements

### Planned Features
- [ ] Memory usage monitoring
- [ ] Distributed experiment execution
- [ ] Interactive result visualization
- [ ] More benchmark datasets (Deep1B, GIST1M)
- [ ] Advanced statistical analysis
- [ ] Theoretical guarantee tracking

### Contributing
1. Add new algorithms in `src/algorithms/`
2. Implement new metrics in `src/benchmark/metrics.py`
3. Add dataset support in `src/benchmark/dataset.py`
4. Create new automation scripts in `scripts/`

## Summary

The enhanced experimental loop provides a complete research infrastructure for vector retrieval algorithms with:

- **Systematic Evaluation**: Automated benchmarking across datasets
- **Parameter Optimization**: Grid search and performance analysis  
- **Statistical Validation**: Multi-run experiments with significance testing
- **Comprehensive Reporting**: Automated analysis and visualization
- **Research Reproducibility**: Version control and standardized protocols

This system enables rigorous research on vector retrieval algorithms with retrieval guarantees, supporting both algorithm development and performance analysis.