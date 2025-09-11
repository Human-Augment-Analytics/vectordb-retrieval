# Vector DB Retrieval Guarantee Research

This repository provides a comprehensive framework for researching, benchmarking, and analyzing vector retrieval algorithms, with a special focus on retrieval guarantees. The codebase is designed to facilitate reproducible experiments and in-depth performance comparisons across various datasets and algorithm configurations.

## Features

- **Extensible Algorithm Framework**: Easily add new vector search algorithms by inheriting from a `BaseAlgorithm` class.
- **Automated Benchmark Suite**: A single script (`scripts/run_full_benchmark.py`) to run a full suite of experiments across multiple datasets.
- **Standard Datasets**: Built-in support for standard benchmark datasets like SIFT1M and GloVe, with automated download and preprocessing.
- **Comprehensive Metrics**: Tracks key performance indicators including recall, queries per second (QPS), index build time, and index memory usage.
- **Automated Reporting**: Automatically generates detailed Markdown summary reports and raw JSON results for each benchmark run.

## Project Structure

- `src/`: Source code for the experimental framework.
  - `algorithms/`: Implementations of vector retrieval algorithms (e.g., ExactSearch, HNSW).
  - `benchmark/`: Utilities for dataset handling, evaluation, and metrics.
  - `experiments/`: The core experimental runner and configuration management.
- `scripts/`: High-level scripts for automating experiments.
  - `run_full_benchmark.py`: The main entry point for running the full benchmark suite.
- `configs/`: Directory for experiment configuration files (in YAML format).
- `data/`: Default directory for storing downloaded and processed datasets.
- `benchmark_results/`: Default output directory for benchmark reports and raw results.

## Setup

```bash
# Create and activate a virtual environment (e.g., using conda)
conda create -n vectordb-env python=3.10
conda activate vectordb-env

# Install dependencies
pip install -r requirements.txt
```

## Running the Benchmark

The primary way to run experiments is using the full benchmark runner script.

### 1. Create Default Configuration

First, generate the default benchmark configuration file. This file defines which datasets and algorithms to test.

```bash
python scripts/run_full_benchmark.py --create-config
```
This will create `configs/benchmark_config.yaml`. You can edit this file to customize the benchmark run.

### 2. Run the Benchmark Suite

Once the configuration is ready, launch the benchmark suite:

```bash
python scripts/run_full_benchmark.py --config configs/benchmark_config.yaml
```

The script will automatically download the required datasets if they are not found locally, run all experiments, and save the results to the `benchmark_results/` directory.

## Benchmark Results

After a benchmark run completes, you will find the following in the `benchmark_results/benchmark_<timestamp>/` directory:

- `benchmark_summary.md`: A human-readable summary of the results in Markdown format.
- `all_results.json`: The complete raw results in JSON format.
- `benchmark.log`: A detailed log of the entire benchmark run.

An example of the performance table from the summary report:

#### Algorithm Performance

| Algorithm | Recall@10 | QPS | Mean Query Time (ms) | Build Time (s) | Index Memory (MB) |
|-----------|-----------|-----|----------------------|----------------|-------------------|
| exact     | 1.0000    | 1.83| 545.39               | 0.01           | 500.00            |
| hnsw      | 0.9850    | 95.4| 10.48                | 5.23           | 500.00            |


## Adding New Algorithms

To add a new algorithm for benchmarking:

1.  Create a new Python file in `src/algorithms/`.
2.  Define a class that inherits from `src.algorithms.BaseAlgorithm`.
3.  Implement the required abstract methods:
    - `build_index(self, vectors)`: To build the search index from a set of vectors.
    - `search(self, query, k)`: To find the `k` nearest neighbors for a single query vector.
    - `batch_search(self, queries, k)`: To find neighbors for a batch of query vectors.
4.  Add your new algorithm to the `algorithms` section in your `benchmark_config.yaml` file.
