# Vector DB Retrieval Guarantee Research

This repository provides a comprehensive framework for researching, benchmarking, and analyzing vector retrieval algorithms, with a special focus on retrieval guarantees. The codebase is designed to facilitate reproducible experiments and in-depth performance comparisons across various datasets and algorithm configurations.

## Features

- **Extensible Algorithm Framework**: Easily add new vector search algorithms by inheriting from a `BaseAlgorithm` class.
- **Automated Benchmark Suite**: A single script (`scripts/run_full_benchmark.py`) to run a full suite of experiments across multiple datasets.
- **Modular Index/Search Pipelines**: Combine any indexing strategy with any search strategy through declarative config (e.g., pair FAISS HNSW indexing with linear or FAISS searchers).
- **Standard Datasets**: Built-in support for benchmark datasets like SIFT1M, GloVe, and MS MARCO (TF-IDF projection or pre-embedded Cohere vectors), with automated download and preprocessing.
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
- `data/`: Default directory for storing downloaded and processed datasets (configurable via `data_dir`).
- `benchmark_results/`: Default output directory for benchmark reports and raw results (configurable via `output_dir`).

## Setup

```bash
# Create and activate a virtual environment (e.g., using conda)
conda create -n vectordb-env python=3.10
conda activate vectordb-env

# Install dependencies
pip install -r requirements.txt
```

## Testing

Fast smoke checks are available via `pytest`. This runs lightweight algorithm/indexer tests without needing full dataset downloads.

```bash
pytest

# Skip FAISS-dependent tests if the backend is unavailable
pytest -m "not requires_faiss"
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
#below is WIP
python scripts/run_full_benchmark.py --config configs/benchmark_config.yaml   

#below works
python scripts/run_full_benchmark.py --config configs/benchmark_config_test1.yaml 
python scripts/run_full_benchmark.py --config configs/benchmark_config_ms.yaml 
```

The script will automatically download the required datasets if they are not found in the configured `data_dir`, run all experiments, and save the results under the configured `output_dir`.

> **PACE deployment note:** the repository configuration (`configs/benchmark_config.yaml`) points to the shared storage locations:
> - Datasets: `/storage/ice-shared/cs8903onl/vectordb-retrieval/datasets`
> - Benchmark results: `/storage/ice-shared/cs8903onl/vectordb-retrieval/results`
> - MS MARCO (pre-embedded Cohere vectors): `/storage/ice-shared/cs8903onl/vectordb-retrieval/datasets/msmarco_pre_embeded/`
>
> Adjust those paths if you are running on a different machine or prefer a different layout.

### Dataset-specific Options

Dataset entries can carry bespoke options via the `dataset_options` key. For example, the MS MARCO configuration in `configs/benchmark_config.yaml` now consumes the pre-embedded Cohere vectors by pointing at the shared parquet cache, constraining how many passages/queries are loaded, and routing processed caches to the writable results folder. Additional knobs (`query_relevance_offsets_column`, `relevance_candidates_limit`) control how many of the provided top-1k references are considered when constructing ground-truth labels. Tweak those knobs (`base_limit`, `query_limit`, `cache_dir`, etc.) to balance fidelity and runtime for your environment.

### Modular Indexing & Searching

Each benchmark configuration can declare reusable `indexers` and `searchers`, then mix-and-match them per algorithm via references. For example, `exact` uses the `brute_force_l2` indexer together with the `linear_l2` searcher, while the MS MARCO override swaps in cosine-compatible variants. This structure lets you explore new combinations (e.g., FAISS IVF indexer + linear searcher) without touching code—just add a new entry under `algorithms` with the desired `indexer_ref` / `searcher_ref` or inline overrides.

## Benchmark Results

After a benchmark run completes, you will find the following in the `benchmark_results/benchmark_<timestamp>/` directory:

- `benchmark_summary.md`: A human-readable summary of the results in Markdown format.
- `all_results.json`: The complete raw results in JSON format.
- `benchmark.log`: A detailed log of the entire benchmark run.

An example of the performance table from the summary report:

#### Algorithm Performance

| Algorithm | Recall@10 | QPS | Mean Query Time (ms) | Build Time (s) | Index Memory (MB) |
|-----------|-----------|-----|---------------------|----------------|-------------------|
| exact     | 1.0000    | 1.83| 545.39              | 0.01           | 500.00            |
| hnsw      |   | |                 |            |       |


## Adding New Algorithms

To add a new algorithm for benchmarking:

1.  Create a new Python file in `src/algorithms/`.
2.  Define a class that inherits from `src.algorithms.BaseAlgorithm`.
3.  Implement the required abstract methods:
    - `build_index(self, vectors)`: To build the search index from a set of vectors.
    - `search(self, query, k)`: To find the `k` nearest neighbors for a single query vector.
    - `batch_search(self, queries, k)`: To find neighbors for a batch of query vectors.
4.  Add your new algorithm to the `algorithms` section in your `benchmark_config.yaml` file.
