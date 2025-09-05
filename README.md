# Vector DB Retrieval Guarantee Research

This repository contains code infrastructure for researching new algorithms for vector databases with retrieval guarantees. The code is designed to benchmark and compare different vector retrieval algorithms.

## Setup

```bash
# Create and activate conda environment
conda create -n vectordb-env python=3.10
conda activate vectordb-env

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

- `src/` - Source code
  - `algorithms/` - Implementation of vector retrieval algorithms
  - `benchmark/` - Benchmark datasets and evaluation utilities
  - `experiments/` - Experimental loops and configuration
  - `utils/` - Utility functions and helpers

## Running Experiments

```bash
python -m src.experiments.run_experiment --config configs/default.yaml
```

## Adding New Algorithms

To add a new algorithm, create a new class in `src/algorithms/` that inherits from `BaseAlgorithm` and implements the required methods.