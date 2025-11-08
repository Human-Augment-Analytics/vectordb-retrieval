# LSH Benchmarking Status

*Last updated: 2025-11-08*

During the `benchmark_results/benchmark_20251108_095624/` run (config `configs/benchmark_config.yaml`), the LSH rows report **perfect recall and precision identical to ExactSearch** on both random and GloVe50. That is not expected from the current LSH pipeline (8 tables, 18-bit hashes, candidate multiplier 8). Hypotheses:

1. **Indexer/searcher fallback path**: `LSHSearcher` may always fall back to brute-force because the candidate set is empty; if so we are effectively re-running exact search.
2. **Metrics mis-plumbed**: Results may be picked up from the wrong algorithm key when the evaluator aggregates metrics.
3. **Dataset-specific overrides**: Some configs might replace LSH with the brute-force indexer for particular datasets.

### Next Steps

- Reproduce the issue by running `python scripts/run_full_benchmark.py --config configs/benchmark_config.yaml` (or PACE `slurm_jobs/singlerun_complete_benchmarking_pat.sbatch`). Inspect `benchmark_results/.../random/lsh_results.json` and `.../glove50/lsh_results.json` to confirm the candidate indices really match exact search.
- Instrument `src/algorithms/modular.py` and `src/algorithms/lsh.py` to log when the fallback path is hit.
- Validate whether `candidate_multiplier` or hash params are insufficient after we shrank the random dataset (5k train vectors). Adjust parameters or regenerate the dataset so LSH exercises its approximate behavior.

Document findings and fixes here so future agents know the resolution before re-running the full benchmark.
