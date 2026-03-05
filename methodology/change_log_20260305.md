# Change Log - 2026-03-05

## Topic

1. Integrated automatic one-page benchmark summary generation.
2. Diagnosed low MSMARCO recall for LSH with current cosine settings.

## What changed

- `src/benchmark/runner.py`
  - Added automatic one-page summary generation at the end of benchmark runs.
  - New outputs per run directory:
    - `one-page-summary.md`
    - `qps_recall_summary.md` (compatibility copy)
    - `qps_recall_<dataset>.svg` (one SVG per dataset with QPS-vs-recall points)
  - The summary includes:
    - per-dataset QPS-vs-recall plots,
    - per-algorithm score table,
    - algorithm implementation details from `<dataset>/<dataset>_config.yaml`,
    - dataset details and concise takeaways.

- `tests/test_benchmark_runner_modular.py`
  - Extended to assert one-page artifacts are generated:
    - `one-page-summary.md`
    - `qps_recall_summary.md`
    - at least one `qps_recall_*.svg`.

## Validation

PACE test commands run:

```bash
cd /home/hice1/pli396/PycharmProjects/vectordb-retrieval
$HOME/scratch/vector-db-venv/bin/python -m pytest -q tests/test_benchmark_runner_modular.py
$HOME/scratch/vector-db-venv/bin/python -m pytest -q tests/test_dataset_msmarco_preembedded_limits.py tests/test_experiment_runner_persistence.py
```

Observed:
- `test_benchmark_runner_modular.py`: `1 passed`
- dataset + persistence tests: `6 passed`

## LSH MSMARCO diagnostic notes

Diagnostic run (PACE, cosine LSH settings from `benchmark_all_datasets_msm100k_covertree_reuse.yaml`):
- `num_tables=12`, `hash_size=12`, `candidate_multiplier=32`, `fallback_to_bruteforce=false`
- Dataset: MSMARCO cached split `(100000, 384)` train, `(70, 384)` queries.

Observed candidate stats:
- candidate count per query: min `247`, mean `354.2`, p90 `427.2`, max `527`
- empty candidate queries: `0`

Observed accuracy:
- `recall@1 = 0.2286`
- `recall@10 = 0.1014`

Interpretation:
- low recall is not caused by empty buckets.
- it is caused by random-hash collision quality under current cosine LSH hyperparameters: the union of collided buckets (~300-500 candidates) often excludes many true nearest neighbors for top-k evaluation.

## Tuned full benchmark run (LSH improvement)

Submitted and completed on PACE:
- Job: `4206146` (`slurm_jobs/codex_all_datasets_msm100k_reuse_ct_lsh_tuned.sbatch`)
- Config: `configs/benchmark_all_datasets_msm100k_covertree_reuse_lsh_tuned.yaml`
- Output: `benchmark_results/benchmark_20260305_070532`
- Baseline for comparison: `benchmark_results/benchmark_20260304_092747`

LSH tuning applied:
- cosine indexer: `num_tables 12 -> 24`, `hash_size 12 -> 8`
- cosine searcher: `candidate_multiplier 32 -> 128`
- l2 LSH settings unchanged

Observed LSH deltas (baseline -> tuned):

- `random`
  - recall@10: `0.3191 -> 0.3191` (no change)
  - qps: `175.48 -> 172.98` (slight decrease)

- `glove50`
  - recall@10: `0.5074 -> 0.5074` (no change)
  - qps: `87.45 -> 81.85` (slight decrease)

- `msmarco`
  - recall@1: `0.2286 -> 0.5857` (`+0.3571`)
  - recall@10: `0.1014 -> 0.3286` (`+0.2271`)
  - qps: `2350.86 -> 147.29` (lower throughput due to larger candidate pools)

Conclusion:
- LSH quality on MSMARCO improved substantially (especially recall),
- at the expected cost of higher query latency for MSMARCO.
