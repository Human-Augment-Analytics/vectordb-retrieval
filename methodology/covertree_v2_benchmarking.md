# CoverTreeV2 Benchmarking Guide

This note explains how to run the perfect-recall CoverTree baseline through our benchmarking stack and how to interpret the results produced by `configs/benchmark_nomsma_c_v2.yaml`.

---

## Configuration (`configs/benchmark_nomsma_c_v2.yaml`)

| Section | Key Points |
|---------|------------|
| `algorithms.covertree_v2` | Declares the `CoverTreeV2` type with metric `l2`. No additional knobs are required because the implementation always traverses the entire tree when necessary. |
| Dataset overrides | `random` and `glove50` inherit the global algorithm list; the GloVe block keeps an empty override (`{}`) to make it easy to add dataset-specific tweaks later. |
| Shared settings | `query_batch_size=128`, `repeat=2`, and `topk=200` match the non-MS-MARCO benchmark profile so we can compare against FAISS/HNSW/LSH on equal footing. |

If you need cosine search, set `metric: cosine` under `algorithms.covertree_v2` and normalize the dataset (the implementation handles the unit-length conversion internally).

---

## SLURM Job (`slurm_jobs/singlerun_nomsma_benchmarking_c_v2_pat.sbatch`)

1. The script provisions a `uv` virtual environment under `$HOME/scratch/vector-db-venv`, installs `requirements.txt`, and then runs:

   ```bash
   python scripts/run_full_benchmark.py --config configs/benchmark_nomsma_c_v2.yaml
   ```

2. Submit the job from the repo root:

   ```bash
   sbatch slurm_jobs/singlerun_nomsma_benchmarking_c_v2_pat.sbatch
   ```

3. Logs land under `slurm_jobs/slurm_logs/VectorDB-Retrieval-Guarantee_FULL-<jobid>-<node>.log`. The latest successful run is `VectorDB-Retrieval-Guarantee_FULL-3531974-atl1-1-02-003-20-1.log`.

---

## Latest Results (Job 3531974, 2025‑11‑13 08:20 UTC)

| Dataset | Recall@10 (covertree_v2) | Mean Query Time (ms) | Notes |
|---------|-------------------------|----------------------|-------|
| `random` | 1.0000 | 21.0 | Matches brute force with ~5× slower queries than CoverTree v1 due to exhaustive traversal. |
| `glove50` | 1.0000 | 85.7 | Perfect recall verified even with `topk=200`; latency reflects 20 k train vectors. |

Artifacts are stored under `benchmark_results/benchmark_20251113_080327/` with per-dataset plots in the `plots_*/recall_vs_qps.png` directories.

---

## Debugging Tips

- If `covertree_v2` ever reports recall < 1.0, inspect the run log to ensure the dataset size matches expectations—CoverTreeV2 falls back to enumerating unseen indices, so imperfect recall usually signals data preparation or evaluation issues.  
- The implementation currently stores a view of each vector in every node (`src/algorithms/covertree_v2.py`), so memory grows linearly with the number of nodes. For very large datasets, switch back to CoverTree (v1) or add memmap-backed storage.  
- To run smaller smoke tests, override `datasets.random.n_queries` or `repeat` in the config instead of tweaking the algorithm: CoverTreeV2 intentionally does not expose pruning knobs.
