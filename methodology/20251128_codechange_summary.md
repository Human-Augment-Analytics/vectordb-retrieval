# Memory reporting fix (2025-11-28)

## What changed
- Tightened `ExperimentRunner._estimate_memory_usage` to introspect FAISS-based indices, add HNSW graph overhead, and derive code sizes (PQ/SQ) so `index_memory_mb` reflects algorithm-specific footprints instead of always falling back to the raw training vector size.
- Added a best-effort fallback for composite artifacts and LSH wrappers, plus debug logging (file handler only) to keep visibility into the computed components during future runs.

## Validations
- SLURM smoke run (`singlerun_smoke.sbatch`, job `3681003`), config `configs/benchmark_config_smoke.yaml`:
  - `random`: `index_memory_mb` now differentiates algorithms (exact 4.88 MB, hnsw 7.32 MB, ivf_flat 4.88 MB, lsh 23.45 MB).
  - `msmarco`: distinct sizes as well (exact 1464.84 MB, hnsw 1586.91 MB, ivf_flat 1464.84 MB, lsh 1573.95 MB).
- Full benchmark (`singlerun_nomsma_benchmarking_c_v2_pat.sbatch`, job `3681007`, config `configs/benchmark_nomsma_c_v2.yaml`):
  - `random`: cover trees now report their footprint (covertree 5.49 MB, covertree_v2 5.49 MB) alongside PQ/IVF/HNSW (e.g., hnsw 7.32 MB, ivf_pq 1.22 MB, exact 4.88 MB).
  - `glove50`: similarly distinct (covertree 4.43 MB, covertree_v2 4.43 MB, hnsw 6.26 MB, ivf_pq 0.95 MB, exact 3.81 MB).
  - Summary: `benchmark_results/benchmark_20251128_173147/benchmark_summary.md`.

## Notes
- Debug memory logs stay at `DEBUG` level and only emit to the file handler to avoid noisy console output.
- If future FAISS indexes report suspiciously low sizes, the estimator now clamps to at least the expected code footprint per vector and adds HNSW link overhead when available.
