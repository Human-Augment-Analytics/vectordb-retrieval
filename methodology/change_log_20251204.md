# Change Log — 2025-12-04

Summary of today’s updates across code, configs, documentation, and benchmarks (faiss IndexLSH rerank + validation run).

## Code
- `src/algorithms/modular.py`:
  - `FaissLSHIndexer` now tags artifacts with `faiss_index_kind="lsh"` so searchers can detect IndexLSH.
  - `FaissSearcher` detects the LSH tag and optionally reranks an expanded candidate set (`lsh_candidate_multiplier`, `lsh_max_candidates`, `lsh_rerank`) against the original vectors using the true metric (L2/cosine/IP). Non-LSH FAISS indexes keep the original fast path. Queries/vectors are normalized for cosine rerank when needed.
- Tests: added `test_faiss_searcher_lsh_reranks_candidates_without_faiss_dependency` to validate the rerank path with a dummy LSH index and a patched `faiss` symbol (no FAISS backend required).

## Configuration & Docs
- `configs/benchmark_nomsma_c_v2.yaml`: the shared `faiss_l2` searcher now carries `lsh_candidate_multiplier: 64.0` so the `faiss_lsh` algorithm reranks 64×k candidates.
- `README.md`: noted that the FAISS IndexLSH variant reranks expanded candidate sets for better recall.
- `methodology/known_followups.md`: recorded the IndexLSH recall issue and the rerank-based fix (item 7).

## Benchmarks Run (configs/benchmark_nomsma_c_v2.yaml)
- SLURM job: `3911643` via `slurm_jobs/singlerun_nomsma_benchmarking_c_v2_pat.sbatch`.
- Outputs: `benchmark_results/benchmark_20251204_152957/`; log at `slurm_jobs/slurm_logs/VectorDB-Retrieval-Guarantee_FULL-3911643-atl1-1-01-005-4-2.log`.
- Goal: verify FAISS IndexLSH after rerank changes.
- Key `faiss_lsh` results (recall/QPS/mean_query_time_ms/build_time_s/index_mem_MB):
  - random: `0.9672 / 3700.7 / 0.27 / 0.20 / 0.61` (previous run `benchmark_20251204_091614` was recall `0.2133` with inflated QPS from the non-reranked path).
  - glove50: `0.9980 / 1708.7 / 0.59 / 0.04 / 0.61` (previous run recall `0.3758`).
- Other algorithms’ metrics stayed in line with prior runs; covertree build/search times remain the runtime driver (≈20 min per variant). Total wallclock ≈1h14m; benchmark summary at `benchmark_results/benchmark_20251204_152957/benchmark_summary.md`.
