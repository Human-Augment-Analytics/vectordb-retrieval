# Change Log — 2025-11-21

Summary of today’s updates across code, configs, documentation, and benchmarks.

## Code
- `src/algorithms/covertree.py`: added `candidate_limits_enabled` flag (default `False`); when disabled, candidate pooling and visit caps are bypassed so searches traverse all nodes. Search helper now respects the flag for pool size and visit budget.

## Configuration & Scripts
- `configs/benchmark_nomsma_c_v2.yaml`:
  - Added CoverTree (v1) alongside CoverTreeV2.
  - Set `candidate_limits_enabled: false` for CoverTree entries.
  - Retuned LSH to avoid brute-force fallback masking empty candidate sets: `hash_size=4`, `bucket_width=20.0`, `candidate_multiplier=64`, `fallback_to_bruteforce=false` (L2); cosine mirror uses `hash_size=12`, `candidate_multiplier=32`, fallback disabled.
- `slurm_jobs/singlerun_nomsma_benchmarking_c_v2_pat.sbatch`: reduced resources to `--cpus-per-task=8`, `--mem-per-cpu=4G` to ease scheduling while running the c_v2 config.

## Documentation
- `README.md`: noted CoverTree limits are temporarily disabled; clarified the c_v2 SLURM job now runs both CoverTree and CoverTreeV2.
- `AGENTS.md`: added guidance that CoverTree v1 is currently run with limits disabled.
- `methodology/covertree_benchmarking.md`: documented the temporary full-traversal mode and referenced the CoverTree/CovertreeV2 comparison config.
- `methodology/lsh_benchmarking.md`: recorded the LSH fallback issue and the new tuning.
- `methodology/known_followups.md`: marked the LSH “perfect recall” issue as addressed for current configs; noted CoverTree full-traversal QPS context.

## Benchmarks Run (configs/benchmark_nomsma_c_v2.yaml)
- Job `3611237` → `benchmark_results/benchmark_20251121_123917` (CoverTree limits disabled, LSH fallback still on): LSH showed perfect recall; cover trees perfect recall with QPS ~8–12.
- Job `3611497` → `benchmark_results/benchmark_20251121_135718` (fallback off, old LSH params): LSH recall collapsed to ~0, confirming empty candidate sets without fallback.
- Job `3611844` → `benchmark_results/benchmark_20251121_151457` (fallback off, retuned LSH): approximate behaviour restored. Key numbers: random recall ≈0.32 (QPS ≈203) / glove50 recall ≈0.51 (QPS ≈94); CoverTree/CovertreeV2 recall 1.0 with QPS ~9/12 and build ~18–20 minutes. Logs at `slurm_jobs/slurm_logs/VectorDB-Retrieval-Guarantee_FULL-3611844-atl1-1-01-005-3-1.log`.

## Open Follow-ups
- If higher LSH recall is needed without fallback, sweep `hash_size`, `bucket_width`, `candidate_multiplier`, or add a multiprobe-like expansion.
- Re-enable CoverTree candidate/visit limits after recall investigations; consider adding instrumentation around timing if limits return.***
