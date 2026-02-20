# Change Log - 2026-02-19

## Benchmark run: non-MSMARCO cover tree v2_2

- Requested benchmark config: `configs/benchmark_nomsma_covertree_v2_2.yaml`
- Environment: PACE ICE (detected from `pwd` path containing `hice1`)
- SLURM job script created from template:
  - `slurm_jobs/codex_nomsma_covertree_v2_2.sbatch`

## Commands executed

```bash
sbatch slurm_jobs/codex_nomsma_covertree_v2_2.sbatch
```

## Run attempts and operational fixes

1. First submission: job `4092699`
- Status: started and executed benchmark.
- Issue observed: benchmark output was sparse in SLURM logs during long-running experiment sections, making progress monitoring difficult.

2. Fix applied
- Updated `slurm_jobs/codex_nomsma_covertree_v2_2.sbatch` to run Python unbuffered:
  - from: `"$VENV_DIR/bin/python" scripts/run_full_benchmark.py ...`
  - to: `"$VENV_DIR/bin/python" -u scripts/run_full_benchmark.py ...`
- Canceled job `4092699` and repeated submission.

3. Second submission: job `4092706`
- Status: completed successfully.
- Log path:
  - `slurm_jobs/slurm_logs/Codex-NomSMA-CTv2_2-4092706-atl1-1-01-005-2-1.log`
- Output directory:
  - `benchmark_results/benchmark_20260219_105419`
 - Runtime: ~11m45s wall clock (`2026-02-19 10:53:55` to `2026-02-19 11:05:40` from SLURM log epilog)

## Current observed behavior

- Benchmark completed successfully for both datasets (`random`, `glove50`) with full artifact generation:
  - `benchmark_results/benchmark_20260219_105419/benchmark_summary.md`
  - `benchmark_results/benchmark_20260219_105419/all_results.json`
  - `benchmark_results/benchmark_20260219_105419/glove50/plots_20260219_110044/operations_vs_recall.png`
- Dataset load logs are visible in real time after `-u` fix.
- No runtime exceptions observed.

## Key metrics snapshot (recall, qps)

- random:
  - `covertree_v2_2`: recall `1.0000`, qps `34.91`, mean query `28.64 ms`, build `346.95 s`
  - `hnsw`: recall `0.9148`, qps `10063.94`
  - `ivf_sq8`: recall `0.5090`, qps `235263.33`
- glove50:
  - `covertree_v2_2`: recall `1.0000`, qps `38.25`, mean query `26.14 ms`, build `258.95 s`
  - `hnsw`: recall `0.9742`, qps `220480.87`
  - `faiss_lsh`: recall `0.9980`, qps `4373.27`

## Warnings observed

- Non-fatal plotting warning during glove operations-vs-recall plot:
  - `UserWarning: Attempt to set non-positive xlim on a log-scaled axis will be ignored.`
  - source location in log points to `src/benchmark/evaluation.py:248`
  - plot still generated successfully.

## Reproduce

```bash
sbatch slurm_jobs/codex_nomsma_covertree_v2_2.sbatch
squeue -j 4092706 -o '%i %t %M %D %R'
tail -f slurm_jobs/slurm_logs/Codex-NomSMA-CTv2_2-4092706-atl1-1-01-005-2-1.log
sacct -j 4092706 --format=JobID,State,Elapsed,CPUTime,MaxRSS
```
