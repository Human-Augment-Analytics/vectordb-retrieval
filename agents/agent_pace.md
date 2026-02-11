# PACE SLURM Guide (PACE ICE Only)

This file is the PACE-only execution guide. Use it when `pwd` contains `hice1`.

## 1) Scope
- Applies only on **PACE ICE** (`pwd` contains `hice1`).
- Benchmark and heavy test workloads should run through SLURM scripts in `slurm_jobs/`.

## 2) Pick or Create the Benchmark Job
- Default full sweep config: `configs/benchmark_nomsma_covertree_v2_2.yaml`.
- Default script: `slurm_jobs/singlerun_nomsma_benchmarking_c_v2_pat.sbatch`.
- For custom runs, clone the closest SLURM script as `slurm_jobs/codex_<desc>.sbatch` and point to your config.

## 3) Submit Job
- From repo root:
  - `sbatch slurm_jobs/singlerun_nomsma_benchmarking_c_v2_pat.sbatch`
- Resource tuning guidance:
  - up to `--cpus-per-task=24`
  - up to `--mem-per-cpu=16G`
  - increase `#SBATCH -t` for heavy runs
- Scripts bootstrap `uv` env at `$HOME/scratch/vector-db-venv` and install `requirements.txt`.

## 4) Monitor Job
- Queue: `squeue -j <jobid> -o '%i %t %M %D %R'`
- Resource stats: `sstat -j <jobid>.batch --format=AveCPU,AveRSS,JobID`
- Live log: `tail -f slurm_jobs/slurm_logs/<jobname>-<jobid>-<node>.log`

## 5) Collect Outputs
- Expect outputs in `benchmark_results/<timestamp>/`:
  - `benchmark_summary.md`
  - `all_results.json`
  - per-dataset result JSONs
  - plots
- Keep SLURM log references for documentation and PR notes.

## 6) Debug and Re-run
- On timeout/failure: tune `#SBATCH -t`, CPU/memory, or narrow algorithms.
- For MSMARCO or similarly heavy datasets, prefer narrower suites or longer walltime.

## 7) Documentation Automation (Required)
Whenever work is meaningful, create/update docs under `methodology/`.

Trigger examples:
- algorithm optimization/change,
- new algorithm/indexer/searcher,
- performance benchmark investigation,
- nontrivial config strategy updates.

Documentation must include:
- what changed and why,
- files/configs/scripts changed,
- exact reproduce commands/configs,
- key metrics/findings,
- follow-up actions.

Also:
- log benchmark runs in `methodology/change_log_<date>.md`,
- update `methodology/known_followups.md` for open/resolved issues.
