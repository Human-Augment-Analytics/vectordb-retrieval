# PACE SLURM Guide (PACE ICE Only)

This file is the PACE-only execution guide. Use it when `pwd` contains `hice1`.

## 1) Scope
- Applies only on **PACE ICE** (`pwd` contains `hice1`).
- Benchmark and heavy test workloads should run through SLURM scripts in `slurm_jobs/`.

## 2) Pick or Create the Benchmark Job
- Default full sweep config: `configs/benchmark_nomsma_covertree_v2_2.yaml`.
- Default script: `slurm_jobs/a_slurmjob_template.sbatch`.
- For custom runs, create a SLURM script from `slurm_jobs/a_slurmjob_template.sbatch` as `slurm_jobs/codex_<desc>.sbatch` and point to your config.

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

## 7) Persisted Covertree MSMARCO Runbook
- Use this flow when MSMARCO Covertree build time is too large for a single end-to-end job.
- The current design persists only `covertree_v2_2`; other algorithms still build in the active job.
- The current fixed shared artifact path is:
  - `/storage/ice-shared/cs8903onl/vectordb-retrieval/indexes/covertree_v2_2/msmarco_cosine_base50000_q200_gt200`
- The folder name is only a human label. Real compatibility is enforced by the persisted manifest, `dataset_fingerprint`, and `config_hash`.

### Build The Persisted MSMARCO Covertree Index
- Config:
  - `configs/benchmark_all_covertree_v2_2_build.yaml`
- Job script:
  - `slurm_jobs/codex_covertree_v2_2_msmarco_build.sbatch`
- Submit:
  - `sbatch slurm_jobs/codex_covertree_v2_2_msmarco_build.sbatch`
- Behavior:
  - runs `covertree_v2_2` on MSMARCO with `persistence.mode=build_only`
  - writes the persisted artifact to shared storage
  - skips retrieval/evaluation for that run
  - overwrites the target artifact when `force_rebuild: true`

### Retrieve Using The Persisted MSMARCO Covertree Index
- Config:
  - `configs/benchmark_all_covertree_v2_2_retrieve.yaml`
- Job script:
  - `slurm_jobs/codex_covertree_v2_2_msmarco_retrieve.sbatch`
- Submit:
  - `sbatch slurm_jobs/codex_covertree_v2_2_msmarco_retrieve.sbatch`
- Behavior:
  - runs `covertree_v2_2` on MSMARCO with `persistence.mode=retrieve_only`
  - loads the existing artifact from shared storage
  - hard-fails if the artifact is missing because `fail_if_missing: true`
  - reports `index_source=loaded` and carries `build_time_s` from the persisted manifest

### Run Full All-Algorithm Comparison While Reusing Covertree
- Config:
  - `configs/benchmark_all_datasets_msm100k_covertree_reuse.yaml`
- Job script:
  - `slurm_jobs/codex_all_datasets_msm100k_reuse_ct.sbatch`
- Submit:
  - `sbatch slurm_jobs/codex_all_datasets_msm100k_reuse_ct.sbatch`
- Behavior:
  - runs the full benchmark suite
  - loads the pre-persisted MSMARCO Covertree index
  - builds all non-Covertree algorithms in the active job
  - keeps MSMARCO at the same configured size used by the persisted Covertree artifact

### Optional Dependency Chaining
- Use SLURM dependency chaining if you want the retrieve job to start only after the build job succeeds.
- Example:
```bash
build_id=$(sbatch slurm_jobs/codex_covertree_v2_2_msmarco_build.sbatch | awk '{print $4}')
sbatch --dependency=afterok:${build_id} slurm_jobs/codex_covertree_v2_2_msmarco_retrieve.sbatch
```
- `afterok` means the second job runs only if the first job exits successfully. This avoids loading a partial or failed build.

### Validate The Persisted Artifact
- Expected files under the artifact directory:
  - `manifest.json`
  - `build_metrics.json`
  - `vectors.npy`
  - `tree_indices.npy`
  - `tree_levels.npy`
  - `tree_child_offsets.npy`
  - `tree_children.npy`
  - `WRITE_COMPLETE`
- If `WRITE_COMPLETE` is missing, treat the artifact as invalid and rebuild.
- If retrieval fails with fingerprint/config mismatch, rebuild the artifact with the current config instead of forcing reuse.

### Output Expectations
- Benchmark outputs land under `benchmark_results/<timestamp>/`.
- Standard outputs now include:
  - `benchmark_summary.md`
  - `one-page-summary.md`
  - `qps_recall_summary.md`
  - `all_results.json`
  - per-dataset result files
  - `qps_recall_<dataset>.svg`
- Use `one-page-summary.md` as the primary quick-read deliverable after a completed run.

## 8) Documentation Automation (Required)
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
