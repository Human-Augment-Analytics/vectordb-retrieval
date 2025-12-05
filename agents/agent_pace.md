# PACE SLURM Quickstart (for agents)

Practical checklist for running and validating benchmarks on PACE via SLURM.

1) Pick or create the config/script
- Default full sweep: `configs/benchmark_nomsma_c_v2.yaml` via `slurm_jobs/singlerun_nomsma_benchmarking_c_v2_pat.sbatch`. always check if the config yaml file needs to be updated first.
- Create a new  config yaml and slurm_job.sbatch if needed  or being asked.
- Smoke or bespoke: clone the closest script under `slurm_jobs/` (name it `codex_<desc>.sbatch`) and point to your config/command.

2) Submit the job
- From repo root: `sbatch slurm_jobs/singlerun_nomsma_benchmarking_c_v2_pat.sbatch`
- Resources baked in: `--cpus-per-task=24`, `--mem-per-cpu=8G`, walltime `15:00:00`, partition `coc-cpu`, QOS `coc-ice`., adjust the number of cpus (maximum 24) and mem-per-cpu (maximum 16G) if necessary.
- Scripts auto-create a `uv` venv at `$HOME/scratch/vector-db-venv` and install `requirements.txt` before running.

3) Monitor progress
- Queue: `squeue -j <jobid> -o '%i %t %M %D %R'`
- Resource stats: `sstat -j <jobid>.batch --format=AveCPU,AveRSS,JobID`
- Live log: `tail -f slurm_jobs/slurm_logs/<jobname>-<jobid>-<node>.log`

4) Collect results
- Outputs land under `benchmark_results/<timestamp>/` with `benchmark_summary.md`, `all_results.json`, per-dataset result JSONs, and plots.
- Slurm log remains in `slurm_jobs/slurm_logs/`; reference it in change logs.

5) Debug/re-run
- If a job times out or fails, adjust `#SBATCH -t`, CPU/memory, or prune algorithms in the sbatch script; resubmit.
- For MSMARCO or other large runs, consider narrower algorithm sets or longer walltime to stay under limits.

6) Environment tips
- Shared datasets live at `/storage/ice-shared/cs8903onl/vectordb-retrieval/datasets`; don’t re-download.
- Reuse the `uv` env between runs to save setup time (`source $HOME/scratch/vector-db-venv/bin/activate`).

7) Reporting
- Record each run’s job id, config, and key outcomes (recall/QPS deltas, issues) in `methodology/change_log_<date>.md`.
- Update `methodology/known_followups.md` if new issues surface.
