# Repository Guidelines

## Purpose & Scope
This is a research repository for benchmarking the existing vector DB retrieval algorithms and developing vector retrieval algorithms with retrieval guarantees (e.g., formal/empirical recall bounds). Contributions should highlight the guarantee type, assumptions (distribution, metric, parameters), and how to reproduce the evidence (configs + commands).

## Project Structure & Module Organization
- `src/algorithms/`: Vector search implementations (e.g., `ExactSearch`, `HNSW`, `LSH`, FAISS wrappers).
- `src/benchmark/`: Dataset loading, metrics, and benchmark orchestration.
- `src/experiments/`: Config parsing and `ExperimentRunner` glue code.
- `scripts/`: Entry scripts for full suites and comparisons.
- `configs/`: YAML configs (e.g., `default.yaml`, `benchmark_config.yaml`).
- `/storage/ice-shared/cs8903onl/vectordb-retrieval/datasets/`: Downloaded/processed datasets (tracked via Git LFS for some files).
- `/storage/ice-shared/cs8903onl/vectordb-retrieval/results/`: Generated reports and logs.

## Build, Test, and Development Commands
- Provision environments inside the SLURM scripts. The default helpers bootstrap a `uv` virtualenv and install `requirements.txt` before execution.
- Submit full benchmark runs with `sbatch slurm_jobs/singlerun_complete_benchmarking_pat.sbatch`. Adjust the script if configs, resources, or artefact paths change.
- For smoke or dataset-specific checks (e.g., verifying new MSMARCO embeddings), edit `slurm_jobs/singlerun_smoke.sbatch` to match the scenario and submit it via `sbatch`.
- When a bespoke experiment is required, clone/tweak the closest script under `slurm_jobs/`, document the invocation, and ensure the job captures logs under `slurm_logs/` or `Report-<jobid>.log`.

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indentation, type hints required for public APIs.
- Docstrings: Google/NumPy style triple-quoted; explain args/returns.
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE`.
- Use `logging` (no print) and `tqdm` for progress when appropriate.
- Keep algorithms small and composable; prefer pure functions in `utils/`.

## Testing Guidelines
- A `pytest` framework is in place. Package it into an appropriate SLURM job when test coverage is required (e.g., clone `slurm_jobs/singlerun_smoke.sbatch` and swap in `pytest`).
- Use SLURM smoke runs for quick validation:
  - Submit the modified `slurm_jobs/singlerun_smoke.sbatch` for lightweight checks (e.g., configs such as `benchmark_config_test1.yaml`).
  - For targeted experiments, clone the smoke script, adjust the command (e.g., `python -m src.experiments.run_experiment --config configs/default.yaml`), and submit with `sbatch`.
- Reproducibility: keep `seed` in configs; avoid nondeterministic ops.
- Verify outputs: presence of `benchmark_results/.../benchmark_summary.md` and `all_results.json`.

## Commit & Pull Request Guidelines
- Follow Conventional Commits: `feat(scope): ...`, `docs: ...`, `chore: ...` (see `git log`), follwing by multiline commit message with detailed changes description, what files updated and why
- PRs must include: clear description, linked issue, sample config, brief results (attach summary table or log), and reproduction steps.
- Keep changes focused; update configs and README when behavior/CLI changes.

## Security & Configuration Tips

- Prefer `faiss-cpu` (default). Document any BLAS/OMP tweaks and hardware in PRs.
- Do not store secrets or API keys in YAML configs.
- For memory-constrained runs, keep `use_memmap_cache: true` inside `dataset_options` (especially for MS MARCO) so `passage_embeddings.npy` is opened through a read-only memmap. Pair this with `query_batch_size` (global or per dataset) to bound concurrent search work and keep SLURM jobs within walltime/memory limits.

## PACE Cluster Usage (SLURM)
- Submit long runs with SLURM (`sbatch singlerun.sbatch`) on the PACE cluster; 
- `singlerun.sbatch` launches `python scripts/run_full_benchmark.py --config configs/benchmark_config.yaml` so the generated MS MARCO embedding bundle (`/storage/ice-shared/cs8903onl/vectordb-retrieval/msmarco_v1_embeddings`) is consumed; increase `#SBATCH -t` if you refresh the subset/embeddings so the new artefacts can be written before reuse.
- Keep walltime/CPU settings in sync with project needs; request more resources via the `#SBATCH` lines and notify maintainers if configs require different quotas.
- Cluster-specific environment modules (e.g., GCC, OpenBLAS) should be loaded inside the script before dependency installation; document any additions in PR descriptions.
- For interactive debugging use `srun --pty bash` with matching module loads, then reuse `uv` environments instead of re-installing requirements each run.
- Monitor jobs with `squeue -u $USER` and collect logs from `Report-<jobid>.log` to attach to experiment summaries.

## Agent-Specific Instructions
- Treat this AGENTS.md as authoritative for the entire repo tree.
- Do not rename top-level dirs or public APIs without updating configs, scripts, and README.
- When adding algorithms, place them in `src/algorithms/`, document guarantees in class docstrings, and add a minimal config example under `configs/`.
- Prefer composing new retrieval variants by wiring `indexers` + `searchers` in YAML (see `configs/benchmark_config.yaml`) before adding new composite classes.
- Agents execute directly on PACE; run benchmarks through the SLURM helpers in `slurm_jobs/`: submit `singlerun_complete_benchmarking_pat.sbatch` for full-suite runs and update/submit `singlerun_smoke.sbatch` for smoke checks (e.g., validating fresh MSMARCO embeddings). Pick or tailor the SLURM script that fits each request before dispatching it.
- Monitor SLURM job logs (e.g., `Report-<jobid>.log`, `slurm_logs/`) and debug issues in place. You are expected to access any project paths involved; if access is missing, ask for guidance and explain how to grant it.
- Name any newly created SLURM helpers as `slurm_jobs/codex_<descriptive_name>.sbatch` (or `.sh`) so automation can discover agent-authored scripts.
- When the user requests a commit, describe the code changes and propose the commit message for approval before running `git commit`.
- MSMARCO subsampling/embedding lives in `src/dataprep/`; keep `configs/ms_marco_subset_embed.yaml`, README.md, and this file in sync whenever the sampling parameters, output locations, or artefact layout change.
- keep readme.md and agents.mdup-to-date with new features and changes.

## Known Follow-ups (keep in sync as you investigate)

- **LSH metrics look identical to ExactSearch** (see `benchmark_results/benchmark_20251108_095624/benchmark_summary.md`). This is almost certainly a bug: reproduce the run with `configs/benchmark_config.yaml`, inspect the LSH pipeline, and document findings in `methodology/lsh_benchmarking.md` (created for this purpose). Fix the implementation/config once the root cause is known.
- **Random dataset triggers FAISS k-means warnings** when building IVF/PQ indexes because we only keep 5k training vectors. Decide whether to regenerate a larger random dataset (>= nlist*40) or downsize the IVF/PQ configs; track the discussion in this file once you pick a direction. The warning is emitted repeatedly in `slurm_logs/VectorDB-Retrieval-Guarantee_FULL-3513016-atl1-1-03-004-5-1.log`.
