# Repository Guidelines

## Purpose & Scope
This is a research repository for benchmarking the existing vector DB retrieval algorithms and developing vector retrieval algorithms with retrieval guarantees (e.g., formal/empirical recall bounds). Contributions should highlight the guarantee type, assumptions (distribution, metric, parameters), and how to reproduce the evidence (configs + commands).

## Project Structure & Module Organization
- `src/algorithms/`: Vector search implementations (e.g., `ExactSearch`, `HNSW`, FAISS wrappers).
- `src/benchmark/`: Dataset loading, metrics, and benchmark orchestration.
- `src/experiments/`: Config parsing and `ExperimentRunner` glue code.
- `scripts/`: Entry scripts for full suites and comparisons.
- `configs/`: YAML configs (e.g., `default.yaml`, `benchmark_config.yaml`).
- `/storage/ice-shared/cs8903onl/vectordb-retrieval/datasets/`: Downloaded/processed datasets (tracked via Git LFS for some files).
- `/storage/ice-shared/cs8903onl/vectordb-retrieval/results/`: Generated reports and logs.

## Build, Test, and Development Commands
- Create env and install deps: `pip install -r requirements.txt`.
- Run a single experiment: `python -m src.experiments.run_experiment --config configs/default.yaml --output-dir results`.
- Run full benchmark suite: `python scripts/run_full_benchmark.py --config configs/benchmark_config.yaml` (see `README.md` for quick `benchmark_config_test1.yaml`).
- Main entry alternative: `python main.py --config configs/default.yaml --verbose`.

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indentation, type hints required for public APIs.
- Docstrings: Google/NumPy style triple-quoted; explain args/returns.
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE`.
- Use `logging` (no print) and `tqdm` for progress when appropriate.
- Keep algorithms small and composable; prefer pure functions in `utils/`.

## Testing Guidelines
- A `pytest` framework is in place. To run the tests, execute `pytest` from the project root.
- Use smoke runs for quick validation:
  - Fast check: `python scripts/run_full_benchmark.py --config configs/benchmark_config_test1.yaml`.
  - Single run: `python -m src.experiments.run_experiment --config configs/default.yaml`.
- Reproducibility: keep `seed` in configs; avoid nondeterministic ops.
- Verify outputs: presence of `benchmark_results/.../benchmark_summary.md` and `all_results.json`.

## Commit & Pull Request Guidelines
- Follow Conventional Commits: `feat(scope): ...`, `docs: ...`, `chore: ...` (see `git log`).
- PRs must include: clear description, linked issue, sample config, brief results (attach summary table or log), and reproduction steps.
- Keep changes focused; update configs and README when behavior/CLI changes.

## Security & Configuration Tips

- Prefer `faiss-cpu` (default). Document any BLAS/OMP tweaks and hardware in PRs.
- Do not store secrets or API keys in YAML configs.

## PACE Cluster Usage (SLURM)
- Submit long runs with SLURM (`sbatch singlerun.sbatch`) on the PACE cluster; adjust `REPO_DIR` and `VENV_DIR` in the script to match your home paths.
- `singlerun.sbatch` ensures `uv` is available, provisions a virtualenv, installs `requirements.txt`, then launches `python scripts/run_full_benchmark.py --config configs/benchmark_config.yaml` under the new environment.
- Keep walltime/CPU settings in sync with project needs; request more resources via the `#SBATCH` lines and notify maintainers if configs require different quotas.
- Cluster-specific environment modules (e.g., GCC, OpenBLAS) should be loaded inside the script before dependency installation; document any additions in PR descriptions.
- For interactive debugging use `srun --pty bash` with matching module loads, then reuse `uv` environments instead of re-installing requirements each run.
- Monitor jobs with `squeue -u $USER` and collect logs from `Report-<jobid>.log` to attach to experiment summaries.

## Agent-Specific Instructions
- Treat this AGENTS.md as authoritative for the entire repo tree.
- Do not rename top-level dirs or public APIs without updating configs, scripts, and README.
- When adding algorithms, place them in `src/algorithms/`, document guarantees in class docstrings, and add a minimal config example under `configs/`.
- Prefer composing new retrieval variants by wiring `indexers` + `searchers` in YAML (see `configs/benchmark_config.yaml`) before adding new composite classes.
- Prefer `/home/hice1/pli396/miniconda3/envs/vectordb-retrieval/bin/python  scripts/run_full_benchmark.py --config configs/benchmark_config_test1.yaml` for runnable examples; keep changes surgical and reproducible.
