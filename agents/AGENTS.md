# Repository Guidelines

## Purpose & Scope
This repository benchmarks vector retrieval algorithms and develops retrieval methods with formal or empirical guarantees. Every meaningful code change should preserve reproducibility via config + command + result artifacts.

## Environment Detection (PACE ICE)
Agents must auto-detect whether they are running on PACE ICE before selecting run strategy.

- Run `pwd`.
- If the returned path contains `hice1`, treat the environment as **PACE ICE cluster**.
- If `hice1` is absent, treat the environment as local/non-PACE.

Execution rule:
- On PACE ICE: follow `agents/agent_pace.md` as the execution playbook for benchmarking/testing.
- Off PACE: run local validation commands directly unless the user asks for SLURM-specific execution.

## Project Structure & Module Organization
- `src/algorithms/`: Vector search implementations.
- `src/benchmark/`: Dataset loading, metrics, benchmark orchestration.
- `src/experiments/`: Config parsing and experiment runner glue.
- `scripts/`: Entry scripts for full suites and utilities.
- `configs/`: YAML experiment/benchmark configs.
- `slurm_jobs/`: PACE batch scripts.
- `methodology/`: Technical notes, performance docs, change logs, and follow-ups.

## Build, Test, and Development Commands
- Environments are provisioned inside SLURM scripts (`uv` venv + `requirements.txt`).
- Standard benchmark entrypoint: `python scripts/run_full_benchmark.py --config <config.yaml>`.
- Preferred cover-tree-v2_2 benchmark config: `configs/benchmark_nomsma_covertree_v2_2.yaml`.

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indentation, type hints on public APIs.
- Naming: `snake_case` (modules/functions), `PascalCase` (classes), `UPPER_SNAKE` (constants).
- Use `logging` instead of `print` for runtime information.

## Testing Guidelines
- Use `pytest` for unit/integration checks.
- On PACE, wrap tests in a SLURM script when required by scope or runtime.
- Preserve reproducibility: keep config seeds explicit and avoid hidden nondeterminism.

## PACE Handoff
- Keep PACE-specific operational steps in `agents/agent_pace.md`.
- If PACE ICE is detected (`pwd` includes `hice1`), consult and follow `agents/agent_pace.md` before running jobs.

## Documentation Automation (Required)
Agents must proactively create/update documentation in `methodology/` whenever changes are meaningful.

Create or update docs when any of the following occurs:
- algorithm optimization or behavioral change,
- new algorithm/indexer/searcher added,
- benchmark/performance investigation with actionable findings,
- metric/regression analysis or large config strategy changes.

Minimum documentation content:
- what changed and why,
- files/configs/scripts touched,
- how to reproduce (exact command/config),
- key results/metrics or observed behavior,
- open risks/follow-ups.

File conventions:
- Use focused filenames, e.g. `methodology/<topic>_<YYYYMMDD>.md`, or update an existing canonical file if one already exists.
- Record run summaries in `methodology/change_log_<date>.md` when benchmarks are executed.
- Update `methodology/known_followups.md` when unresolved issues are discovered or resolved.

## Commit & PR Guidelines
- Use Conventional Commits (e.g., `feat(scope): ...`, `fix: ...`, `docs: ...`, `chore: ...`).
- Keep PRs focused and include reproduction details for benchmark-impacting changes.
- Never commit secrets or credentials.

## Agent-Specific Instructions
- Treat this file as authoritative for the repo.
- Do not rename top-level dirs or public APIs without updating dependent configs/scripts/docs.
- Keep `README.md`, `agents/AGENTS.md`, and `agents/agent_pace.md` aligned after process changes.
- Prefer SLURM helpers on PACE and preserve logs/artifacts for traceability.
