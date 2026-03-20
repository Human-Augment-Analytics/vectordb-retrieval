# Tradeoff Curves Implementation (2026-03-19)

## Executive Summary
Implemented per-algorithm tradeoff curves using config-driven grid sweeps and integrated them into benchmark outputs and `one-page-summary.md`.

The implementation now includes:
- explicit variant IDs (`p01`, `p02`, ...),
- explicit parameter mapping per variant,
- and a per-algorithm Pareto points table (non-dominated frontier) in `one-page-summary.md`.
- a GitHub-compatible table of contents (TOC) at the top of `one-page-summary.md`.
- automatic Mermaid quadrant charts (with `★` on Pareto points) in both summary and per-algorithm output files.
  - updated to parser-safe ASCII `*` markers due Mermaid lexical issues with Unicode `★` in some environments.

## What Changed And Why
- Added optional `tradeoff_curves` config block to define per-algorithm grid sweeps.
- Expanded algorithm variants deterministically (`<algo>__pXX`) from cartesian products.
- Enforced deterministic sweep cap (`max_points_per_algorithm`) to control runtime.
- Added per-result metadata:
  - `base_algorithm`
  - `tradeoff_variant_id`
  - `tradeoff_params`
  - `is_tradeoff_variant`
- Generated per-algorithm artifacts per dataset:
  - SVG curve: `tradeoff_<dataset>_<algorithm>_recall_qps.svg`
  - JSON data: `tradeoff_<dataset>_<algorithm>.json`
- Extended `one-page-summary.md` to include, for each algorithm:
  - full variant table (`Variant | QPS | Recall | Parameters`)
  - Pareto points table (`Pareto Points (Non-dominated Frontier)`)
- Added a top-level TOC in `one-page-summary.md` with dataset and tradeoff section links:
  - `#dataset-<name_slug>`
  - `#tradeoff-<name_slug>`
- Extended TOC depth to include dataset-level H3 sections:
  - `Algorithm Implementation Details`
  - `Dataset Details`
- Added Mermaid quadrant chart generation per algorithm:
  - embedded fenced `mermaid` block in `one-page-summary.md` with a legend (`* marks Pareto points`)
  - `.mmd` source file under `/<dataset>/tradeoff_curves/`
- Extended JSON payload with `pareto_frontier_points`.
- Extended JSON payload with `mermaid_quadrant_chart` and `mermaid_quadrant_chart_file`.

Why: prior output only showed a small number of variants and did not make efficient operating points obvious. The frontier table makes configuration selection faster by filtering to non-dominated tradeoff points.

## Files Updated
- `src/benchmark/runner.py`
- `src/benchmark/evaluation.py`
- `src/experiments/experiment_runner.py`
- `tests/test_tradeoff_curves.py`
- `tests/test_benchmark_runner_modular.py`
- `configs/benchmark_all_datasets_msm100k_covertree_reuse_lsh_tuned_tradeoff.yaml`

## Config Usage (10-Point Sweep)
In `configs/benchmark_all_datasets_msm100k_covertree_reuse_lsh_tuned_tradeoff.yaml`:
- `tradeoff_curves.enabled: true`
- `tradeoff_curves.max_points_per_algorithm: 10`
- 10-value grids configured for:
  - `hnsw.indexer.efSearch`
  - `ivf_flat.searcher.nprobe`
  - `ivf_pq.searcher.nprobe`
  - `ivf_sq8.searcher.nprobe`
  - `lsh.searcher.candidate_multiplier`

Algorithms without a configured grid still generate a baseline point (`tradeoff_variant_id=base`).

## Reproduce (PACE)
From `/home/hice1/pli396/PycharmProjects/vectordb-retrieval`:

```bash
sbatch slurm_jobs/codex_all_datasets_msm100k_reuse_ct_lsh_tuned_tradeoff.sbatch
```

Validation commands:

```bash
squeue -j <jobid> -o '%i %t %M %D %R'
sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode -n
```

## Latest Verified Run
- Job ID: `4502213`
- State: `COMPLETED`
- Elapsed: `00:23:13`
- Output: `/home/hice1/pli396/PycharmProjects/vectordb-retrieval/benchmark_results/benchmark_20260319_132416`
- Summary: `/home/hice1/pli396/PycharmProjects/vectordb-retrieval/benchmark_results/benchmark_20260319_132416/one-page-summary.md`

Observed in summary:
- `Table of Contents` section is present at the top and now includes deeper dataset H3 links.
- `Tradeoff Curves by Algorithm` section is present for each dataset.
- Variant parameter table includes explicit `pXX -> parameter values`.
- `Pareto Points (Non-dominated Frontier)` table is present per algorithm.
- `Mermaid Quadrant Chart` section is present per algorithm, with parser-safe `*` marking Pareto points.

Observed in JSON:
- `pareto_frontier_points` exists and is populated.
- `mermaid_quadrant_chart` and `mermaid_quadrant_chart_file` exist and are populated.
- Example: `msmarco/tradeoff_curves/tradeoff_msmarco_lsh.json` has 10 total points and 9 Pareto points.

## Test / Validation
PACE tests passed:

```bash
pytest -q tests/test_tradeoff_curves.py tests/test_benchmark_runner_modular.py
```

Result: `5 passed`.

## Follow-Ups
- Consider adding a cross-algorithm Pareto-comparison table per dataset (merge frontiers from all algorithms into one shortlist).
