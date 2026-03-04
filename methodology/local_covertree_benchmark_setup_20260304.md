# Local Covertree Benchmark Setup (20260304)

## What Changed and Why

- Added CLI overrides to the benchmark entrypoint so configs authored for PACE (`/storage/...`) can be executed locally without editing YAML.
- Updated benchmark docs to use the current cover-tree smoke config filename and to document local path overrides explicitly.

This was done to unblock local benchmark runs, especially the CoverTreeV2_2 smoke benchmark.

## Files Touched

- `scripts/run_full_benchmark.py`
- `src/benchmark/runner.py`
- `README.md`

## Reproduction

Run the covertree smoke benchmark locally:

```bash
python scripts/run_full_benchmark.py --config configs/covertree_v2_2_smoke.yaml --data-dir data --output-dir benchmark_results
```

Run the larger no-MSMARCO benchmark set locally:

```bash
python scripts/run_full_benchmark.py --config configs/benchmark_nomsma_covertree_v2_2.yaml --data-dir data --output-dir benchmark_results
```

## Expected Behavior

- Datasets resolve under `data/<dataset_name>/`.
- Outputs/logs are written under `benchmark_results/benchmark_<timestamp>/`.
- Existing PACE workflows remain available by omitting these CLI overrides.

## Risks / Follow-ups

- This change does not auto-remap dataset-specific absolute paths such as MSMARCO `embedded_dataset_dir`; those still need valid local paths if MSMARCO is included in a run.
- No full benchmark was executed in this change; validate with a local smoke run before relying on throughput numbers.
