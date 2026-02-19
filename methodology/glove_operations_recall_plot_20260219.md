# Glove Operations-vs-Recall Plot (2026-02-19)

## What changed and why

- Added a new benchmark visualization for Glove datasets to show the trade-off between retrieval quality and query cost:
  - `operations_vs_recall.png`
- This complements the existing `recall_vs_qps.png` plot by presenting the same trade-off from a cost/operations perspective.

## Files touched

- `src/benchmark/evaluation.py`
  - Added `plot_operations_vs_recall(...)`.
  - Added operations-metric resolution logic with fallback priority:
    1. explicit operations metrics (if present),
    2. timing metrics (`mean_query_time_ms` / `mean_query_time` / `total_query_time_s`),
    3. derived per-query cost from `qps`.
- `src/experiments/experiment_runner.py`
  - Updated `_generate_plots(...)` to auto-generate `operations_vs_recall.png` when dataset name contains `glove`.
- `tests/test_operations_recall_plot.py`
  - Added tests for operations-vs-recall plot output and Glove-only plot generation behavior.

## How to reproduce

1. Run a benchmark config that includes `glove50`:

```bash
python3 scripts/run_full_benchmark.py --config configs/benchmark_nomsma_covertree_v2_2.yaml
```

2. Inspect the Glove dataset plot directory under the new benchmark run:

```bash
ls benchmark_results/benchmark_<timestamp>/glove50/plots_<experiment_id>/
```

Expected files include:
- `recall_vs_qps.png`
- `operations_vs_recall.png`

## Observed behavior / validation notes

- Static syntax validation succeeded with:

```bash
python3 -m py_compile src/benchmark/evaluation.py src/experiments/experiment_runner.py tests/test_operations_recall_plot.py
```

- Full pytest validation could not be executed in this environment because:
  - available `pytest` is `5.0.1` while `pytest.ini` requires `>=7.0`,
  - the local `python3` environment is missing runtime dependencies (for example `numpy`).

## Open risks / follow-ups

- Current runs do not persist a canonical explicit operation-count metric across all algorithms; the new plot therefore uses a robust fallback chain (often mean query time).
- If true operation counters are later added in algorithm outputs, the plot will prefer them automatically without further pipeline changes.
