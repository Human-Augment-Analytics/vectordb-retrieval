# Local CoverTree Profiling (cProfile)

Use these commands from the repository root to profile the local CoverTree smoke run.

## 1) Run smoke benchmark under cProfile

```powershell
.\.venv\Scripts\python.exe -m cProfile -o .\covertree_smoke.prof scripts/run_full_benchmark.py --config configs/covertree_smoke.yaml --data-dir data --output-dir benchmark_results
```

This writes raw profiler data to `covertree_smoke.prof`.

## 2) Print top CoverTree functions by cumulative and self time

```powershell
.\.venv\Scripts\python.exe -c "import pstats; s=pstats.Stats('covertree_smoke.prof'); s.strip_dirs().sort_stats('cumtime').print_stats('covertree|covertree_v2_2',40); s.sort_stats('tottime').print_stats('covertree|covertree_v2_2',40)"
```

- `cumtime`: total time spent in a function and its callees.
- `tottime`: time spent only inside the function body.

## 3) (Optional) Dump full sorted stats to text files

```powershell
.\.venv\Scripts\python.exe -c "import pstats; s=pstats.Stats('covertree_smoke.prof'); s.strip_dirs().sort_stats('cumtime').dump_stats('covertree_cum.prof')"
.\.venv\Scripts\python.exe -c "import pstats; s=pstats.Stats('covertree_smoke.prof'); s.strip_dirs().sort_stats('tottime').dump_stats('covertree_tot.prof')"
```

You can inspect those `.prof` files later with `pstats` or visualization tools.
