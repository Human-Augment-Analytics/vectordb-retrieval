# Benchmark Metrics Methodology

This note documents how the benchmark calculates every metric that appears in
`*_results.json` and `benchmark_summary.md`. Each section cites the exact code
paths so you can trace the implementation or extend it safely.

## Timing & Throughput

- **Build time (`build_time_s`)**  
  Measured by wrapping `algorithm.build_index(train_vectors)` with
  `time.time()` calls in `src/experiments/experiment_runner.py:161-164`. The
  reported value is `(time_after - time_before)` in seconds.

- **Index memory (`index_memory_mb`)**  
  Obtained via `_estimate_memory_usage()` in
  `src/experiments/experiment_runner.py:272-295`. If the algorithm exposes a
  `get_memory_usage()` helper, its return value (assumed bytes) is converted to
  megabytes. Otherwise the loader falls back to `train_vectors.nbytes / 2**20`,
  i.e. the raw data size.

- **Per-query timings (`query_times`)**  
  During search the runner either uses `batch_search` or falls back to single
  `search` calls (`src/experiments/experiment_runner.py:187-239`). Every timing
  is recorded **outside** FAISS/algorithm internals: the harness calls
  `time.time()` immediately before invoking `algorithm.batch_search()` (or
  `algorithm.search()`), then again right after the call returns. The difference
  is the elapsed wall-clock time for that call.

  - *Batced execution*: the total batch duration is divided evenly across
    queries in that batch (`per_query = elapsed / batch_size`) and written to
    `query_times[cursor:end]`. This keeps the array per-query even when the
    underlying search primitive operates on a matrix.
  - *Single-query fallback*: each `search()` call receives its own start/stop
    timing, and the raw duration is stored directly in `query_times[idx]`.

  No FAISS profiling APIs are involved; the harness uses Python-level timers, so
  results include everything FAISS (or any other backend) does internally plus
  the Python overhead of marshalling inputs.

- **Total query time (`total_query_time_s`)**  
  Tracks the sum of the elapsed times recorded above with a safety check to
  ensure it never underestimates the array sum
  (`max(total_query_time, query_times.sum())`, see
  `src/experiments/experiment_runner.py:241-248`).

- **Mean query time (`mean_query_time_ms`)**  
  Computed as `(total_query_time / n_queries) * 1000` in
  `src/experiments/experiment_runner.py:250-253`. Note the unit conversion to
  milliseconds.

- **Queries per second (`qps`)**  
  Defined as `n_queries / total_query_time` in
  `src/experiments/experiment_runner.py:255-256`. Because every query receives a
  per-query duration, the same value also appears in the evaluator’s summary
  (`1.0 / np.mean(query_times)` in `src/benchmark/evaluation.py:33-54`).


  - Exact vs HNSW: The “exact” engine brute-force scans all 1M vectors for every query, so each lookup costs a few milliseconds even in FAISS’ optimized BLAS. HNSW only touches a tiny fraction of nodes per search (roughly efSearch × log N probes), so once the graph is built it can return answers in
        sub-millisecond time. FAISS’ C++ implementation is tight, so seeing ~0.3ms/query translates to several thousand QPS.
  - Batch timing: When our harness calls algorithm.batch_search(), it times the whole batch and divides by the number of queries. That means any FAISS SIMD parallelism inside the batch effectively boosts QPS. We’re measuring end-to-end wall-clock time, not per-distance computations, so the speed
        differential reflects how little work HNSW has to do once the index is built.
  - Dataset mix: On the synthetic random dataset you’ll see even higher QPS because the vectors are tiny (128d) and the search is embarrassingly easy. On msmarco it drops to ~0.3ms/query, which is still realistic for a warmed-up FAISS graph at that scale.

- **Latency summary statistics**  
  When the evaluator runs, it stores mean/median/min/max query times (still in
  milliseconds) alongside QPS
  (`src/benchmark/evaluation.py:33-54`). Percentile calculations (`compute_cost_latency`)
  exist in `src/benchmark/metrics.py:110-137` but are not yet wired into the
  public reports.

## Retrieval Metrics

All retrieval metrics consume two integer matrices:
`ground_truth` (shape `(n_queries, ground_truth_k)`) and `predicted_indices`
(shape `(n_queries, topk)`).

- **Recall@k / Precision@k**  
  Implemented in `src/benchmark/metrics.py:5-46`. Both functions convert the
  relevant slices into Python sets per query, compute intersections, and return
  the mean across queries. Recall divides the hit count by the number of
  ground-truth entries (up to `k`), while precision divides by `k`.

- **MAP@10**  
  `mean_average_precision()` in `src/benchmark/metrics.py:48-86` iterates over
  each prediction rank, accumulates precision-at-hit positions, and averages by
  the number of relevant items. The evaluator only records `map@10` when at
  least 10 predictions are available (`src/benchmark/evaluation.py:32-41`).

- **Additional metrics (not surfaced in the summary)**  
  `src/benchmark/metrics.py` also defines NDCG@k, hit rate@k, and mean reciprocal
  rank. They follow the standard textbook formulas (see lines 88-169) and can be
  enabled by extending `Evaluator.evaluate`.

## Derived Fields in `*_results.json`

Every algorithm run stores a JSON blob assembled in
`src/experiments/experiment_runner.py:257-271`. Key fields include:

- Dataset information (`n_train`, `n_test`, `dimensions`, `topk`).
- Timing measurements (`build_time_s`, `total_query_time_s`,
  `mean_query_time_ms`).
- Throughput (`qps`).
- Memory footprint (`index_memory_mb`).
- Configuration snapshot (`parameters`) so downstream consumers can reproduce
  the run.

`Evaluator.evaluate()` merges these with the retrieval metrics before the
pipeline writes `*_results.json` and the aggregated `*_all_results.json`.

## Summary Table Generation

`scripts/run_full_benchmark.py` eventually calls `ExperimentRunner._write_summary`
which reads the merged results and formats the Markdown table you saw. The QPS
and timing figures originate from the metrics above; there is no additional
post-processing during summary generation.

## Potential Improvements

1. **Explicit warm-up phase** — prime caches and the FAISS/OMP thread pools
   before timing, then discard warm-up durations so QPS reflects steady-state
   performance.
2. **Expose percentile latencies** — `compute_cost_latency()` already computes
   P95/P99; wiring those into `Evaluator.evaluate()` would give a fuller latency
   profile.
3. **Separate batch timing from per-query averages** — right now batch elapsed
   time is divided evenly among queries. Recording both the raw batch timings
   and derived per-query mean would clarify how much batching influences QPS.
4. **Unify QPS reporting** — although `len(test)/total_time` equals
   `1/mean(query_time)` when per-query durations cover all queries, keeping only
   one definition (or checking they match) would prevent drift if the timing
   logic changes.
5. **Report variance / confidence intervals** — storing standard deviation of
   query times or bootstrapped confidence intervals would help interpret runs
  with a small number of queries.
