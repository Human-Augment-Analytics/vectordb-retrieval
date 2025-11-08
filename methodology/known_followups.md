# Known Follow-ups

Use this checklist to pick up the outstanding benchmarking/debugging work before starting a new session. Update the notes whenever you make progress so the next agent can continue seamlessly.

---

## 1. LSH reporting perfect recall

- **Symptom:** `benchmark_results/benchmark_20251108_095624/benchmark_summary.md` shows `lsh` rows identical to `exact` (recall/precision/QPS), which is unrealistic for the current configuration (8 tables, 18-bit hashes, candidate multiplier 8).
- **Hypotheses:**
  - `LSHSearcher` might always fall back to brute force when its candidate set is empty, effectively cloning ExactSearch.
  - Metrics wiring could be pulling the wrong indices when aggregating results.
  - Dataset overrides may silently replace the LSH index/search pair with brute-force components.
- **Reproduce:** `python scripts/run_full_benchmark.py --config configs/benchmark_config.yaml` (or `sbatch slurm_jobs/singlerun_complete_benchmarking_pat.sbatch`).
- **Action items:**
  1. Inspect `benchmark_results/.../random/lsh_results.json` and `.../glove50/lsh_results.json` to confirm returned indices match the exact baseline.
  2. Instrument `src/algorithms/modular.py` / `src/algorithms/lsh.py` to log when the fallback path executes.
  3. Adjust LSH parameters or fallback handling so the algorithm exhibits meaningful approximate behaviour.
  4. Document findings and fixes in `methodology/lsh_benchmarking.md`.

---

## 2. Random dataset too small for FAISS IVF/PQ

- **Symptom:** `slurm_logs/VectorDB-Retrieval-Guarantee_FULL-3513016-atl1-1-03-004-5-1.log` (and later runs) spam:

  ```
  WARNING clustering 5000 points to 256 centroids: please provide at least 9984 training points
  ```

  because we now limit the random dataset to 5k training vectors but still request `nlist=256` for IVF/PQ.

- **Options:**
  1. Regenerate a larger random dataset (e.g., 20k train vectors) specifically for FAISS algorithms.
  2. Reduce the IVF/PQ `nlist` settings when running the downsized dataset.

- **Action items:** Decide which path is more appropriate, implement it, and document the change (config + README + AGENTS). Until then, expect FAISS warnings on every run.

---

## 3. CoverTree QPS sanity check

- **Symptom:** After integrating CoverTree, the reported QPS jumped to very high values (e.g., `benchmark_results/benchmark_20251108_095624/random/covertree_results.json` reports ~49 QPS even though each query spends ~20 ms). The user has flagged this as suspicious.

- **Potential causes:**
  - `ExperimentRunner` may mis-handle `total_query_time` if we mix batch and single-query paths.
  - `repeat` settings or averaging at the evaluator layer might be double-counting queries.
  - Our CoverTree implementation might be returning instantly because the candidate pool is too small (causing low latency but poor recall).

- **Action items:**
  1. Add temporary logging in `ExperimentRunner._run_single_algorithm` to dump `total_query_time`, `query_times.sum()`, and `used_batch_api` for CoverTree.
  2. Verify that `query_times` reflect reality (time the `search()` call with `time.perf_counter()`).
  3. If the numbers are legitimate, document why (e.g., 256 queries * 20 ms ≈ 5 s ⇒ 49 QPS); if not, fix the timing logic.

- **Note:** The smoke run and the latest benchmark both show ~20 ms/query, so the QPS figure is mathematically consistent; confirm whether the concern is about realism or instrumentation.

---

Keep this file updated whenever you start/complete work on any item above or add new follow-ups.
