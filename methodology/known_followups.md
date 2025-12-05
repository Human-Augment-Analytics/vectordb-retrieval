# Known Follow-ups

Use this checklist to pick up the outstanding benchmarking/debugging work before starting a new session. Update the notes whenever you make progress so the next agent can continue seamlessly.

---

## 1. LSH reporting perfect recall (addressed for current configs)

- **Original symptom:** `benchmark_results/benchmark_20251108_095624/benchmark_summary.md` showed `lsh` rows identical to `exact` (recall/precision/QPS). Root cause was the fallback-to-bruteforce path firing because candidate sets were empty under the old `(num_tables=12, hash_size=18, bucket_width=4, fallback=true)` setup.
- **Fix applied:** Retuned `configs/benchmark_nomsma_c_v2.yaml` to widen buckets and shrink hash_size, and disabled fallback. Current params: `hash_size=4`, `bucket_width=20.0`, `candidate_multiplier=64`, `fallback_to_bruteforce=false` (cosine searcher mirrors the fallback-off stance with `candidate_multiplier=32`). See the sampling notebook in this session (`python` snippets in shell) for quick recall probes on the random dataset.
- **Result check:** SLURM job `3611844` (`benchmark_results/benchmark_20251121_151457/`) now reports approximate behaviour: random recall ≈0.32 (QPS ≈203) and glove50 recall ≈0.51 (QPS ≈94). Earlier rerun `3611497` confirmed the fallback-path hypothesis by producing ~0 recall once fallback was disabled without retuning.
- **Next steps (if revisiting):** If we need better recall/QPS trade-offs, consider tuning `num_tables/hash_size/bucket_width` systematically or adding a “multi-probe” style expansion instead of brute-force fallback. No instrumentation added yet.

---

## 2. Random dataset size vs. FAISS IVF/PQ (resolved)

- **Status:** Fixed in `configs/benchmark_config*.yaml` / `configs/covertree_smoke.yaml` on 2025‑11‑13 by bumping `train_size` to 20 000 (test size unchanged) and deleting `/storage/ice-shared/cs8903onl/vectordb-retrieval/datasets/random` so the cache regenerates. Subsequent `slurm_jobs/singlerun_complete_benchmarking_pat.sbatch` run (job `3532085`) no longer emits the FAISS warning.
- **Follow-up:** Nothing pending—just remember to wipe the cached `random` dataset if you tweak `train_size` again so future jobs pick up the change.

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

- **Note:** We temporarily disabled CoverTree candidate/visit caps (`candidate_limits_enabled=false`) and reran (`benchmark_results/benchmark_20251121_151457/` via job `3611844`). QPS now drops to ~9–12 with perfect recall (full traversal) and long build times (~18–20 min index build recorded as ~110–113 ms/query search). Leave the logging steps above if we re-enable pruning; current numbers look internally consistent.

## 4. Benchmark summary shows identical index memory across algorithms

- **Symptom:** In `benchmark_results/benchmark_20251121_151457/benchmark_summary.md`, every algorithm reports the same index memory (e.g., 4.88 MB for random, 3.81 MB for glove50), which is implausible given differing index structures.
- **Fix:** Updated `ExperimentRunner._estimate_memory_usage` to introspect FAISS indices (including PQ/SQ code sizes and HNSW link overhead), LSH artifacts, and tree structures before falling back to raw train-vector size. Added debug logging (file handler only) for the computed components.
- **Status:** Fixed. Smoke run `3681003` (`configs/benchmark_config_smoke.yaml`) and full job `3681007` (`configs/benchmark_nomsma_c_v2.yaml`) both show differentiated footprints: e.g., `benchmark_results/benchmark_20251128_173147/benchmark_summary.md` lists covertree 5.49 MB vs. HNSW 7.32 MB vs. IVF_PQ 1.22 MB (random), and covertree 4.43 MB vs. HNSW 6.26 MB vs. IVF_PQ 0.95 MB (glove50). The estimator now inspects FAISS code sizes, HNSW links, LSH buffers, and tree nodes before falling back to raw vectors.
- **Follow-up:** If future FAISS runs still look too uniform, consider adding centroid memory for IVF and enabling DEBUG logs to inspect the per-algorithm breakdowns.

---

Keep this file updated whenever you start/complete work on any item above or add new follow-ups.

## 5. Inflated QPS metrics on random dataset (20251128_154708)

- **Symptom:** `benchmark_results/benchmark_20251128_154708/random/random_results.json` reports extreme QPS values (e.g., `ivf_sq8` ≈162k, `ivf_flat` ≈104k, `hnsw` ≈11k) that are out of line with realistic throughput.
- **What the logs show:** The run’s `benchmark.log` spans 15:47:08 → 16:29:24 because the CoverTree variants each spent ~1 200 s building (`covertree`/`covertree_v2` `build_time_s` ≈1207 s) and ~20–30 s searching, while the FAISS/HNSW algorithms finished in milliseconds. The inflated QPS is purely from the per-query timing path, not from overall wall clock.
- **Root cause in metric calculation:** `Evaluator.evaluate` sets `qps = 1 / np.mean(query_times)` using the per-query latencies emitted by `ExperimentRunner`’s batch path. With only 256 queries and sub-millisecond batch timings from FAISS, the mean latency drops to single-digit microseconds, so `1 / mean` explodes even though end-to-end runtime is dominated by slow builds that the QPS metric ignores.
- **Next steps:** Recompute QPS from total search wall time (`len(test_queries) / total_query_time`) using `time.perf_counter()` for better resolution, and consider a “throughput including build” metric for fairness. Also bump the query count for sanity checks so per-query timing isn’t dominated by timer noise on tiny batches.

---

## 6. CoverTree full-suite run timed out on MSMARCO (20251202)

- **Symptom:** `Codex-Covertree-All` job (`3778745`) using `configs/benchmark_all_covertree.yaml` produced `benchmark_results/benchmark_20251202_182606/` with only random and glove50 outputs; MSMARCO results are missing.
- **Log evidence:** `slurm_jobs/slurm_logs/Codex-Covertree-All-3778745-atl1-1-02-010-1-2.log` shows the run reached MSMARCO at 19:37:41 and was cancelled at 06:25:53 due to the 12-hour walltime limit (`slurmstepd: ... CANCELLED ... DUE TO TIME LIMIT`). A numpy overflow warning appeared just before cancellation but no stack trace was logged.
- **Follow-up:** Re-run MSMARCO (or the full config) with a longer walltime or narrower algorithm set (e.g., drop CoverTree variants) to keep within limits. Capture the new slurm log and benchmark summary once complete.

---

## 7. FAISS IndexLSH recall under baseline configs (resolved)

- **Symptom:** In `benchmark_results/benchmark_20251204_091614/benchmark_summary.md`, `faiss_lsh` shows markedly lower recall (≈0.21 on `random`, ≈0.38 on `glove50`) despite extremely high reported QPS, making it a weak baseline compared to both HNSW/IVF and the custom Python LSH.
- **Root cause:** The `FaissLSHIndexer` built a plain `faiss.IndexLSH` and `FaissSearcher` delegated directly to `index.search`, so results were ranked purely by the LSH backend with no candidate expansion or re-scoring against the true metric (L2/cosine). This kept latency tiny but sacrificed recall.
- **Fix:** `FaissLSHIndexer` now tags its artifacts with `faiss_index_kind='lsh'`, and `FaissSearcher` detects this and performs an optional rerank pass: it asks the FAISS index for an expanded candidate set (`lsh_candidate_multiplier`, default 8×; set to 64× in `configs/benchmark_nomsma_c_v2.yaml` via `faiss_l2.lsh_candidate_multiplier`), then re-scores those candidates against the original vectors using the configured metric before returning the top‑k. Non-LSH FAISS indexes still use the original fast path.
- **Expected behaviour:** Future runs of `configs/benchmark_nomsma_c_v2.yaml` (e.g., via `slurm_jobs/singlerun_complete_benchmarking_pat.sbatch`) should report substantially improved `faiss_lsh` recall at modest additional query cost, with index memory still reflecting the compact LSH codes. Tune `lsh_candidate_multiplier` in the searcher config to trade QPS for recall if needed.

---

## 8. CoverTree Optimization Next Steps (Post-V2.2)

- **Status:** `CoverTreeV2_2` (Python + NumPy vectorization) improved QPS by ~3x and build time by ~3.5x over V2, reaching ~30 QPS on random/glove50. However, this is still orders of magnitude slower than FAISS HNSW (~3500-8000 QPS).
- **Next Steps for Optimization:**
    1.  **JIT Compilation (Numba):** The current overhead is dominated by Python interpreter steps during tree traversal. Decorating the distance/traversal logic with `@numba.jit` could yield C-like performance without rewriting in C++.
    2.  **Cython/C++ Extension:** Moving the `_CoverTreeV2Node` structure and traversal logic entirely to C++ (wrapped via Cython or PyBind11) would eliminate pointer-chasing overhead and Python object creation costs.
    3.  **Memory Layout:** Convert the pointer-based tree to a flat array (CSR-like or similar) to improve cache locality.
    4.  **Approximate variants:** Implement epsilon-approximate search or limit the traversal depth/nodes visited to trade recall for QPS, as V2.2 currently enforces strict exact search.