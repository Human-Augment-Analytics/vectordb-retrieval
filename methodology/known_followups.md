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

---

Keep this file updated whenever you start/complete work on any item above or add new follow-ups.
