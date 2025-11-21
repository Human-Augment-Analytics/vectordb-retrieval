# LSH Benchmarking Status

*Last updated: 2025-11-21*

During the `benchmark_results/benchmark_20251108_095624/` run (config `configs/benchmark_config.yaml`), the LSH rows reported **perfect recall and precision identical to ExactSearch** on both random and GloVe50. Root cause: the candidate sets were empty under the old `(num_tables=12, hash_size=18, bucket_width=4, candidate_multiplier=8, fallback=true)` settings, so `LSHSearcher` always fell back to brute force.

### Fix applied

- Retuned the CoverTree benchmark config (`configs/benchmark_nomsma_c_v2.yaml`) to use wider buckets and smaller hashes: `hash_size=4`, `bucket_width=20.0`, `candidate_multiplier=64`, `fallback_to_bruteforce=false` for L2; cosine mirror uses `hash_size=12`, `candidate_multiplier=32`, also with fallback disabled.
- Quick local probes on the random dataset (first 50 queries; see shell snippets in this session) showed recall@10 rising from ~0 to ~0.22 with the new params (no fallback).
- Full SLURM rerun `3611844` (`benchmark_results/benchmark_20251121_151457/`) now shows approximate behaviour: random recall ≈0.32 (QPS ≈203) and glove50 recall ≈0.51 (QPS ≈94).

### Next Steps

- If we need higher recall without brute-force fallback, sweep `hash_size`, `bucket_width`, and `candidate_multiplier` systematically (consider a multiprobe variant instead of bumping `candidate_multiplier` further).
- Add lightweight logging around `_gather_candidates` / `_select_candidates` if candidate depletion resurfaces in other configs.
