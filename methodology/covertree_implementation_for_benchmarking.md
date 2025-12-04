# CoverTree Implementation Comparison for Benchmarking

This note contrasts the CoverTree algorithm that currently lives on this branch (`src/algorithms/covertree.py`) with the historical implementation on `origin/feature/covertree` and explains why the benchmarked version fails to achieve perfect recall. File references that start with `a/` point to the `feature/covertree` snapshot, while bare paths refer to this branch.

---

## Implementation Differences

| Aspect | Current branch (`src/algorithms/covertree.py`) | `feature/covertree` (`a/src/algorithms/covertree.py`) | Impact |
|--------|-----------------------------------------------|-------------------------------------------------------|--------|
| **Node payload / invariants** | Stores `_CoverTreeNode(index, level, children)` so the tree keeps only vector IDs and each node knows the radius implied by its level (`src/algorithms/covertree.py:13-24`). | `Node` keeps the entire vector (`value`) and a `children` list, but it lacks explicit level metadata and reuses raw vectors at every level (`a/src/algorithms/covertree.py:13-21`). | Current branch avoids copying vectors and can reason about bounds via level, but it also drops the “self on every level” invariant—the original code mirrors Beygelzimer’s pseudocode more closely even though it is incomplete. |
| **Vector storage & metrics** | Normalizes vectors up front, caches contiguous arrays, and supports `l2`, cosine, and inner-product metrics plus custom metrics via `_metric_fn` (`src/algorithms/covertree.py:66-151`, `272-308`). | Accepts a `metric` constructor arg but treats it like a callable even when passed a string (`a/src/algorithms/covertree.py:21-31`) and never normalizes data. | Current branch integrates cleanly with the benchmarking stack and avoids shape bugs; the feature branch breaks whenever configs pass `"l2"` instead of `np.linalg.norm`. |
| **Index construction** | `_insert_index` and `_insert` operate on integer IDs, maintain `max_level`, and reuse the same recursive structure as Algorithm 2 while keeping the tree compact (`src/algorithms/covertree.py:153-197`). | `build_index` and `_insert` follow the same textbook flow but copy full vectors at every node and never emit metadata or config (`a/src/algorithms/covertree.py:36-99`). | Current branch scales to millions of vectors because only indices live in the tree; the feature branch inflates memory and cannot be serialized for benchmarking. |
| **Search strategy** | Uses a best-first priority queue keyed by `max(0, dist(q,node) - 2^level)` and collects a bounded candidate pool before doing an exact re-ranking (`src/algorithms/covertree.py:202-270`). | Implements Algorithm 1 verbatim: breadth-first covers with `Q_i` sets and only returns the single nearest vector (`a/src/algorithms/covertree.py:101-150`). | The PQ approach gives tunable latency but can miss neighbors when the candidate pool or visit budget is exhausted; the BFS version would achieve perfect recall if it exposed k-NN outputs and distances. |
| **API surface / BaseAlgorithm contract** | Implements `build_index`, `search(k) -> (dist, idx)`, and `batch_search` that returns `(n_queries, k)` arrays for ExperimentRunner (`src/algorithms/covertree.py:91-125`). | `batch_search` is a stub (`pass`) and `search` returns a raw vector or `None`, so it cannot satisfy the benchmarking interface (`a/src/algorithms/covertree.py:33-34`, `101-150`). | Only the current branch can participate in automated benchmarks without adapter glue. |
| **Performance controls** | Exposes `candidate_pool_size`, `max_visit_nodes`, and `visit_multiplier`, tracking them in `self.config` so metrics JSON captures the search budget (`src/algorithms/covertree.py:32-61`, `202-245`). | No knobs exist beyond whatever implicit work `_insert` performs, so every query traverses the entire cover chain. | Current branch can be tuned for smoke tests, but the defaults trade recall for speed. |
| **Testing** | Dedicated pytest module compares results against brute force for both `search` and `batch_search` (`tests/algorithms/test_covertree.py`). | Embeds `unittest` cases inside the algorithm file and exercises only single-query search (`a/src/algorithms/covertree.py:167-261`). | Head’s tests integrate with CI; the feature branch mixes algorithm + tests and cannot be imported cleanly. |

---

## Why the Benchmark Version Misses Perfect Recall

1. **Candidate pool saturation** – `_collect_candidates` stops expanding the tree once `len(candidates) == max_candidates` (`src/algorithms/covertree.py:202-244`). With the default `candidate_pool_size=256` and `pool_size = max(k, candidate_pool_size)` (`src/algorithms/covertree.py:98-103`), any dataset that requires looking at more than 256 points will see recall < 1.0. This matches the benchmark symptom: the search hits the pool limit before exploring the remainder of the tree.

2. **Visit budget truncation** – Even if we raise the pool size, the maximum number of heap pops is `visit_budget = min(max_visit_nodes, n_vectors)` where `max_visit_nodes` defaults to `candidate_pool_size * visit_multiplier` (`src/algorithms/covertree.py:44-48`, `222-226`). If a dataset needs >4 096 node visits (256 × visit_multiplier 16) to reach all true neighbors, the traversal will exit early and recall will again plateau.

3. **Best-first vs. cover-layer search** – The PQ ranks nodes by `dist(q, node) - 2^level`, which is admissible but not identical to the breadth-first cover sets from the paper. Once a node is popped it never re-enters the queue; only its children are pushed. That is fine for correctness because we append the popped node’s index to the candidate array (`src/algorithms/covertree.py:231-239`), but it means the tree is explored along a single narrow frontier. If the frontier ignores an entire sibling cover, no amount of final exact scoring can recover the missed nodes. The original implementation iterates cover sets in lockstep (`a/src/algorithms/covertree.py:101-150`), guaranteeing that all points within admissible thresholds are eventually discovered (albeit with higher latency).

4. **“Self as child” concern** – The feature branch replicates nodes across levels only when a root promotion happens (`a/src/algorithms/covertree.py:50-69`); ordinary points still appear exactly once. Our version makes the same simplification but compensates by recording the node’s own index as soon as it is popped from the PQ, so a node does not disappear after being visited. Missing recall therefore stems from the budgets above, not from candidates being dropped outright.

In short, the benchmark branch trades perfect recall for bounded latency. Setting `candidate_pool_size >= n_database` and `max_visit_nodes >= n_database` restores brute-force equivalence (and perfect recall) but at the cost of scanning the entire tree.

---

## Which Implementation Serves Benchmarking Better?

- **Current branch advantages**  
  - Fully compatible with `ExperimentRunner` (k-NN outputs, batch API, metric metadata).  
  - Tunable knobs let us dial accuracy vs. runtime, which is essential for integrating with SLURM smoke runs.  
  - Memory footprint scales with the number of vectors because nodes store indices, not entire embeddings.

- **`feature/covertree` advantages**  
  - Adheres more closely to the cover-set traversal from Beygelzimer et al., so—once completed—it would naturally deliver perfect recall by visiting every admissible node.  
  - Easier to reason about correctness because it follows the textbook pseudocode without heuristic budgets.

- **Recommendation for “perfect/good recall” benchmarks**  
  1. Keep the current branch as the production implementation because it already satisfies the repository contracts and benchmarking infrastructure.  
  2. Raise `candidate_pool_size` and `max_visit_nodes` to at least the dataset size when perfect recall is required, or make them dataset-driven via configs so the smoke profile differs from the accuracy profile.  
  3. Incorporate the level-synchronous filtering step from the feature branch as an optional “strict” search mode (visit all nodes whose `dist <= min_child + 2^level`) so we can recover the theoretical guarantees without losing the BaseAlgorithm features.  
  4. If we port portions of the feature branch, fix its API mismatches first (`batch_search`, returning `(distances, indices)`, and honoring metric strings) so it can be benchmarked.

With those adjustments, the current implementation can hit perfect recall while remaining fit for benchmarking, whereas the `feature/covertree` version would require significant rewrites before it could even be evaluated at scale.
