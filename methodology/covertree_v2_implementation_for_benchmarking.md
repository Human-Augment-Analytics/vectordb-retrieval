# CoverTreeV2 Implementation Notes

CoverTreeV2 extends the historical `feature/covertree` prototype—referred to here as **Convertree V0**—so it can run inside our benchmarking harness while guaranteeing perfect recall. The production-ready code lives in `src/algorithms/covertree_v2.py`.

---

## Version Map

- **Convertree V0 (`feature/covertree`)** – Original research prototype that mirrors the Beygelzimer cover-set traversal almost verbatim but only returns a single neighbor and lacks the BaseAlgorithm contract.
- **CoverTree (v1, `src/algorithms/covertree.py`)** – Benchmark-integrated rewrite that adds batch APIs, configurable latency knobs, and vector caching but trades away perfect recall by imposing candidate/visit limits.
- **CoverTreeV2 (`src/algorithms/covertree_v2.py`)** – Hybrid that preserves the breadth-first traversal fidelity from Convertree V0 while keeping the BaseAlgorithm integration, normalized vector store, and recall guarantees expected from benchmark baselines.

---

## Design Goals

1. **Faithful search order** – Reuse the breadth-first cover-set traversal from the prototype instead of the best-first heap that CoverTree v1 uses.  
2. **Perfect recall** – Never cap the traversal by a candidate pool; fall back to enumerating remaining vectors when the cover set prunes an entire branch.  
3. **Benchmark compatibility** – Implement the `BaseAlgorithm` interface (`build_index`, `search`, `batch_search`) and obey repo-wide metric handling (L2, cosine, inner product).

---

## Data Structures (`src/algorithms/covertree_v2.py:8-82`)

| Component | Details |
|-----------|---------|
| `_CoverTreeV2Node` | Stores the dataset index, level, cached vector view, and child references. Using vector views matches the original implementation while keeping memory overhead low. |
| `CoverTreeV2` | Tracks `root`, `max_level`, and the normalized vector cache (`self._working_vectors`). Configuration only records the metric to keep parity with the prototype. |

Vectors are converted to contiguous `float32` blocks and optionally normalized for cosine search (`_prepare_vectors`, `_normalize_vectors`, `_prepare_query`). Metric helpers mirror `CoverTree` so we can reuse the benchmarking configs.

---

## Index Construction (`src/algorithms/covertree_v2.py:84-154`)

Insertion follows Algorithm 2 from Beygelzimer et al.:

1. **Root bootstrapping** – First point forms the root at level 0.  
2. **Recursive insert** – `_insert` receives the current candidate queue (`Q_i`) and radius `2^level`. Children that satisfy the cover radius form `Q_{i-1}`. We recurse until either a parent is found or the queue is empty.  
3. **Root promotion** – If no parent exists at the current level, we promote the root: create a new node at `level + 1`, attach the previous root as its child, and restart from the new queue.

Nodes store the vector view and level so the search routine can recover the implied radius without recomputing it from scratch.

---

## Search Procedure (`src/algorithms/covertree_v2.py:156-243`)

`_cover_search` mirrors Algorithm 1:

1. Maintain the current cover set `Q_i` starting at the root level.  
2. Gather `Q_children` (all children of the nodes in `Q_i`) and compute their exact distances to the query.  
3. Record **every** child in the candidate list before filtering, which keeps the traversal close to breadth-first order and prevents us from dropping a point permanently when it fails the threshold test.  
4. Form the new cover set `Q_{i-1}` by keeping only the children whose distance is within `min_child_dist + 2^i`.  
5. Iterate until we run out of children or levels.

Because some branches may still remain unexplored (e.g., when the threshold becomes too tight at low levels), we append any unseen indices at the end (`len(candidate_indices) < total_points`). That final sweep guarantees perfect recall even when the cover-set pruning omits faraway leaves.

`_rank_candidates` then computes exact distances for the candidate indices and returns the top-k `(distances, indices)` pair expected by `ExperimentRunner`. `batch_search` simply loops over individual `search` calls to keep the implementation simple.

---

## Comparison with Convertree V0 (`feature/covertree`)

| Aspect | Convertree V0 | CoverTreeV2 | Benchmark impact |
|--------|----------------|-------------|------------------|
| **API contract** | Exposes `build_tree`, `insert`, and `search` that return the raw vector (not `(distances, indices)`), and leaves `batch_search` unimplemented. | Implements `build_index`, `search`, and `batch_search` that emit NumPy arrays compatible with `ExperimentRunner`. | V2 can be dropped into configs and SLURM jobs without adapter glue, whereas V0 cannot be benchmarked. |
| **Search outputs** | Algorithm 1 traversal returns only the single closest vector and never materializes a candidate list for top-k scoring. | Records every candidate discovered per layer, then re-ranks exactly to return top-k distances/indices. | V2 supports arbitrary `k` and provides perfect-recall leaderboard entries; V0 cannot demonstrate recall/QPS trade-offs. |
| **Data handling** | Stores full vectors inside each node, never normalizes inputs, and treats `metric` as a callable even when provided a string. | Keeps vectors in contiguous float32 caches, normalizes on demand, and reuses the same metric helpers CoverTree v1 already exposes. | V2 matches repository metric conventions, avoids redundant allocations, and scales to MS MARCO-sized datasets. |
| **Instrumentation** | Lacks `self.config`, logging hooks, and reproducible settings; tree parameters are implicit. | Captures metric + recall knobs in `self.config`, so benchmark JSON includes the exact search mode. | Runs become reproducible, and downstream analyses can cite the precise CoverTree variant used. |
| **Recall guarantees** | The traversal would be perfect if it produced k-NN outputs, but the implementation short-circuits after finding the first neighbor. | Emits the entire candidate frontier and appends unseen indices when necessary, ensuring perfect recall for any k. | V2 is suitable as the “correctness reference” baseline; V0 remains a historical artifact useful only for tracing the original derivation. |

In practice, CoverTreeV2 should be viewed as a direct successor to Convertree V0: it preserves the theoretical behavior but upgrades every touchpoint needed for benchmarking (API shape, metrics, configs, logging, and scalability).

---

## Comparison with CoverTree (v1)

| Aspect | CoverTree (`src/algorithms/covertree.py`) | CoverTreeV2 (`src/algorithms/covertree_v2.py`) |
|--------|-------------------------------------------|-----------------------------------------------|
| Traversal | Best-first heap with visit/pool budgets. | Level-synchronous cover sets with no hard budgets. |
| Recall guarantee | Depends on `candidate_pool_size` and `max_visit_nodes`. | Always hits perfect recall; fallback enumerates unseen indices. |
| Config knobs | Metric, `candidate_pool_size`, `max_visit_nodes`, `visit_multiplier`. | Metric only (prototype-style) because the traversal is deterministic. |
| Implementation roots | Production rewrite that prioritizes tunable latency and plugs into BaseAlgorithm. | Prototype-faithful port focused on correctness with the production glue preserved. |
| Intended use | Fast smoke benchmarks and recall/QPS exploration. | Reference-quality baseline for perfect-recall runs. |

Use CoverTreeV2 whenever the experiment requires correctness above all else; keep CoverTree (v1) for latency-sensitive sweeps.
