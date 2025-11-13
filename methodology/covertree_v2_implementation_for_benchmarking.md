# CoverTreeV2 Implementation Notes

CoverTreeV2 extends the historical `feature/covertree` prototype so it can run inside our benchmarking harness while guaranteeing perfect recall. It lives in `src/algorithms/covertree_v2.py`.

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

## Comparison with CoverTree (v1)

| Aspect | CoverTree (`src/algorithms/covertree.py`) | CoverTreeV2 (`src/algorithms/covertree_v2.py`) |
|--------|-------------------------------------------|-----------------------------------------------|
| Traversal | Best-first heap with visit/pool budgets. | Level-synchronous cover sets with no hard budgets. |
| Recall guarantee | Depends on `candidate_pool_size` and `max_visit_nodes`. | Always hits perfect recall; fallback enumerates unseen indices. |
| Config knobs | Metric, `candidate_pool_size`, `max_visit_nodes`, `visit_multiplier`. | Metric only (prototype-style). |
| Intended use | Fast smoke benchmarks and recall/QPS exploration. | Reference-quality baseline for perfect-recall runs. |

Use CoverTreeV2 whenever the experiment requires correctness above all else; keep CoverTree (v1) for latency-sensitive sweeps.
