# Cover Tree V2.2 Implementation & Optimizations

## Overview

`CoverTreeV2_2` is an optimized implementation of the Cover Tree algorithm, designed to be a drop-in replacement for `CoverTreeV2`. It addresses performance bottlenecks while restoring algorithmic correctness for exact k-Nearest Neighbor search, and includes subsequent hot-loop optimizations in `_search_exact_k` (2026-02-13).

## Key Changes & Optimizations

### 1. Exact k-NN Search (Restoring 100% Recall)

The previous `CoverTreeV2` implementation relied on a heuristic traversal followed by a "brute-force fallback" to ensure recall. This was inefficient.

`CoverTreeV2_2` implements a **standard exact k-NN search** algorithm for Cover Trees (adapted from Beygelzimer et al., 2006).

*   **Mechanism:** It maintains a max-heap of the best $k$ candidates found so far during traversal.
*   **Dynamic Pruning:** At each step, it prunes a branch (subtree) *only* if the minimum possible distance to any point in that subtree exceeds the distance to the current $k$-th nearest neighbor.
    *   **Pruning Condition:** $d(query, node) - radius(node) > k\_th\_dist$
    *   Where $radius(node) = 2^{level+1}$ (conservative bound).
*   **Heap-Update Gating (2026-02-13):** Once the heap already contains $k$ points, children with $d(query, child) \ge k\_th\_dist$ are skipped for heap insertion. This reduces Python heap operations without changing correctness.
*   **Bound Refresh:** The pruning bound is refreshed after candidate updates, so subtree filtering uses the tightest known $k$-th distance at each level.
*   **Guarantee:** This logic guarantees finding the true $k$ nearest neighbors (100% recall) without ever falling back to a full scan.

### 2. Vectorized Distance Calculations

To improve performance in Python, all distance calculations during traversal are batched.

*   **Implementation:** Instead of iterating through children and computing distances one by one, child vectors are stacked into a NumPy array and distances are computed in a single vectorized operation.
*   **Frontier Collection:** Child-frontier expansion uses a nested list-comprehension rather than repeated `extend` calls, reducing Python loop overhead in the traversal hot path.
*   **Vectorized Pruning Filter (2026-02-13):**
    *   Fast path: when a frontier has uniform child levels, use one scalar radius and evaluate `dist <= pruning_bound + 2^(level+1)` in one vectorized mask.
    *   Fallback: for mixed levels, compute a vector of per-child radii and apply the same bound elementwise.
*   **Benefit:** Reduces interpreter overhead and uses optimized NumPy kernels for distance and mask operations.

### 3. Merged Search and Ranking

In `CoverTreeV2`, the search phase returned a list of candidates, and a separate ranking phase re-calculated distances to sort them.

*   **Optimization:** `CoverTreeV2_2` calculates distances once during the tree traversal. These distances are stored directly in the results heap.
*   **Benefit:** Halves the number of distance computations for visited nodes.

## Trade-offs

*   **Memory:** The recursive tree structure (`_CoverTreeV2Node`) is still pointer-based, which is less cache-efficient than a flat array layout. Vectorization involves creating temporary batch arrays, which increases transient memory usage slightly.
*   **Complexity:** The search logic is more complex than a simple greedy traversal because it combines exactness guarantees, dynamic bounds, and vectorized frontier filtering.
*   **Performance Sensitivity:** The 2026-02-13 pruning fast path is most beneficial when frontiers are large and child levels are uniform; mixed-level frontiers use a vectorized fallback with slightly more overhead.

## Usage

`CoverTreeV2_2` is registered in the algorithm registry and can be used in benchmarks by setting the algorithm type to `"CoverTreeV2_2"`.

## Performance Results (Benchmark 2025-12-04)

Testing against `random` and `glove50` datasets confirms significant improvements over previous versions while maintaining 100% recall.

| Algorithm | Dataset | Recall | QPS | Mean Query Time (ms) | Build Time (s) |
|-----------|---------|--------|-----|----------------------|----------------|
| CoverTree (V1) | random | 1.0 | 7.85 | 127.32 | 1198.45 |
| CoverTreeV2 | random | 1.0 | 11.66 | 85.74 | 1205.80 |
| **CoverTreeV2_2** | **random** | **1.0** | **30.44** | **32.85** | **350.22** |
| CoverTree (V1) | glove50 | 1.0 | 7.82 | 127.81 | 870.54 |
| CoverTreeV2 | glove50 | 1.0 | 11.50 | 86.98 | 870.47 |
| **CoverTreeV2_2** | **glove50** | **1.0** | **31.14** | **32.11** | **256.67** |

**Key Observations:**
1.  **Speedup:** V2.2 achieves roughly **2.6x - 2.7x higher QPS** compared to V2, and **~3.9x** compared to V1.
2.  **Build Time:** Construction is drastically faster (~3.4x speedup vs V2), likely due to vectorization in the insertion path.
3.  **Recall:** 100% recall is preserved, validating the exact search logic.

The 2026-02-13 hot-loop updates were validated with:

```bash
pytest -q tests/algorithms/test_covertree_v2_2.py
```

At that time, targeted correctness tests passed (`2 passed`) with no API or output-shape changes.
