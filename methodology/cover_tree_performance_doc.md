# Cover Tree V2 Performance Analysis

Based on the benchmark results from `benchmark_results/benchmark_20251128_173147`, the `covertree_v2` implementation is significantly slower than other algorithms like `hnsw` and `ivf_flat`. This document outlines the primary reasons for this performance difference.

## 1. Lack of Low-Level Optimization

The most significant factor contributing to the slow performance is that `covertree_v2` is implemented in pure Python and NumPy. In contrast, the other algorithms in the benchmark (e.g., `hnsw`, `ivf_flat`, `ivf_pq`) are wrappers around the **FAISS** library.

FAISS is written in C++ and is highly optimized for vector search, utilizing:
- **SIMD (Single Instruction, Multiple Data):** To compute multiple distances simultaneously on the CPU.
- **BLAS libraries:** Optimized linear algebra subroutines.
- **Multi-threading:** To parallelize search operations.

The Python implementation of `covertree_v2` cannot take advantage of these low-level optimizations, leading to a substantial performance gap.

## 2. Iterative Distance Calculations in Python

The `covertree_v2` code performs distance calculations within Python loops, which introduces significant overhead. This happens in two key places:

- **Index Building (`_insert`):** When inserting a new vector, the algorithm traverses the tree and computes distances to existing nodes one by one to find the correct placement.
- **Searching (`_cover_search`):** During a search, the algorithm again traverses the tree, and for each level, it computes the distance from the query vector to each child node in a Python list comprehension:
  ```python
  child_distances = [self._distance_to_query(query, child.index) for child in Q_children]
  ```
This iterative approach is inherently slower than the vectorized and parallelized distance computations in FAISS.

## 3. Expensive Candidate Ranking

The search in `covertree_v2` is a two-stage process:
1.  **Candidate Selection (`_cover_search`):** Identify a subset of points from the dataset that are potential nearest neighbors.
2.  **Ranking (`_rank_candidates`):** Compute the exact distance to all candidate points and return the top-k results.

If the initial `_cover_search` stage is not effective at pruning the search space, it can return a large number of candidate points. In the worst case, it could return all points in the dataset. The `_rank_candidates` function then performs a brute-force distance calculation on all these candidates. This is computationally expensive and, when combined with the overhead of Python, leads to high query times.

## Conclusion

The `covertree_v2` implementation serves as a correct, but not performant, version of the cover tree algorithm. The observed 100x+ slowdown is expected when comparing a pure Python implementation to a production-ready, highly optimized C++ library like FAISS. The primary bottlenecks are the lack of low-level optimizations, the use of Python loops for iterative distance calculations, and a potentially expensive final ranking step.

## Line-by-Line Profiling Workflow (Local)

To move from high-level suspicion to concrete bottlenecks, use `line_profiler` with:

- Script: `scripts/profile_covertree_lines.py`
- Target module: `src/algorithms/covertree.py` (`CoverTreeV2_2`)

### 1) Install dependency

```powershell
.\.venv\Scripts\python.exe -m pip install line_profiler
```

### 2) Run line-by-line profiling (smoke-like settings)

```powershell
.\.venv\Scripts\python.exe scripts/profile_covertree_lines.py --train-size 20000 --query-size 256 --dimension 50 --metric cosine --k 200 --profile-phase both --search-runs 1 --output covertree_line_profile.txt --dump-raw covertree_line_profile.lprof
```

### 3) Focus on likely hotspots in `src/algorithms/covertree.py`

- `build_index` insertion loop
- `_insert` operations that repeatedly call `np.stack(...)`
- `_search_exact_k` traversal and candidate filtering
- `_compute_distance_batch_to_1` distance kernel
- `batch_search` query loop (calls `search` per query)

Line-profiler output reports `% Time` per line so you can directly see which exact statements dominate runtime.
