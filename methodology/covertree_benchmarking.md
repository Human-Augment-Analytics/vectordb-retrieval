# CoverTree Benchmarking Methodology

This note captures the methodology we follow when benchmarking the CoverTree baseline in this repository. It covers (1) the algorithmic design we implemented, and (2) the end-to-end evaluation flow so future contributors can reproduce or extend the experiments.

---

## 1. Algorithm Overview

Cover Trees are hierarchical metric indexes that recursively build “covers” of the data at exponentially decreasing radii. Each node owns:

| Concept            | Role                                                                                                     |
|--------------------|----------------------------------------------------------------------------------------------------------|
| **Level**          | Determines the cover radius (typically `2^level`). Higher levels coarsely summarize the space.           |
| **Parent/Children**| Every node at level `i` is within radius `2^i` of its parent; children sit on level `i-1`.               |
| **Invariant**      | Any two nodes at the same level are at least `2^i` apart (packing), guaranteeing bounded branching.      |

### Algorithm Design in This Repo

1. **Node Layout**  
   Each `_CoverTreeNode` stores the original vector index, its level, and child references. Levels decrease by exactly one from parent to child, preserving the cover-degree invariants described in Beygelzimer et al. (2006). We keep indices instead of raw vectors so the structure stays lightweight and copies are avoided.

2. **Insert Routine**  
   - Begin at the current max level with the root in the candidate queue.
   - Compute the cover radius `2^level`. Children whose centers lie within that radius become candidates for the next recursive call.
   - If no suitable parent exists one level down, we try to attach the point to the current queue nodes. Failing that, the root is promoted: a new dummy node at level `max_level + 1` becomes the parent, preserving the cover property.
   - This mirrors Algorithm 2 (“Insert”) in the original Cover Tree paper, but we frame it in terms of vector indices to keep metadata compact.

3. **Search Routine**  
   - Maintain a priority queue keyed by `max(0, dist(q, node) - 2^level)`—a lower bound on any descendant’s distance. We lazily compute exact distances only when popping from the heap, caching results per node to avoid duplicate work.
   - Candidate pooling (`candidate_pool_size`) limits how many point IDs we track for the final scoring pass; `max_visit_nodes` bounds the number of heap pops so the algorithm can’t thrash on high-degree layers. **For the current investigation we temporarily disable both limits** (collect all points and visit all nodes) to rule out recall loss coming from aggressive pruning.
   - Once the heap is exhausted or the visit budget runs out, we score the collected candidate IDs exactly (`_compute_distances`) and return the top-k results. Cosine and inner-product metrics reuse the same logic with normalized vectors.

4. **Batch Search**  
   The benchmark runner expects `(distances, indices)` arrays per batch. Rather than vectorizing the recursive search, we invoke `search()` per query and pack the outputs. This keeps the code easier to reason about while still satisfying the `ExperimentRunner` API.

5. **Metric Handling & Normalization**  
   - L2: vectors stay unchanged.  
   - Cosine: training vectors and queries are normalized to unit length before insertion / lookup so `1 - dot(a, b)` acts as a distance.  
   - Inner product: we negate the dot product to reuse the same “distance minimization” pathway.  
   All conversions happen in `_prepare_vectors` / `_prepare_query` so inserts and searches share the same representation.

6. **Complexity Controls**  
   Cover Trees guarantee logarithmic search under bounded expansion constants, but real datasets (especially high-dimensional embeddings) can still cause large traversals. We therefore expose:
   - `candidate_pool_size`: max number of unique point IDs to re-score exactly.  
   - `max_visit_nodes`: cap on the number of nodes popped from the heap.  
   - `visit_multiplier`: used to derive a default `max_visit_nodes` when the user only sets the pool size.  
   These knobs mirror the “ef” controls seen in HNSW—turn them up for higher recall, down for faster smoke runs.  
   - **Temporary change:** we currently run with `candidate_limits_enabled: false`, meaning both pool size and visit caps are ignored so every node is considered. This is intentional to validate correctness; re-enable once recall investigations conclude.

### Implementation Highlights (`src/algorithms/covertree.py`)

| Aspect                      | How it’s handled                                                                 |
|-----------------------------|-----------------------------------------------------------------------------------|
| Construction                | Operates on contiguous `float32` arrays; stores indices only to save memory.     |
| Heap Ordering               | Uses admissible bounds (`distance - 2^level`) to prioritize promising regions.   |
| Candidate Scoring           | Vectorized `_compute_distances()` reuses metric-specific paths to rank quickly.  |
| Error Handling              | Raises `RuntimeError` if `search`/`batch_search` run before `build_index`.       |
| Config Reporting            | `self.config` records metric/pool settings so results JSON files are informative.|
| Testing                     | `tests/algorithms/test_covertree.py` compares results against brute force to guard regressions. |

---

## 2. Benchmarking Methodology

### Configuration

CoverTree is first-class in `configs/benchmark_config.yaml`, and the CoverTree/CovertreeV2 comparison lives in `configs/benchmark_nomsma_c_v2.yaml`:

| Dataset  | Metric | Parameters                                | Rationale                                    |
|----------|--------|-------------------------------------------|----------------------------------------------|
| random   | L2     | pools/visits disabled (full traversal)    | Eliminates pruning effects while debugging recall. |
| glove50  | L2     | pools/visits disabled (full traversal)    | Same rationale; keeps per-query latency measurable. |
| msmarco  | Cosine | pool 1024 / visits 16384                  | Handles higher dimensional Cohere embeddings.|

Run `python scripts/run_full_benchmark.py --config configs/benchmark_config.yaml` (or the SLURM wrapper) to evaluate CoverTree alongside FAISS, HNSW, LSH, etc.

### Evaluation Steps

1. **Launch**: `python scripts/run_full_benchmark.py --config configs/benchmark_config.yaml` or the matching SLURM script (e.g., `slurm_jobs/singlerun.sbatch`).  
2. **Datasets**:
   - Random: generated via `_generate_random_dataset` using the per-dataset options (dimensions, train/test sizes, seed).  
   - GloVe: loader reads `glove.6B.50d.txt` from `/storage/ice-shared/.../datasets/glove50/`, skipping downloads if the file exists.  
   - MS MARCO: uses the pre-embedded Cohere vectors already staged under `/storage/ice-shared/.../datasets/msmarco_pre_embeded/`.  
3. **Metrics Recorded**:
   - Build time, QPS, mean/median/min/max query latency, recall@k, precision@k, MAP@k, plus whichever recall bucket the summary table picks up (usually recall@min(100, topk)).  
   - Plots (`recall_vs_qps.png`) are generated per dataset.  
4. **Result Artifacts**:  
   - `<dataset>/covertree_results.json`: full metric dictionary per run.  
   - `<dataset>_results.json` and `_all_results.json`: cross-algorithm comparisons.  
   - `benchmark_summary.md`: Markdown report listing recall/QPS/build-time after the run completes.  

### Data / Quota Considerations

| Dataset | Path (read-only)                                                        | Notes                                                |
|---------|-------------------------------------------------------------------------|------------------------------------------------------|
| GloVe   | `/storage/ice-shared/cs8903onl/vectordb-retrieval/datasets/glove50/`    | `glove.6B.zip` + decompressed `.txt` files already available. |
| Random  | Generated; no persistent storage required.                              |                                                      |
| MS MARCO| `/storage/ice-shared/cs8903onl/vectordb-retrieval/datasets/msmarco_v1_embeddings/` | Uses memmap caches under `/results/cache`. |

Do **not** re-download shared datasets. If you need space in your home directory, delete old `benchmark_results/benchmark_<timestamp>/` folders inside your repo clone—leave `/storage/ice-shared/...` untouched so other users and jobs stay consistent.
