# MSMARCO Subset Methodology

This document describes the reproducible pipeline used in this repository to
derive a manageable MSMARCO passage-ranking benchmark with sentence-transformer
embeddings and explicit ground-truth neighbors. It mirrors the implementation
as of 2025-02-14.

## Source Data

- **Corpus**: `msmarco-passage` (v1) accessed through `ir_datasets`
  (`https://ir-datasets.com/msmarco-passage.html`).
- **Relevance judgments**: `msmarco-passage/dev` qrels, also provided via
  `ir_datasets`.
- **Shared cache**: PACE deployments mount the raw download at
  `/storage/ice-shared/cs8903onl/vectordb-retrieval/ms_marco_v1_raw`.
  The scripts respect `IR_DATASETS_HOME`, defaulting to this location when the
  environment variable is unset.

## Subsampling Pipeline

Implementation: `src/dataprep/subsample_msmarco.py` (lines 15–193).

1. **Configuration**  
   Loads `configs/ms_marco_subset_embed.yaml`. Key parameters:

   - `subset.SEED` controls the reproducible RNG seed (default `42`).
   - `subset.CORPUS_SAMPLE_SIZE` and `subset.QUERY_SAMPLE_SIZE` dictate the
     number of passages (default `1_000_000`) and dev queries (default `1_000`)
     retained.
   - `subset.OUTPUT_DIR` specifies the TSV destination
     (defaults to `/storage/ice-shared/.../msmarco_v1_subsampled` on PACE).

2. **Passage sampling**  
   - Calls `ir_datasets.load("msmarco-passage")`.
   - Draws `CORPUS_SAMPLE_SIZE` unique indices with Python’s RNG seeded to
     `subset.SEED`.
   - Streams the entire corpus once, writing the selected passages to
     `corpus.tsv` (`doc_id`, `text`).

3. **Query sampling**  
   - Calls `ir_datasets.load("msmarco-passage/dev")`.
   - Uniformly samples `QUERY_SAMPLE_SIZE` dev queries with the same seed.
   - Streams and writes the chosen queries to `queries.tsv` (`query_id`, `text`).

Both TSVs include a header row and are saved alongside progress logging. The
script reports errors early if the requested sample exceeds the available data
or if the environment lacks the MSMARCO download.

## Embedding Pipeline

Implementation: `src/dataprep/embed_msmarco.py`.

1. **Configuration**  
   Uses `configs/ms_marco_subset_embed.yaml` to locate the TSV snapshot
   (`embeddings.INPUT_DIR`) and to choose:

   - `embeddings.MODEL_NAME`: default `all-MiniLM-L6-v2`.
   - `embeddings.BATCH_SIZE`: default `256`.
   - `embeddings.OUTPUT_DIR`: default
     `/storage/ice-shared/.../msmarco_v1_embeddings`.

2. **Loading TSVs**  
   Reads the subsampled `corpus.tsv` and `queries.tsv`, collecting parallel
   lists of IDs and text. Any malformed rows are skipped with warnings.

3. **Query retention**  
   All sampled queries are preserved. The previous qrels-alignment step (which
   dropped queries lacking positives in the subsampled corpus) has been
   removed, so `query_embeddings.npy` now covers every row in `queries.tsv`.

4. **Embedding generation**

   - Picks `cuda` when available, otherwise `cpu`.
   - Loads `SentenceTransformer(MODEL_NAME)` once.
   - Encodes all passages and queries with the configured batch size.
   - Persists the float32 arrays to disk.

5. **Outputs**

   The following files are written atomically to `embeddings.OUTPUT_DIR`:

   | File | Contents |
   |------|----------|
   | `passage_embeddings.npy` | `(num_passages, d)` float32 embeddings |
   | `passage_ids.npy` | `num_passages` array of string IDs (parallel to embeddings) |
   | `query_embeddings.npy` | `(num_queries, d)` float32 embeddings |
   | `query_ids.npy` | `num_queries` array of string IDs |
   | `metadata.json` | Seed, counts, embedding dim, model name, and `ground_truth_precomputed=false` |

   Existing files are overwritten. Dimensional consistency between IDs and
   embeddings is verified before saving.

## Benchmark Ground-Truth Construction

Implementation: `Dataset._process_msmarco_preembedded` in
`src/benchmark/dataset.py`.

- When `use_preembedded: true`, the loader reads the saved numpy bundles,
  normalises them as needed, and builds an **exact** ground-truth matrix using
  FAISS (`IndexFlatIP` for cosine similarity, `IndexFlatL2` for Euclidean).
- The target width is taken from `dataset_options.ground_truth_k` (default
  `100`). If fewer neighbours exist than requested, the effective width is
  reduced automatically.
- Cosine similarity defaults to raw inner-product scoring (matching the
  current `ExactSearch` implementation). Set
  `dataset_options.normalize_cosine_groundtruth: true` if you want the loader
  to L2-normalise vectors before running the brute-force search.
- The resulting indices are cached under
  `/storage/ice-shared/cs8903onl/vectordb-retrieval/results/cache/` with the
  pattern `msmarco_<hash>_top{K}groundtruth.npy`, making the cache key explicit
  about the ground-truth depth.
- Subsequent runs reuse the cache through the standard dataset memoisation
  path; the hash suffix includes `_ground_truth_method=bruteforce_v1` so legacy
  qrels-based caches are bypassed automatically.

## Benchmark Integration

To consume the generated bundle, point the MSMARCO entry in a benchmark config
at the embedding directory, e.g.:

```yaml
datasets:
  - name: msmarco
    metric: cosine
    dataset_options:
      use_preembedded: true
      embedded_dataset_dir: /storage/ice-shared/cs8903onl/vectordb-retrieval/datasets/msmarco_v1_embeddings
      ground_truth_k: 200
      use_memmap_cache: true
```

`Dataset._process_msmarco_preembedded()` (see
`src/benchmark/dataset.py`) now recomputes the ground truth from embeddings on
load, caches the `top{K}groundtruth` file, and memory-maps the passages when
`use_memmap_cache: true`.

## Reproduction Checklist

1. Ensure `ir_datasets` is installed and `IR_DATASETS_HOME` points at the raw
   MSMARCO cache (or allow the scripts to set the default).
2. `python src/dataprep/subsample_msmarco.py`
3. `python src/dataprep/embed_msmarco.py`
4. Execute your preferred benchmark config (e.g.,
   `python scripts/run_full_benchmark.py --config configs/benchmark_config.yaml`).

The smoke/full benchmark run will build the brute-force ground-truth cache on
its first pass. Subsequent runs reuse the cached numpy files.

## Limitations & Considerations

- **Embedding-defined relevance**: Ground truth now derives from brute-force
  nearest-neighbour search over the sentence-transformer embeddings. This
  provides deterministic recall targets but no longer reflects human-labelled
  qrels. Changing the embedding model or normalisation directly changes the
  reference answers.
- **Ground-truth width**: If `ground_truth_k` exceeds the corpus size or a
  query’s unique neighbour count, the loader automatically shrinks the width to
  the available neighbours. Downstream evaluation should cope with varying `K`.
- **Model choice**: Because relevance is embedding-driven, swapping
  `MODEL_NAME` requires regenerating embeddings *and* recomputing the cached
  ground truth so that datasets and evaluation remain aligned.
- **Cache hygiene**: The `msmarco_<hash>_top{K}groundtruth.npy` filenames encode
  the effective `K`. Clearing `results/cache/` forces a rebuild if you change
  the metric, embeddings, or requested depth.

Update this methodology whenever the sampling, embedding, or ground-truth code
changes to keep the documentation accurate.
