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

## Embedding & Ground-Truth Pipeline

Implementation: `src/dataprep/embed_msmarco.py` (lines 17–260).

1. **Configuration**  
   Uses the same YAML (`configs/ms_marco_subset_embed.yaml`) to locate the TSV
   snapshot (`embeddings.INPUT_DIR`) and to choose:

   - `embeddings.MODEL_NAME`: default `all-MiniLM-L6-v2`.
   - `embeddings.BATCH_SIZE`: default `256`.
   - `embeddings.GROUND_TRUTH_K`: number of positives stored per query
     (default `200`).
   - `embeddings.OUTPUT_DIR`: default
     `/storage/ice-shared/.../msmarco_v1_embeddings`.

2. **Loading TSVs**  
   Reads the subsampled `corpus.tsv` and `queries.tsv`, collecting parallel
   lists of IDs and text. Any malformed rows are skipped with warnings.

3. **Ground-truth construction**

   - Loads the dev qrels (`ir_datasets.load("msmarco-passage/dev")`) and maps
     `query_id → list[doc_id]` pairs where `relevance > 0`.
   - Builds a lookup from passage IDs to their index in the sampled corpus.
   - For each sampled query:
     - Collects all relevant passage indices in corpus order, deduplicating
       while preserving the qrels sequence.
     - Truncates to `GROUND_TRUTH_K` positives; if fewer exist, pads by
       repeating the last available index so every row has a fixed width.
     - Drops queries with zero surviving positives (e.g., because the sampled
       corpus omitted all relevant passages).
   - Records how many queries were discarded; if none remain the script aborts
     with a diagnostic so the sample sizes can be increased.

   The resulting matrix has shape `(n_queries_retained, GROUND_TRUTH_K)` with
   `int32` indices into the passage embedding array.

4. **Embedding generation**

   - Picks `cuda` when available, otherwise `cpu`.
   - Loads `SentenceTransformer(MODEL_NAME)` once.
   - Encodes all passages and retained queries with the configured batch size.
   - Ensures the embeddings are stored as `float32`.

5. **Outputs**

   The following files are written atomically to `embeddings.OUTPUT_DIR`:

   | File | Contents |
   |------|----------|
   | `passage_embeddings.npy` | `(num_passages, d)` float32 embeddings |
   | `passage_ids.npy` | `num_passages` array of string IDs (parallel to embeddings) |
   | `query_embeddings.npy` | `(num_queries_retained, d)` float32 embeddings |
   | `query_ids.npy` | `num_queries_retained` array of string IDs |
   | `ground_truth.npy` | `(num_queries_retained, GROUND_TRUTH_K)` int32 indices |
   | `ground_truth_doc_ids.json` | Mapping `query_id → [passage_id, ...]` |
   | `metadata.json` | Seed, counts, embedding dim, model name, etc. |

   Existing files are overwritten. The script verifies dimensional consistency
   between embeddings and ground truth before saving.

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

`Dataset._process_msmarco_preembedded()` (see `src/benchmark/dataset.py:763-843`)
detects these options, memory-maps `passage_embeddings.npy` when requested, and
validates that each ground-truth index falls within the passage array. The
benchmarking harness can then run directly on the dense vectors without reading
parquet shards or recomputing qrels.

## Reproduction Checklist

1. Ensure `ir_datasets` is installed and `IR_DATASETS_HOME` points at the raw
   MSMARCO cache (or allow the scripts to set the default).
2. `python src/dataprep/subsample_msmarco.py`
3. `python src/dataprep/embed_msmarco.py`
4. Execute your preferred benchmark config (e.g.,
   `python scripts/run_full_benchmark.py --config configs/benchmark_config.yaml`).

Both scripts are deterministic given the fixed configuration; setting different
sample sizes or seeds will necessarily change the subset, embeddings, and
ground truth.

## Limitations & Considerations

- **Positive coverage**: Queries whose relevant passages fall outside the sampled
  corpus are dropped. Monitor the `dropped_queries` field in `metadata.json` and
  increase `CORPUS_SAMPLE_SIZE` if you need more recall.
- **Ground-truth width**: Padding ensures a fixed `GROUND_TRUTH_K`, but if a
  query only had (say) 5 positives, the remaining 195 entries repeat the last
  index. Downstream metrics that assume unique neighbors should account for this.
- **Model choice**: Changing `MODEL_NAME` alters both embedding geometry and
  ground truth. Regenerate the bundle and update `ground_truth_k` as needed.
- **Metric alignment**: The benchmark config currently evaluates cosine
  similarity, matching the default sentence-transformer embeddings. Switching
  to L2 would require recomputing ground truth or accepting a metric mismatch.
- **Why we keep human qrels**: The ground truth stays tied to MS MARCO’s
  annotated relevance judgments. Replacing them with nearest neighbors produced
  by the same embedding model would bias the evaluation toward that model (a
  brute-force cosine search would achieve perfect recall by construction). The
  current approach keeps the human signal and only drops queries whose positives
  fall outside the sampled corpus; increase the sample size or resample to
  improve coverage if needed.

Update this methodology whenever the sampling, embedding, or ground-truth code
changes to keep the documentation accurate.
