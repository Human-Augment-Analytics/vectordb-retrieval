# Glove50 Benchmark Methodology

This document captures the exact process the repository uses to turn the
Stanford GloVe 50-dimensional embeddings into a repeatable benchmarking task.
The goal is to make every assumption explicit so future academic write-ups can
reference a stable description of the dataset preparation and evaluation
procedure.

## Source Data

- **Corpus**: `glove.6B.50d.txt` from the official Stanford GloVe release  
  (`http://nlp.stanford.edu/data/glove.6B.zip`).
- **Embedding space**: 50-dimensional real-valued vectors trained on
  6B tokens of Wikipedia + Gigaword.
- **Download path**: by default the file is stored under
  `data/glove50/glove.6B.50d.txt`. The `Dataset.download()` helper handles
  retrieval if the file is missing.

## Preprocessing Pipeline

The implementation lives in `src/benchmark/dataset.py`, method
`Dataset._process_glove()` (lines 530–566 at the time of writing). The steps are:

1. **Load embeddings**  
   Parse every row of `glove.6B.50d.txt`, splitting on whitespace to obtain the
   token (ignored) and the 50D embedding. Each vector is converted to a
   `float32` NumPy array.

2. **Train/test split**  
   - Fix the NumPy random seed to `42` for reproducibility.
   - Sample `1,000` unique indices uniformly at random without replacement to
     serve as the *query* set.
   - Assign all remaining vectors to the *training* (a.k.a. database) split.
   - Shapes produced by the current corpus: ~399,000 training vectors and
     1,000 query vectors.

3. **Ground-truth construction**  
   - For each query vector, compute the Euclidean (`L2`) distance against every
     training vector.
   - Sort the distances and record the indices of the `k = 100` nearest
     neighbors. These index lists become the ground-truth matrix
     (`test_size × k`, `int32`).
   - No ties are broken specially; NumPy’s `argsort` order on equal distances is
     deterministic.

4. **Caching**  
   The dataset loader caches NumPy arrays for train vectors, queries, and ground
   truth inside the dataset’s cache directory so subsequent runs can load the
   preprocessed tensors without recomputation.

## Evaluation Settings

- **Similarity metric**: Euclidean distance (`metric: l2`) in both preprocessing
  and benchmark runs. This matches the ground-truth construction.
- **Default query budget**: Benchmarks typically evaluate the full set of
  1,000 queries, though configs can override `n_queries`.
- **Top-k evaluation**: The global benchmark `topk` setting (100 by default in
  `configs/default.yaml`) should not exceed the stored `ground_truth_k` (100).

## Reproducibility Notes

- To regenerate the processed dataset from scratch, delete the cached arrays
  under `data/glove50/` and run any benchmark entry point that references the
  `glove50` dataset, e.g.:

  ```bash
  python -m src.experiments.run_experiment --config configs/default.yaml --output-dir results
  ```

  Ensure the config includes `glove50` in its dataset list or select a profile
  like `configs/benchmark_config.yaml` where `glove50` is enabled.

- All randomness is controlled by the fixed seed (`42`) inside
  `_process_glove()`. Changing the seed or sampling strategy will alter the
  query split and thus the ground truth.

## Limitations & Considerations

- **Semantic labeling**: Unlike MS MARCO, there are no human relevance judgments.
  Ground truth is purely geometric: the “correct” neighbors are those closest in
  Euclidean distance. This makes Glove50 suitable for measuring the fidelity of
  approximate nearest-neighbor algorithms in a static vector space.

- **Vocabulary bias**: Queries and training vectors originate from the same
  distribution (GloVe vocabulary). Evaluations therefore test retrieval within
  the embedding space rather than task-specific relevance.

- **Metric dependence**: Because both ground truth and evaluation use L2,
  swapping to cosine similarity would require recomputing the ground truth or
  accepting a metric mismatch.

Document history: first authored on 2025-01-06 from the repository state wherein
`Dataset._process_glove()` computes training/query splits and ground truth as
described above. Update this file if the preprocessing code changes.
