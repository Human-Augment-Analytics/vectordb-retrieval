# One-page summary workflow (QPS vs Recall)

This doc explains how the one-page summary in `benchmark_results/qps_recall_summary.md` was produced so it can be replicated for future runs.

## Inputs

- Required: one or more benchmark run folders, each containing `all_results.json` and `*_config.yaml` files.
- This summary used:
  - `benchmark_results/benchmark_20251204_173318/benchmark_summary.md`
  - `benchmark_results/benchmark_20251204_152957/benchmark_summary.md`
  - `benchmark_results/benchmark_20251204_173318/all_results.json`
  - `benchmark_results/benchmark_20251204_152957/all_results.json`
  - `benchmark_results/benchmark_20251204_173318/random/random_config.yaml`
  - `benchmark_results/benchmark_20251204_173318/glove50/glove50_config.yaml`

## Process overview

1) **Read metrics**
   - Use each run’s `all_results.json` to extract per‑algorithm `qps` and `recall` for each dataset.
   - Keep algorithm names as the plot labels.

2) **Plot QPS vs recall**
   - Make one plot per dataset (e.g., `random`, `glove50`).
   - X‑axis: QPS on a log scale.
   - Y‑axis: recall for `topk` (these runs used `topk=20`).
   - Overlay runs using different markers/colors; add a legend keyed by run timestamp.
   - Annotate each point with the algorithm name.
   - Output SVGs to `benchmark_results/`:
     - `qps_recall_random.svg`
     - `qps_recall_glove50.svg`

3) **Collect algorithm implementation details**
   - Read the dataset config YAMLs (`*_config.yaml`) to list algorithm definitions:
     - Indexer/searcher class names
     - Key parameters (e.g., `HNSWIndexer` M/ef, Faiss index keys, `nprobe`, LSH bits)
   - Include common run settings (metric, topk, n_queries, repeat).

4) **Collect dataset details**
   - Read dataset settings from config YAML:
     - Dataset name, dimensions, train size/limit, test size, queries, ground_truth_k, dataset seed.

5) **Write one-page summary**
   - Create `benchmark_results/qps_recall_summary.md` with sections:
     - Plots (embedded SVGs)
     - Algorithms and implementation details
     - Dataset details
     - Brief takeaways based on the plots

## Example plotting script (SVG only)

The environment lacked plotting libraries, so a minimal Python script generated SVG directly. This is sufficient for QPS‑recall scatter plots and avoids extra dependencies.

Key points to implement:
- Parse `all_results.json`.
- Use log‑scale on QPS.
- Draw axes, ticks, labels, points, and a legend.
- Save to `benchmark_results/qps_recall_<dataset>.svg`.

## Naming and locations

- Output summary: `benchmark_results/qps_recall_summary.md`
- Plot assets: `benchmark_results/qps_recall_<dataset>.svg`

## Checklist for future runs

- [ ] Identify run folders and list datasets
- [ ] Extract `qps` and `recall` per algorithm from `all_results.json`
- [ ] Generate QPS‑recall plots (per dataset)
- [ ] Pull algorithm config details from `*_config.yaml`
- [ ] Pull dataset sizes and dimensions from `*_config.yaml`
- [ ] Write the one‑page summary markdown with plots + details + takeaways
