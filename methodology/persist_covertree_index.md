# Covertree v2.2 MSMARCO Index Persistence

## Executive Summary

`covertree_v2_2` index persistence is now implemented and used to split the expensive MSMARCO index build from downstream retrieval on PACE. The goal is operational rather than algorithmic: build the Cover Tree once on shared storage, then reuse it in later jobs so long-running build steps do not repeatedly consume Slurm walltime.

The implementation is generic at the benchmark orchestration layer and specific at the algorithm layer. `ExperimentRunner` understands persistence modes (`build_only`, `retrieve_only`, `build_and_retrieve`) for any algorithm, while `CoverTreeV2_2` is the only algorithm that currently implements `save_index()` and `load_index()`. In full comparison runs, only Covertree loads a persisted artifact on MSMARCO; all other algorithms still build their indexes in the current job.

The current checked-in PACE workflow uses a fixed artifact path on shared storage:

`/storage/ice-shared/cs8903onl/vectordb-retrieval/indexes/covertree_v2_2/msmarco_cosine_base50000_q200_gt200`

That directory name is now only a human label. Compatibility is enforced by the persisted manifest, dataset fingerprint, and config hash, not by the folder name. This matters because the current build/retrieve configs use `base_limit: 100000` while still reusing the legacy fixed directory name. Correctness comes from the fingerprint validation, and overwriting is controlled by `force_rebuild`.

The persistence workflow has been validated on PACE. Build-only and retrieve-only Covertree runs complete successfully, and the full all-algorithms MSMARCO comparison now behaves correctly after the separate MSMARCO memmap cache-loader fix. Benchmark runs also now auto-generate `one-page-summary.md` and QPS-vs-recall SVGs in addition to the standard `benchmark_summary.md`.

## Purpose

The persistence design exists to solve one concrete problem:

1. MSMARCO Covertree builds can take long enough to hit Slurm walltime limits.
2. Rebuilding the same Covertree index for every downstream retrieval/comparison run wastes cluster time.
3. The build artifact needs compatibility checks so an old index is not silently reused against a changed dataset slice or changed algorithm configuration.

## Implemented Design

### Persistence API

`BaseAlgorithm` now exposes optional persistence hooks:

1. `save_index(artifact_dir, context=None)`
2. `load_index(artifact_dir, context=None)`

These default to `NotImplementedError`, so persistence is opt-in per algorithm. `CoverTreeV2_2` overrides both methods.

Relevant code:

1. [base_algorithm.py](/Users/lipengyuan/PycharmProjects/vectordb-retrieval/src/algorithms/base_algorithm.py)
2. [covertree_v2_2.py](/Users/lipengyuan/PycharmProjects/vectordb-retrieval/src/algorithms/covertree_v2_2.py)

### Benchmark Orchestration

`ExperimentRunner` now supports algorithm-level persistence configuration through the `persistence` block in an algorithm config.

Supported modes:

1. `build_only`
2. `retrieve_only`
3. `build_and_retrieve`

Mode behavior:

1. `build_only`
   - build index
   - persist artifact
   - skip search/evaluation
   - return metrics with `status=build_only`
2. `retrieve_only`
   - require configured `artifact_dir`
   - load persisted index if present
   - run search/evaluation using the loaded index
   - report `index_source=loaded`
3. `build_and_retrieve`
   - build index
   - optionally persist it
   - run search/evaluation in the same job

Relevant code:

1. [experiment_runner.py](/Users/lipengyuan/PycharmProjects/vectordb-retrieval/src/experiments/experiment_runner.py)

### Current Failure Semantics

The code supports both hard-fail and fallback behavior, but the checked-in PACE configs intentionally hard-fail on missing artifacts.

Behavior:

1. If `retrieve_only` and `fail_if_missing=true`, a missing artifact raises `FileNotFoundError`.
2. If `retrieve_only` and `fail_if_missing=false`, the code can fall back to rebuilding in the current job.
3. The recommended and checked-in MSMARCO retrieve configs use `fail_if_missing=true`.

## Persisted Artifact Format

`CoverTreeV2_2.save_index()` writes the artifact atomically into the configured directory.

Current files:

1. `manifest.json`
2. `build_metrics.json`
3. `vectors.npy`
4. `tree_indices.npy`
5. `tree_levels.npy`
6. `tree_child_offsets.npy`
7. `tree_children.npy`
8. `WRITE_COMPLETE`

What they contain:

1. `vectors.npy`
   - persisted working vectors used during search
   - for cosine, these are already normalized
2. `tree_indices.npy`
   - vector index stored at each serialized node
3. `tree_levels.npy`
   - Cover Tree level for each serialized node
4. `tree_child_offsets.npy`
   - CSR-style offsets into the flattened child array
5. `tree_children.npy`
   - flattened child node ids
6. `manifest.json`
   - schema version
   - algorithm type and algorithm name
   - metric
   - dimension
   - vector count
   - node count
   - max level
   - root node id
   - creation timestamp
   - dataset fingerprint
   - dataset fingerprint payload
   - config hash
   - embedded `build_metrics`
   - file names
7. `build_metrics.json`
   - convenience copy of build-time metadata
8. `WRITE_COMPLETE`
   - sentinel proving the artifact directory was fully written

Write behavior:

1. create a temporary directory under the artifact parent
2. write all numpy arrays and metadata there
3. write `WRITE_COMPLETE` last
4. rename the temp directory into the final artifact path

Load behavior:

1. require artifact directory to exist
2. require `WRITE_COMPLETE`
3. require `manifest.json`
4. validate manifest compatibility
5. load arrays
6. reconstruct tree nodes and child links
7. restore `root`, `max_level`, `self._working_vectors`, `self.vectors`, and `index_built`

## Compatibility Model

Two hashes are used:

1. `dataset_fingerprint`
2. `config_hash`

### Dataset Fingerprint

`dataset_fingerprint` is a SHA-256 hash over a structured payload built by `ExperimentRunner`.

Current payload fields:

1. `dataset`
2. `algorithm_name`
3. `algorithm_type`
4. `metric`
5. `dimension`
6. `train_count`
7. selected `dataset_options`

Selected `dataset_options` currently included:

1. `embedded_dataset_dir`
2. `passage_embeddings_path`
3. `query_embeddings_path`
4. `base_limit`
5. `query_limit`
6. `ground_truth_k`
7. `use_preembedded`
8. `use_memmap_cache`

If a passage-embedding file can be resolved, file metadata is also included:

1. absolute path
2. size in bytes
3. mtime

### Config Hash

`config_hash` is another SHA-256 hash over:

1. dataset name
2. dataset options
3. algorithm name
4. algorithm config with the `persistence` block removed
5. `topk`
6. `n_queries`
7. `query_batch_size`

### Validation Rules

`CoverTreeV2_2.load_index()` currently validates:

1. `schema_version`
2. `algorithm_type`
3. metric match
4. dimension match
5. dataset fingerprint match when one is supplied by the caller
6. config hash match when both caller and artifact provide one

This means the fixed artifact path is safe as long as the retrieval job passes the same dataset slice and compatible algorithm settings.

## Path Policy

Two path policies are implemented:

1. `fixed`
2. `versioned`

Behavior:

1. `fixed`
   - uses `artifact_dir` exactly as configured
2. `versioned`
   - resolves to `artifact_dir/<version_tag>`
   - if `version_tag` is absent, resolves to `artifact_dir/<dataset_fingerprint>`

Current checked-in PACE configs use `path_policy: fixed`.

## MSMARCO-Specific Notes

### Pre-embedded Data

The current MSMARCO workflow uses pre-embedded numpy arrays stored on shared storage and loaded through dataset options such as:

1. `use_preembedded: true`
2. `embedded_dataset_dir: /storage/ice-shared/.../msmarco_v1_embeddings`
3. `base_limit`
4. `query_limit`
5. `ground_truth_k`
6. `cache_dir`

### Memmap Cache Fix

Persistence and MSMARCO cache loading are related operationally but are separate features.

The codebase now includes an MSMARCO memmap metadata fix so `.npy`-backed cached train vectors are reloaded with `np.load(..., mmap_mode="r")` instead of raw `np.memmap(...)`. Without that fix, non-Covertree algorithms could read corrupted vectors from cache and produce misleading recall results. This bug was independent of Covertree persistence, but it affected full MSMARCO comparison runs and is now part of the validated workflow.

## Current Configs And Jobs

Current persistence-oriented configs:

1. [benchmark_all_covertree_v2_2_build.yaml](/Users/lipengyuan/PycharmProjects/vectordb-retrieval/configs/benchmark_all_covertree_v2_2_build.yaml)
2. [benchmark_all_covertree_v2_2_retrieve.yaml](/Users/lipengyuan/PycharmProjects/vectordb-retrieval/configs/benchmark_all_covertree_v2_2_retrieve.yaml)
3. [benchmark_all_datasets_msm100k_covertree_reuse.yaml](/Users/lipengyuan/PycharmProjects/vectordb-retrieval/configs/benchmark_all_datasets_msm100k_covertree_reuse.yaml)

Current PACE job scripts:

1. [codex_covertree_v2_2_msmarco_build.sbatch](/Users/lipengyuan/PycharmProjects/vectordb-retrieval/slurm_jobs/codex_covertree_v2_2_msmarco_build.sbatch)
2. [codex_covertree_v2_2_msmarco_retrieve.sbatch](/Users/lipengyuan/PycharmProjects/vectordb-retrieval/slurm_jobs/codex_covertree_v2_2_msmarco_retrieve.sbatch)
3. [codex_all_datasets_msm100k_reuse_ct.sbatch](/Users/lipengyuan/PycharmProjects/vectordb-retrieval/slurm_jobs/codex_all_datasets_msm100k_reuse_ct.sbatch)

How they are intended to be used:

1. build the Covertree artifact once with the build config
2. run a Covertree-only retrieval job, or
3. run the full all-algorithms comparison where Covertree loads the persisted artifact and all other algorithms build normally

Important operational note:

1. the checked-in artifact path still contains the suffix `base50000`
2. current configs use `base_limit: 100000`
3. this is safe because compatibility is checked by the manifest fingerprint and config hash
4. human-readable folder naming is stale, but the implementation is still correct

## Metrics And Reporting

Persistence adds the following metrics to algorithm results:

1. `index_source`
2. `index_load_time_s`
3. `persistence_mode`
4. `persist_dir`
5. `dataset_fingerprint`
6. `config_hash`

Special cases:

1. `build_only` results carry `status=build_only`
2. `retrieve_only` results report `build_time_s` from the persisted manifest, not from the current retrieval job

Benchmark outputs now include:

1. `benchmark_summary.md`
2. `one-page-summary.md`
3. `qps_recall_summary.md`
4. `qps_recall_<dataset>.svg`

The one-page summary integration is orthogonal to persistence, but it is now part of the standard benchmark workflow used to validate persistence-based runs.

## Validated Runs

PACE runs that validated the current persistence workflow:

1. Build-only Covertree run
   - job `4183296`
   - output dir `benchmark_results/benchmark_20260304_081404`
   - completed successfully
   - benchmark runtime `4389.77s`
2. Full all-algorithms comparison reusing persisted Covertree on MSMARCO
   - job `4183297`
   - output dir `benchmark_results/benchmark_20260304_092747`
   - completed successfully
   - Covertree loaded persisted index
   - non-Covertree algorithms built normally in-job
3. Later full comparison with LSH tuning and automatic one-page-summary validation
   - job `4206146`
   - output dir `benchmark_results/benchmark_20260305_070532`
   - completed successfully
   - emitted `one-page-summary.md` and per-dataset SVG plots automatically

## Tests

Tests covering the implemented design:

1. [test_covertree_v2_2.py](/Users/lipengyuan/PycharmProjects/vectordb-retrieval/tests/algorithms/test_covertree_v2_2.py)
   - persistence roundtrip
   - mismatch handling
   - incomplete artifact handling
2. [test_experiment_runner_persistence.py](/Users/lipengyuan/PycharmProjects/vectordb-retrieval/tests/test_experiment_runner_persistence.py)
   - build-only then retrieve-only workflow
   - missing-artifact failure
3. [test_dataset_msmarco_preembedded_limits.py](/Users/lipengyuan/PycharmProjects/vectordb-retrieval/tests/test_dataset_msmarco_preembedded_limits.py)
   - MSMARCO pre-embedded limit behavior
   - cache metadata compatibility
4. [test_benchmark_runner_modular.py](/Users/lipengyuan/PycharmProjects/vectordb-retrieval/tests/test_benchmark_runner_modular.py)
   - modular benchmark runner behavior
   - summary artifact generation including `one-page-summary.md`

PACE validation status:

1. persistence tests passed
2. dataset cache regression tests passed
3. benchmark runner summary-generation tests passed

## Limitations And Follow-Ups

1. Only `CoverTreeV2_2` currently implements persistence; the orchestration layer is generic, but other algorithms still need their own `save_index()` and `load_index()` methods if they are to reuse artifacts.
2. The current fixed artifact directory name is stale relative to the current 100k MSMARCO slice; this is safe but confusing and should be cleaned up later if readability matters.
3. `retrieve_only` fallback rebuilding still exists in code when `fail_if_missing=false`, even though the recommended PACE configs use hard-fail behavior.
4. The persistence artifact stores the search-ready working vectors, not the original raw vectors. This is intentional for cosine search because the normalized representation is what the loaded Covertree uses at query time.
