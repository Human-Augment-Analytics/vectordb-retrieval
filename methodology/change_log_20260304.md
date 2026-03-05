# Change Log - 2026-03-04

## Topic

MSMARCO cache metadata loader fix for `.npy`-backed memmap entries and one-time forced clean cache rebuild.

## Why this change

Recent MSMARCO all-algorithm runs showed:
- `covertree_v2_2` (loaded persisted index) recall near `1.0`
- all other algorithms (built on-the-fly) recall near `0.0`

Root cause was traced to cached train-vector loading: `.npy` files marked as `format: memmap` were being opened with raw `np.memmap(...)` instead of `np.load(..., mmap_mode='r')`, which can misread data due to `.npy` headers.

## Files changed

- `src/benchmark/dataset.py`
  - Added `MSMARCO_CACHE_KEY_VERSION = "msmarco_cache_v2_npy_memmap_loader_fix_20260304"`.
  - In `Dataset.load()`, MSMARCO now injects `_cache_key_version` by default to force one-time cache-key invalidation.
  - Added `_infer_memmap_backend_from_path(...)`.
  - `_save_memmap_cache(...)` now writes metadata version `2` and `train.memmap_backend` (`npy` or `raw`) when `format: memmap`.
  - `_load_memmap_cache(...)` now:
    - loads `.npy` memmap entries with `np.load(..., mmap_mode='r', allow_pickle=False)`,
    - keeps raw `np.memmap(...)` for `memmap_backend: raw`,
    - falls back to extension-based inference for legacy metadata without `memmap_backend`,
    - validates loaded dtype/shape against metadata.

- `tests/test_dataset_msmarco_preembedded_limits.py`
  - Added `test_msmarco_preembedded_memmap_cache_roundtrip_uses_npy_backend`.
  - Added `test_load_memmap_cache_legacy_npy_without_backend`.
  - Added `test_load_memmap_cache_raw_backend`.

- `methodology/known_followups.md`
  - Added item #11 describing symptom, root cause, fix, and remaining PACE validation.

## Reproduction / validation commands

Intended (PACE):

```bash
cd /home/hice1/pli396/PycharmProjects/vectordb-retrieval
$HOME/scratch/vector-db-venv/bin/python -m pytest -q tests/test_dataset_msmarco_preembedded_limits.py
$HOME/scratch/vector-db-venv/bin/python -m pytest -q tests/test_experiment_runner_persistence.py
$HOME/scratch/vector-db-venv/bin/python -m pytest -q tests/algorithms/test_covertree_v2_2.py
```

Local attempts in this session:

```bash
pytest -q tests/test_dataset_msmarco_preembedded_limits.py
```

Blocked locally because system interpreter is Python 3.6 / pytest 5, while repo requires pytest>=7.

Local sanity checks that did run (direct Python scripts, no pytest):
- legacy `.npy` metadata without `memmap_backend` loads correctly (`loader_compat_ok`)
- raw memmap metadata with `memmap_backend: raw` loads correctly (`loader_raw_ok`)
- `.npy` memmap metadata write+load roundtrip sets `memmap_backend: npy` and reloads correctly (`npy_backend_roundtrip_ok`)

PACE connectivity attempts in this session:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 pace 'hostname && date'
```

Blocked by timeout to `login-ice.pace.gatech.edu:22`.

## Observed behavior after patch (code-level)

- MSMARCO cache keys are forced to new suffix once (clean rebuild trigger).
- Cache metadata can now distinguish `.npy` memmap loading from raw binary memmap loading.
- Legacy metadata remains loadable via path-based backend inference.

## Open risks / follow-ups

1. PACE-side benchmark confirmation is still pending due SSH timeout in this session.
2. Old corrupted cache files remain on shared storage but are no longer selected by default for MSMARCO because of the cache-key bump.
