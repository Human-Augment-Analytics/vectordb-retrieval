# Tests Overview

This directory hosts the project’s `pytest`-based test suite. The goal is to provide fast confidence checks for individual algorithms, modular index/search pairings, and benchmark plumbing without running the full experimental pipeline.

## Prerequisites

- Python 3.10+
- `pytest` (installed via `pip install -r requirements.txt`)
- Optional: `faiss-cpu` if you want to run tests that exercise FAISS/HNSW searchers. Tests that require FAISS are marked and can be skipped.

## Layout

| File | Purpose |
| ---- | ------- |
| `conftest.py` | Bootstraps Python paths so `src/` modules can be imported and exposes reusable helpers (e.g., FAISS availability check). |
| `test_smoke.py` | Simple sanity test ensuring pytest wiring works. |
| `test_composite_algorithm.py` | Validates modular index/search combinations: compares brute-force results with NumPy and (optionally) checks FAISS-backed HNSW search. |
| `test_benchmark_runner_modular.py` | Spins up a tiny benchmark configuration to confirm indexer/searcher references resolve and outputs are produced. |
| `pytest.ini` (repo root) | Central pytest configuration: warning filters, default command-line options, custom markers (`requires_faiss`). |

## Running the Suite

Run everything:

```bash
pytest
```

Skip FAISS-dependent tests (useful if FAISS is not installed):

```bash
pytest -m "not requires_faiss"
```

Run a specific test file:

```bash
pytest tests/test_composite_algorithm.py
```

Run an individual test case using node ids:

```bash
pytest tests/test_composite_algorithm.py::test_composite_bruteforce_linear_matches_numpy
```

## Writing New Tests

1. **File naming**: Create files named `test_*.py`. Pytest will automatically discover tests in this directory.
2. **Imports**: Import from the `src` package (e.g., `from src.algorithms.modular import CompositeAlgorithm`). The path adjustments in `conftest.py` make this work without additional sys.path hacks.
3. **Fixtures/utilities**: Add common fixtures to `conftest.py` so they are available to all tests. Keep helpers simple and deterministic.
4. **Markers**: If a test requires optional dependencies (e.g., FAISS or network access), add a custom marker in `pytest.ini` and decorate the test with `@pytest.mark.<marker>`. For FAISS, reuse `@pytest.mark.requires_faiss`.
5. **Randomness**: Use a fixed `RandomState` or explicit seeds to keep tests reproducible.
6. **Temporary files**: Prefer pytest’s `tmp_path` fixture for creating disposable directories/files.
7. **Assertions**: Use `numpy.testing` helpers when comparing arrays (`np.testing.assert_array_equal`, `assert_allclose`) to get informative diffs.

### Example Skeleton for a New Algorithm Test

```python
import numpy as np

from src.algorithms.modular import CompositeAlgorithm

def test_new_algo_behaviour():
    train = np.random.RandomState(0).randn(128, 16).astype(np.float32)
    queries = np.random.RandomState(1).randn(10, 16).astype(np.float32)

    algo = CompositeAlgorithm(
        name="new_algo",
        dimension=train.shape[1],
        metric="l2",
        indexer={"type": "YourIndexer", "param": 42},
        searcher={"type": "YourSearcher", "metric": "l2"},
    )
    algo.build_index(train)
    distances, indices = algo.batch_search(queries, k=5)

    assert distances.shape == (10, 5)
    assert indices.shape == (10, 5)
```

## Debugging Tips

- Use `pytest -vv` for verbose output, or `-s` to print stdout/stderr from tests.
- To isolate failures, re-run only the failing node id (pytest prints it in the summary).
- If imports fail, confirm the repo root and `src/` directory are on the Python path; `conftest.py` should handle this automatically.

## Contributing Guidelines

- Keep tests fast (sub-second ideally); avoid large datasets or long loops.
- Prefer deterministic data; minimise reliance on external resources.
- Update this README when adding new fixtures, markers, or conventions.
