from __future__ import annotations

import json
import numpy as np

from src.benchmark.dataset import Dataset


def test_msmarco_preembedded_honors_base_and_query_limits(tmp_path) -> None:
    embedded_dir = tmp_path / "embeddings"
    embedded_dir.mkdir(parents=True, exist_ok=True)

    passage_embeddings = np.arange(120, dtype=np.float32).reshape(30, 4)
    query_embeddings = np.arange(80, dtype=np.float32).reshape(20, 4)

    np.save(embedded_dir / "passage_embeddings.npy", passage_embeddings)
    np.save(embedded_dir / "query_embeddings.npy", query_embeddings)

    dataset = Dataset(
        "msmarco",
        data_dir=str(tmp_path / "data"),
        options={
            "use_preembedded": True,
            "embedded_dataset_dir": str(embedded_dir),
            "base_limit": 7,
            "query_limit": 5,
            "ground_truth_k": 3,
            "metric": "l2",
            "use_memmap_cache": False,
            "cache_dir": str(tmp_path / "cache"),
        },
    )
    dataset.load(force_download=True)

    assert dataset.train_vectors.shape == (7, 4)
    assert dataset.test_vectors.shape == (5, 4)
    assert dataset.ground_truth.shape == (5, 3)


def test_msmarco_preembedded_memmap_cache_roundtrip_uses_npy_backend(tmp_path) -> None:
    embedded_dir = tmp_path / "embeddings"
    embedded_dir.mkdir(parents=True, exist_ok=True)

    passage_embeddings = np.arange(120, dtype=np.float32).reshape(30, 4)
    query_embeddings = np.arange(80, dtype=np.float32).reshape(20, 4)
    np.save(embedded_dir / "passage_embeddings.npy", passage_embeddings)
    np.save(embedded_dir / "query_embeddings.npy", query_embeddings)

    options = {
        "use_preembedded": True,
        "embedded_dataset_dir": str(embedded_dir),
        "base_limit": 8,
        "query_limit": 6,
        "ground_truth_k": 3,
        "metric": "l2",
        "use_memmap_cache": True,
        "cache_dir": str(tmp_path / "cache"),
    }

    first = Dataset("msmarco", data_dir=str(tmp_path / "data"), options=options.copy())
    first.load(force_download=True)
    first_train = np.asarray(first.train_vectors).copy()

    meta_path = first._memmap_meta_path()
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["train"]["format"] == "memmap"
    assert meta["train"]["memmap_backend"] == "npy"

    second = Dataset("msmarco", data_dir=str(tmp_path / "data"), options=options.copy())
    second.load(force_download=False)

    assert second.train_vectors.shape == (8, 4)
    assert np.allclose(np.asarray(second.train_vectors), first_train)
    assert np.isfinite(np.asarray(second.train_vectors)).all()


def test_load_memmap_cache_legacy_npy_without_backend(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    train = np.arange(24, dtype=np.float32).reshape(6, 4)
    test = np.arange(12, dtype=np.float32).reshape(3, 4)
    ground_truth = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int32)

    np.save(cache_dir / "train.npy", train)
    np.save(cache_dir / "test.npy", test)
    np.save(cache_dir / "gt.npy", ground_truth)

    meta = {
        "version": 1,
        "train": {
            "path": "train.npy",
            "dtype": "float32",
            "shape": [6, 4],
            "format": "memmap",
        },
        "test": {"path": "test.npy", "format": "numpy"},
        "ground_truth": {"path": "gt.npy", "format": "numpy"},
    }
    meta_path = cache_dir / "legacy_memmap.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    dataset = Dataset(
        "msmarco",
        data_dir=str(tmp_path / "data"),
        options={"cache_dir": str(cache_dir), "use_memmap_cache": True},
    )
    dataset._load_memmap_cache(str(meta_path))

    assert np.array_equal(np.asarray(dataset.train_vectors), train)
    assert np.array_equal(np.asarray(dataset.test_vectors), test)
    assert np.array_equal(np.asarray(dataset.ground_truth), ground_truth)


def test_load_memmap_cache_raw_backend(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    train = np.arange(20, dtype=np.float32).reshape(5, 4)
    train_memmap_path = cache_dir / "train.memmap"
    writable = np.memmap(train_memmap_path, dtype=np.float32, mode="w+", shape=train.shape)
    writable[:] = train
    writable.flush()
    del writable

    test = np.arange(8, dtype=np.float32).reshape(2, 4)
    ground_truth = np.array([[0, 1], [2, 3]], dtype=np.int32)
    np.save(cache_dir / "test.npy", test)
    np.save(cache_dir / "gt.npy", ground_truth)

    meta = {
        "version": 2,
        "train": {
            "path": "train.memmap",
            "dtype": "float32",
            "shape": [5, 4],
            "format": "memmap",
            "memmap_backend": "raw",
        },
        "test": {"path": "test.npy", "format": "numpy"},
        "ground_truth": {"path": "gt.npy", "format": "numpy"},
    }
    meta_path = cache_dir / "raw_memmap.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    dataset = Dataset(
        "msmarco",
        data_dir=str(tmp_path / "data"),
        options={"cache_dir": str(cache_dir), "use_memmap_cache": True},
    )
    dataset._load_memmap_cache(str(meta_path))

    assert np.array_equal(np.asarray(dataset.train_vectors), train)
    assert np.array_equal(np.asarray(dataset.test_vectors), test)
    assert np.array_equal(np.asarray(dataset.ground_truth), ground_truth)
