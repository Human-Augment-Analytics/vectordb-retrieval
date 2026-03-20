"""Core experiment execution utilities."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import yaml

from .config import ExperimentConfig
from ..algorithms.base_algorithm import BaseAlgorithm
from ..benchmark.dataset import Dataset
from ..benchmark.evaluation import Evaluator


class ExperimentRunner:
    """Execute vector-retrieval experiments for a dataset and algorithm set."""

    def __init__(self, config: ExperimentConfig, output_dir: str = "results") -> None:
        self.config = config
        self.output_dir = output_dir
        self.dataset: Optional[Dataset] = None
        self.algorithms: Dict[str, BaseAlgorithm] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.algorithm_result_overrides: Dict[str, Dict[str, Any]] = {}
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = logging.getLogger("experiment_runner")

    # ------------------------------------------------------------------
    # Dataset & algorithm registration
    # ------------------------------------------------------------------
    def load_dataset(self) -> None:
        """Load the dataset defined in the configuration if not already loaded."""
        if self.dataset is not None:
            return

        self.logger.info(f"Loading dataset: {self.config.dataset}")
        dataset = Dataset(self.config.dataset, self.config.data_dir, options=self.config.dataset_options)
        dataset.load(force_download=self.config.force_download)
        self.dataset = dataset

    def register_algorithm(self, algorithm: BaseAlgorithm, name: Optional[str] = None) -> None:
        """Register an algorithm implementation for the current experiment."""
        if not isinstance(algorithm, BaseAlgorithm):
            raise TypeError("algorithm must inherit from BaseAlgorithm")

        algorithm_name = name or algorithm.get_name()
        if not algorithm_name:
            raise ValueError("Algorithm name must be provided")

        # Keep the algorithm's internal name synchronized with registration name.
        if getattr(algorithm, "name", None) != algorithm_name:
            setattr(algorithm, "name", algorithm_name)

        self.algorithms[algorithm_name] = algorithm
        self.logger.info(f"Registered algorithm: {algorithm_name}")

    # ------------------------------------------------------------------
    # Core execution flow
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Dict[str, Any]]:
        """Run all registered algorithms and return their evaluation metrics."""
        if not self.algorithms:
            raise RuntimeError("No algorithms registered for the experiment")

        self.load_dataset()
        assert self.dataset is not None  # mypy guard

        np.random.seed(self.config.seed)

        train_vectors = self.dataset.train_vectors
        test_queries = self.dataset.test_vectors
        ground_truth = self.dataset.ground_truth

        if train_vectors is None or test_queries is None:
            raise RuntimeError("Dataset did not provide train/test vectors")

        test_queries, ground_truth = self._select_query_subset(test_queries, ground_truth)

        algorithm_outputs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.results = {}

        for name, algorithm in self.algorithms.items():
            self.logger.info(f"Running experiment for algorithm: {name}")
            metrics, indices, query_times = self._run_single_algorithm(
                name, algorithm, train_vectors, test_queries
            )
            self.results[name] = metrics
            if indices is not None and query_times is not None:
                algorithm_outputs[name] = (indices, query_times)

        evaluator: Optional[Evaluator] = None
        if ground_truth is not None:
            evaluator = Evaluator(ground_truth)
            for name, (indices, query_times) in algorithm_outputs.items():
                metrics = evaluator.evaluate(name, indices, query_times)
                self.results[name].update(metrics)

                summary_k = min(100, self.config.topk)
                summary_key = f"recall@{summary_k}"

                if summary_key in metrics:
                    self.results[name]["recall"] = metrics[summary_key]
                elif metrics:
                    # Fallback to the largest available recall metric
                    available_recalls = sorted(
                        (key for key in metrics if key.startswith("recall@")),
                        key=lambda x: int(x.split("@")[-1])
                    )
                    if available_recalls:
                        self.results[name]["recall"] = metrics[available_recalls[-1]]

        for name in self.results:
            self._save_algorithm_results(name, self.results[name])

        self._save_combined_results()

        if evaluator is not None:
            evaluator.print_results()
            self._generate_plots(evaluator)

        self.logger.info("Experiment completed successfully.")
        return self.results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _select_query_subset(
        self, queries: np.ndarray, ground_truth: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply the n_queries limit while keeping ground-truth aligned."""
        n_available = len(queries)
        target = min(self.config.n_queries, n_available) if self.config.n_queries else n_available

        if target >= n_available:
            return queries, ground_truth

        indices = np.random.choice(n_available, target, replace=False)
        queries_subset = queries[indices]
        ground_truth_subset = ground_truth[indices] if ground_truth is not None else None

        self.logger.info(f"Using {target}/{n_available} queries for evaluation")
        return queries_subset, ground_truth_subset

    def _stable_hash(self, payload: Dict[str, Any]) -> str:
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _apply_algorithm_result_overrides(self, name: str, metrics: Dict[str, Any]) -> None:
        overrides = self.algorithm_result_overrides.get(name)
        if not isinstance(overrides, dict):
            return
        for key, value in overrides.items():
            metrics[key] = copy.deepcopy(value)

    def _algorithm_config(self, name: str) -> Dict[str, Any]:
        raw = self.config.algorithms.get(name, {})
        return copy.deepcopy(raw) if isinstance(raw, dict) else {}

    def _extract_persistence_config(self, algorithm_cfg: Dict[str, Any]) -> Dict[str, Any]:
        raw = algorithm_cfg.get("persistence", {})
        if not isinstance(raw, dict):
            return {}

        cfg = copy.deepcopy(raw)
        mode = str(cfg.get("mode", "build_and_retrieve")).strip().lower()
        if mode not in {"build_only", "retrieve_only", "build_and_retrieve"}:
            raise ValueError(f"Unsupported persistence mode '{mode}'")
        cfg["mode"] = mode

        path_policy = str(cfg.get("path_policy", "fixed")).strip().lower()
        if path_policy not in {"fixed", "versioned"}:
            raise ValueError(f"Unsupported persistence path_policy '{path_policy}'")
        cfg["path_policy"] = path_policy

        cfg["enabled"] = bool(cfg.get("enabled", False))
        cfg["force_rebuild"] = bool(cfg.get("force_rebuild", False))
        cfg["fail_if_missing"] = bool(cfg.get("fail_if_missing", True))
        return cfg

    def _build_dataset_fingerprint_payload(
        self,
        name: str,
        algorithm: BaseAlgorithm,
        algorithm_cfg: Dict[str, Any],
        train_vectors: np.ndarray,
    ) -> Dict[str, Any]:
        dataset_options = self.config.dataset_options if isinstance(self.config.dataset_options, dict) else {}
        selected_dataset_options: Dict[str, Any] = {}
        for key in (
            "embedded_dataset_dir",
            "passage_embeddings_path",
            "query_embeddings_path",
            "base_limit",
            "query_limit",
            "ground_truth_k",
            "use_preembedded",
            "use_memmap_cache",
        ):
            if key in dataset_options:
                selected_dataset_options[key] = dataset_options[key]

        metric = algorithm_cfg.get("metric")
        if metric is None:
            metric = getattr(algorithm, "metric_name", None) or getattr(algorithm, "metric", None)

        payload: Dict[str, Any] = {
            "dataset": self.config.dataset,
            "algorithm_name": name,
            "algorithm_type": algorithm.__class__.__name__,
            "metric": metric,
            "dimension": int(train_vectors.shape[1]),
            "train_count": int(train_vectors.shape[0]),
            "dataset_options": selected_dataset_options,
        }

        passage_embeddings_path: Optional[Path] = None
        if selected_dataset_options.get("passage_embeddings_path"):
            passage_embeddings_path = Path(selected_dataset_options["passage_embeddings_path"])
        elif selected_dataset_options.get("embedded_dataset_dir"):
            passage_embeddings_path = Path(selected_dataset_options["embedded_dataset_dir"]) / "passage_embeddings.npy"

        if passage_embeddings_path is not None:
            if passage_embeddings_path.exists():
                stat = passage_embeddings_path.stat()
                payload["passage_embeddings_file"] = {
                    "path": str(passage_embeddings_path.resolve()),
                    "size_bytes": int(stat.st_size),
                    "mtime": int(stat.st_mtime),
                }
            else:
                payload["passage_embeddings_file"] = {
                    "path": str(passage_embeddings_path),
                    "missing": True,
                }

        return payload

    def _resolve_persist_dir(
        self,
        persistence_cfg: Dict[str, Any],
        dataset_fingerprint: str,
    ) -> Optional[str]:
        artifact_dir = persistence_cfg.get("artifact_dir")
        if not artifact_dir:
            return None

        base = Path(str(artifact_dir))
        path_policy = persistence_cfg.get("path_policy", "fixed")
        if path_policy == "versioned":
            version_tag = persistence_cfg.get("version_tag")
            suffix = str(version_tag).strip() if version_tag else dataset_fingerprint
            return str(base / suffix)
        return str(base)

    def _run_single_algorithm(
        self,
        name: str,
        algorithm: BaseAlgorithm,
        train_vectors: np.ndarray,
        test_queries: np.ndarray,
    ) -> Tuple[Dict[str, Any], Optional[np.ndarray], Optional[np.ndarray]]:
        """Train the algorithm, execute queries, and collect core metrics."""
        algorithm_cfg = self._algorithm_config(name)
        persistence_cfg = self._extract_persistence_config(algorithm_cfg)
        persistence_enabled = persistence_cfg.get("enabled", False)
        persistence_mode = persistence_cfg.get("mode", "build_and_retrieve")
        persistence_force_rebuild = persistence_cfg.get("force_rebuild", False)
        persistence_fail_if_missing = persistence_cfg.get("fail_if_missing", True)

        dataset_fingerprint_payload = self._build_dataset_fingerprint_payload(
            name,
            algorithm,
            algorithm_cfg,
            train_vectors,
        )
        dataset_fingerprint = self._stable_hash(dataset_fingerprint_payload)

        hash_algorithm_cfg = copy.deepcopy(algorithm_cfg)
        if isinstance(hash_algorithm_cfg.get("persistence"), dict):
            hash_algorithm_cfg.pop("persistence", None)

        config_hash_payload = {
            "dataset": self.config.dataset,
            "dataset_options": self.config.dataset_options,
            "algorithm_name": name,
            "algorithm_config": hash_algorithm_cfg,
            "topk": self.config.topk,
            "n_queries": self.config.n_queries,
            "query_batch_size": self.config.query_batch_size,
        }
        config_hash = self._stable_hash(config_hash_payload)
        persist_dir = self._resolve_persist_dir(persistence_cfg, dataset_fingerprint)
        persistence_context: Dict[str, Any] = {
            "dataset_fingerprint": dataset_fingerprint,
            "dataset_fingerprint_payload": dataset_fingerprint_payload,
            "config_hash": config_hash,
            "force_rebuild": persistence_force_rebuild,
        }

        build_time = 0.0
        index_load_time_s = 0.0
        index_source = "built"

        if persistence_enabled and persistence_mode == "retrieve_only":
            if not persist_dir:
                raise ValueError(
                    f"Algorithm '{name}' has persistence enabled but no persistence.artifact_dir configured."
                )
            if not Path(persist_dir).is_dir():
                if persistence_fail_if_missing:
                    raise FileNotFoundError(
                        f"Missing persisted index for '{name}' at {persist_dir}. "
                        "Run build_only (or build_and_retrieve) first to create the artifact."
                    )
                build_start = time.time()
                algorithm.build_index(train_vectors)
                build_time = time.time() - build_start
            else:
                load_start = time.time()
                load_info = algorithm.load_index(persist_dir, context=persistence_context)
                index_load_time_s = time.time() - load_start
                index_source = "loaded"
                build_time = float(load_info.get("build_time_s", 0.0) or 0.0)
        else:
            build_start = time.time()
            algorithm.build_index(train_vectors)
            build_time = time.time() - build_start

            if persistence_enabled and persistence_mode in {"build_only", "build_and_retrieve"}:
                if not persist_dir:
                    raise ValueError(
                        f"Algorithm '{name}' has persistence enabled but no persistence.artifact_dir configured."
                    )
                persistence_context["build_metrics"] = {
                    "build_time_s": float(build_time),
                    "n_train": int(train_vectors.shape[0]),
                    "dimensions": int(train_vectors.shape[1]),
                    "timestamp": datetime.now().isoformat(),
                }
                algorithm.save_index(persist_dir, context=persistence_context)

        memory_usage_mb = self._estimate_memory_usage(algorithm, train_vectors)

        if persistence_enabled and persistence_mode == "build_only":
            metrics: Dict[str, Any] = {
                "algorithm": name,
                "parameters": algorithm.get_parameters(),
                "dataset": self.config.dataset,
                "n_train": int(train_vectors.shape[0]),
                "n_test": int(len(test_queries)),
                "dimensions": int(train_vectors.shape[1]),
                "topk": self.config.topk,
                "build_time_s": float(build_time),
                "index_memory_mb": float(memory_usage_mb),
                "qps": 0.0,
                "mean_query_time_ms": 0.0,
                "total_query_time_s": 0.0,
                "index_load_time_s": float(index_load_time_s),
                "index_source": index_source,
                "persistence_mode": persistence_mode,
                "persist_dir": persist_dir,
                "dataset_fingerprint": dataset_fingerprint,
                "config_hash": config_hash,
                "status": "build_only",
                "timestamp": datetime.now().isoformat(),
            }
            self._apply_algorithm_result_overrides(name, metrics)
            return metrics, None, None

        # Search phase
        k = self.config.topk
        indices = np.full((len(test_queries), k), -1, dtype=np.int64)
        query_times = np.zeros(len(test_queries), dtype=float)

        total_query_time = 0.0
        used_batch_api = False

        def _normalize_batch_indices(
            batch_result: Any, expected_rows: int, expected_k: int
        ) -> np.ndarray:
            if isinstance(batch_result, tuple):
                if len(batch_result) != 2:
                    raise ValueError("batch_search must return (distances, indices)")
                batch_result = batch_result[1]

            if isinstance(batch_result, list):
                rows = [np.asarray(row) for row in batch_result]
                output = np.full((expected_rows, expected_k), -1, dtype=np.int64)
                for row_idx, row in enumerate(rows[:expected_rows]):
                    limit = min(row.size, expected_k)
                    if limit > 0:
                        output[row_idx, :limit] = row[:limit]
                return output

            arr = np.asarray(batch_result)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.ndim != 2:
                raise ValueError("batch_search returned array with unexpected shape")
            if arr.shape[0] != expected_rows:
                if expected_rows == 1 and arr.shape[0] == expected_k:
                    arr = arr.reshape(1, -1)
                else:
                    raise ValueError(
                        f"batch_search returned {arr.shape[0]} rows, expected {expected_rows}"
                    )

            if arr.shape[1] < expected_k:
                padded = np.full((expected_rows, expected_k), -1, dtype=np.int64)
                padded[:, :arr.shape[1]] = arr
                arr = padded
            elif arr.shape[1] > expected_k:
                arr = arr[:, :expected_k]

            return arr.astype(np.int64, copy=False)

        if len(test_queries) > 0:
            batch_size_cfg = int(getattr(self.config, "query_batch_size", 0) or 0)
            if batch_size_cfg < 0:
                batch_size_cfg = 0
            batch_size = len(test_queries) if batch_size_cfg == 0 else min(batch_size_cfg, len(test_queries))

            try:
                cursor = 0
                while cursor < len(test_queries):
                    end = min(cursor + batch_size, len(test_queries))
                    batch_queries = test_queries[cursor:end]
                    batch_start = time.time()
                    batch_result = algorithm.batch_search(batch_queries, k=k)
                    elapsed = time.time() - batch_start
                    batch_indices = _normalize_batch_indices(batch_result, end - cursor, k)
                    indices[cursor:end] = batch_indices
                    per_query = elapsed / max((end - cursor), 1)
                    query_times[cursor:end] = per_query
                    total_query_time += elapsed
                    cursor = end

                used_batch_api = True
            except (AttributeError, NotImplementedError, TypeError, ValueError):
                # Reset to ensure clean fallback path
                indices.fill(-1)
                query_times.fill(0.0)
                total_query_time = 0.0

        if not used_batch_api:
            for idx, query in enumerate(test_queries):
                single_start = time.time()
                _, single_indices = algorithm.search(query, k=k)
                query_duration = time.time() - single_start
                query_times[idx] = query_duration
                indices[idx] = single_indices
                total_query_time += query_duration

        total_query_time = max(total_query_time, query_times.sum())
        mean_query_time_ms = (
            (total_query_time / max(len(test_queries), 1)) * 1000.0
            if len(test_queries) > 0
            else 0.0
        )

        qps = len(test_queries) / total_query_time if total_query_time > 0 else 0.0

        metrics: Dict[str, Any] = {
            "algorithm": name,
            "parameters": algorithm.get_parameters(),
            "dataset": self.config.dataset,
            "n_train": int(train_vectors.shape[0]),
            "n_test": int(len(test_queries)),
            "dimensions": int(train_vectors.shape[1]),
            "topk": self.config.topk,
            "build_time_s": float(build_time),
            "index_memory_mb": float(memory_usage_mb),
            "qps": float(qps),
            "mean_query_time_ms": float(mean_query_time_ms),
            "total_query_time_s": float(total_query_time),
            "index_load_time_s": float(index_load_time_s),
            "index_source": index_source,
            "persistence_mode": persistence_mode if persistence_enabled else None,
            "persist_dir": persist_dir if persistence_enabled else None,
            "dataset_fingerprint": dataset_fingerprint if persistence_enabled else None,
            "config_hash": config_hash if persistence_enabled else None,
            "timestamp": datetime.now().isoformat(),
        }

        self._apply_algorithm_result_overrides(name, metrics)

        return metrics, indices, query_times

    def _estimate_memory_usage(self, algorithm: BaseAlgorithm, train_vectors: np.ndarray) -> float:
        """Return algorithm-reported memory when available, otherwise estimate."""
        # Prefer an explicit algorithm implementation when present.
        if hasattr(algorithm, "get_memory_usage"):
            try:
                usage = algorithm.get_memory_usage()
                if usage is not None and usage > 0:
                    return float(usage)
            except Exception as exc:  # pragma: no cover - defensive guardrail
                self.logger.warning(f"Failed to read memory usage for {algorithm.get_name()}: {exc}")

        size_bytes = self._introspect_memory_bytes(algorithm, train_vectors=train_vectors)
        if size_bytes > 0:
            return size_bytes / (1024.0 * 1024.0)

        # Fallback: size of the raw training vectors in MB.
        return float(train_vectors.nbytes) / (1024.0 * 1024.0)

    def _introspect_memory_bytes(self, algorithm: BaseAlgorithm, train_vectors: Optional[np.ndarray] = None) -> int:
        """Best-effort memory estimation for algorithms without explicit reporting."""
        # 1) Check common FAISS-backed indices.
        index = getattr(algorithm, "index", None)
        size_bytes = max(0, self._faiss_index_size_bytes(index, train_vectors=train_vectors, algorithm=algorithm))
        if size_bytes > 0:
            size_bytes = self._enforce_minimum_vector_footprint(size_bytes, train_vectors)
            return size_bytes

        # 2) Composite/modular algorithms expose an index artifact.
        artifact = getattr(algorithm, "index_artifact", None)
        size_bytes = max(size_bytes, self._artifact_size_bytes(artifact, algorithm=algorithm, train_vectors=train_vectors))
        if size_bytes > 0:
            size_bytes = self._enforce_minimum_vector_footprint(size_bytes, train_vectors)
            return size_bytes

        # 3) LSH convenience wrapper keeps state on the searcher.
        size_bytes = max(size_bytes, self._lsh_state_size_bytes(getattr(algorithm, "searcher", None)))
        if size_bytes > 0:
            return size_bytes

        # 4) CoverTreeV2_2 stores working vectors and explicit nodes.
        size_bytes = max(size_bytes, self._covertree_size_bytes(algorithm))
        if size_bytes > 0:
            return size_bytes

        # 5) As a minimal signal, fall back to any retained vectors.
        vectors = getattr(algorithm, "vectors", None)
        if isinstance(vectors, np.ndarray):
            return int(vectors.nbytes)

        return 0

    def _enforce_minimum_vector_footprint(self, size_bytes: int, train_vectors: Optional[np.ndarray]) -> int:
        """
        Guard against faiss.index_size() returning placeholder values (e.g., 0–few bytes).
        Clamp to at least the raw training-vector footprint when we know the index
        stores the full dataset.
        """
        if train_vectors is None or not isinstance(train_vectors, np.ndarray):
            return size_bytes

        if size_bytes < 1024:  # clearly too small to hold an index
            return int(max(size_bytes, train_vectors.nbytes))

        return size_bytes

    def _faiss_index_size_bytes(self, index: Any, train_vectors: Optional[np.ndarray] = None, algorithm: Optional[BaseAlgorithm] = None) -> int:
        if index is None:
            return 0
        try:
            import faiss  # type: ignore
        except Exception:
            return 0

        try:
            base_size = int(faiss.index_size(index))
        except Exception:
            base_size = 0

        ntotal = int(getattr(index, "ntotal", 0) or 0)
        code_size = self._faiss_code_size(index, train_vectors)
        vector_bytes = ntotal * code_size if ntotal > 0 and code_size > 0 else 0

        # Add a rough adjacency overhead for HNSW graphs.
        graph_overhead = 0
        hnsw = getattr(index, "hnsw", None)
        if hnsw is not None:
            m_links = int(getattr(hnsw, "M", 0) or 0)
            if m_links <= 0 and algorithm is not None:
                m_links = int(getattr(getattr(algorithm, "indexer", None), "M", 0) or 0)
            # Two floats per edge (dist + id) as a coarse upper bound.
            graph_overhead = ntotal * max(m_links, 0) * 8

        total = max(base_size, vector_bytes) + graph_overhead
        try:
            alg_name = algorithm.get_name() if algorithm is not None else "<unknown>"
            index_name = index.__class__.__name__
            self.logger.debug(
                "Estimated index memory for %s (%s): base=%s bytes, vector_bytes=%s, "
                "graph_overhead=%s, total=%s, ntotal=%s, code_size=%s, M=%s",
                alg_name,
                index_name,
                base_size,
                vector_bytes,
                graph_overhead,
                total,
                ntotal,
                code_size,
                getattr(hnsw, "M", None) if hnsw is not None else None,
            )
        except Exception:
            pass

        return total

    def _faiss_code_size(self, index: Any, train_vectors: Optional[np.ndarray]) -> int:
        code_size = int(getattr(index, "code_size", 0) or 0)
        pq = getattr(index, "pq", None)
        if code_size <= 0 and pq is not None:
            code_size = int(getattr(pq, "code_size", 0) or 0)

        if code_size <= 0:
            dim = int(getattr(index, "d", 0) or 0)
            if dim <= 0 and train_vectors is not None:
                dim = train_vectors.shape[1]
            code_size = dim * 4 if dim > 0 else 0

        return code_size

    def _artifact_size_bytes(
        self,
        artifact: Any,
        algorithm: Optional[BaseAlgorithm] = None,
        train_vectors: Optional[np.ndarray] = None,
    ) -> int:
        if artifact is None:
            return 0

        data = getattr(artifact, "data", None)
        kind = getattr(artifact, "kind", None)

        if kind == "faiss":
            size_bytes = self._faiss_index_size_bytes(data, train_vectors=train_vectors, algorithm=algorithm)
            if size_bytes > 0:
                return size_bytes

        if kind == "raw_vectors" and hasattr(data, "nbytes"):
            return int(getattr(data, "nbytes", 0))

        if kind == "lsh" and isinstance(data, dict):
            return self._lsh_artifact_bytes(data)

        return self._approximate_object_bytes(data)

    def _lsh_artifact_bytes(self, data: Dict[str, Any]) -> int:
        """Estimate memory for LSH artifacts (projections, offsets, tables, vector store)."""
        size_bytes = 0
        for key in ("projections", "offsets", "vector_store", "bit_weights"):
            arr = data.get(key)
            if isinstance(arr, np.ndarray):
                size_bytes += int(arr.nbytes)

        tables = data.get("tables")
        if tables:
            size_bytes += self._lsh_tables_bytes(tables)

        return size_bytes

    def _lsh_state_size_bytes(self, searcher: Any) -> int:
        """Estimate memory when only the LSH searcher is accessible (LSH convenience wrapper)."""
        if searcher is None:
            return 0

        size_bytes = 0
        for attr in ("projections", "offsets", "vector_store", "bit_weights"):
            value = getattr(searcher, attr, None)
            if isinstance(value, np.ndarray):
                size_bytes += int(value.nbytes)

        tables = getattr(searcher, "tables", None)
        if tables:
            size_bytes += self._lsh_tables_bytes(tables)

        return size_bytes

    def _lsh_tables_bytes(self, tables: Any) -> int:
        """Roughly estimate memory for LSH hash tables (list[dict[hash_key -> list[int]]])."""
        if not isinstance(tables, list):
            return 0

        total_entries = 0
        overhead = 0
        for table in tables:
            if not isinstance(table, dict):
                continue
            overhead += sys.getsizeof(table)
            for bucket in table.values():
                if isinstance(bucket, list):
                    overhead += sys.getsizeof(bucket)
                    total_entries += len(bucket)

        # Assume 4 bytes per stored index plus container overhead.
        return overhead + int(total_entries * 4)

    def _covertree_size_bytes(self, algorithm: BaseAlgorithm) -> int:
        """Estimate memory used by the CoverTreeV2_2 implementation."""
        working = getattr(algorithm, "_working_vectors", None)
        size_bytes = int(working.nbytes) if isinstance(working, np.ndarray) else 0

        root = getattr(algorithm, "root", None)
        if root is None:
            return size_bytes

        # Traverse nodes to account for lightweight structural overhead.
        stack: List[Any] = [root]
        node_count = 0
        while stack:
            node = stack.pop()
            node_count += 1
            children = getattr(node, "children", None)
            if children:
                stack.extend(children)

        # Approximate 32 bytes per node for indexes/links.
        size_bytes += node_count * 32
        return size_bytes

    def _approximate_object_bytes(self, obj: Any) -> int:
        """Fallback: recursively approximate Python container sizes."""
        visited: set[int] = set()

        def _estimate(item: Any) -> int:
            obj_id = id(item)
            if obj_id in visited:
                return 0
            visited.add(obj_id)

            if isinstance(item, np.ndarray):
                return int(item.nbytes)
            if isinstance(item, dict):
                size = sys.getsizeof(item)
                for key, value in item.items():
                    size += _estimate(key)
                    size += _estimate(value)
                return size
            if isinstance(item, (list, tuple, set)):
                size = sys.getsizeof(item)
                for elem in item:
                    size += _estimate(elem)
                return size
            try:
                return sys.getsizeof(item)
            except Exception:
                return 0

        return _estimate(obj)

    def _save_algorithm_results(self, name: str, metrics: Dict[str, Any]) -> None:
        path = os.path.join(self.output_dir, f"{name}_results.json")
        with open(path, "w") as handle:
            json.dump(metrics, handle, indent=2)

    def _save_combined_results(self) -> None:
        combined_path = os.path.join(
            self.output_dir, f"{self.config.output_prefix}_all_results.json"
        )
        with open(combined_path, "w") as handle:
            json.dump(self.results, handle, indent=2)

        config_path = os.path.join(
            self.output_dir, f"{self.config.output_prefix}_{self.experiment_id}_config.yaml"
        )
        with open(config_path, "w") as handle:
            yaml.safe_dump(self.config.to_dict(), handle)

    def _generate_plots(self, evaluator: Evaluator) -> None:
        plots_dir = os.path.join(self.output_dir, f"plots_{self.experiment_id}")
        os.makedirs(plots_dir, exist_ok=True)

        plot_path = os.path.join(plots_dir, "recall_vs_qps.png")
        evaluator.plot_recall_vs_qps(
            output_file=plot_path,
            title_suffix=self.config.dataset,
        )

        dataset_name = str(self.config.dataset).lower()
        if "glove" in dataset_name:
            operations_plot_path = os.path.join(plots_dir, "operations_vs_recall.png")
            evaluator.plot_operations_vs_recall(
                output_file=operations_plot_path,
                title_suffix=self.config.dataset,
            )
