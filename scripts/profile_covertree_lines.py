#!/usr/bin/env python
"""
Line-by-line profiler for CoverTreeV2_2.

Example:
    .\.venv\Scripts\python.exe scripts/profile_covertree_lines.py \
        --train-size 20000 --query-size 256 --dimension 50 --metric cosine --k 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    from line_profiler import LineProfiler
except ImportError as exc:  # pragma: no cover - dependency/usage guard
    raise SystemExit(
        "line_profiler is not installed. Install with:\n"
        r".\.venv\Scripts\python.exe -m pip install line_profiler"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.algorithms.covertree import CoverTreeV2_2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile CoverTreeV2_2 line-by-line.")
    parser.add_argument("--train-size", type=int, default=20000, help="Number of training vectors.")
    parser.add_argument("--query-size", type=int, default=256, help="Number of query vectors.")
    parser.add_argument("--dimension", type=int, default=50, help="Vector dimensionality.")
    parser.add_argument("--k", type=int, default=200, help="Top-k neighbors to request.")
    parser.add_argument(
        "--metric",
        type=str,
        default="l2",
        choices=["l2", "euclidean", "cosine", "dot", "ip", "inner_product"],
        help="Distance metric for CoverTreeV2_2.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--profile-phase",
        type=str,
        default="both",
        choices=["build", "search", "both"],
        help="Whether to profile build phase, search phase, or both.",
    )
    parser.add_argument(
        "--search-runs",
        type=int,
        default=1,
        help="How many times to run batch_search (use >1 to amplify search hotspots).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="covertree_line_profile.txt",
        help="Text output path for line-profiler report.",
    )
    parser.add_argument(
        "--dump-raw",
        type=str,
        default="covertree_line_profile.lprof",
        help="Raw line-profiler stats file (.lprof).",
    )
    return parser.parse_args()


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def generate_data(
    train_size: int,
    query_size: int,
    dimension: int,
    metric: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train = rng.standard_normal((train_size, dimension)).astype(np.float32)
    queries = rng.standard_normal((query_size, dimension)).astype(np.float32)

    if metric == "cosine":
        train = _normalize_rows(train).astype(np.float32)
        queries = _normalize_rows(queries).astype(np.float32)

    return train, queries


def build_profiler() -> LineProfiler:
    profiler = LineProfiler()

    # Public API methods
    profiler.add_function(CoverTreeV2_2.build_index)
    profiler.add_function(CoverTreeV2_2.batch_search)
    profiler.add_function(CoverTreeV2_2.search)

    # Internal hotspots
    profiler.add_function(CoverTreeV2_2._insert_index)
    profiler.add_function(CoverTreeV2_2._insert)
    profiler.add_function(CoverTreeV2_2._search_exact_k)
    profiler.add_function(CoverTreeV2_2._compute_distance_batch_to_1)
    profiler.add_function(CoverTreeV2_2._prepare_query)
    profiler.add_function(CoverTreeV2_2._normalize_vectors)

    return profiler


def main() -> int:
    args = parse_args()

    if args.train_size <= 0 or args.query_size <= 0 or args.dimension <= 0:
        raise ValueError("train-size, query-size, and dimension must be positive integers.")
    if args.search_runs <= 0:
        raise ValueError("search-runs must be >= 1.")

    train, queries = generate_data(
        train_size=args.train_size,
        query_size=args.query_size,
        dimension=args.dimension,
        metric=args.metric,
        seed=args.seed,
    )

    effective_k = max(1, min(args.k, args.train_size))
    if effective_k != args.k:
        print(f"Adjusted k from {args.k} to {effective_k} to match train-size bounds.")

    tree = CoverTreeV2_2(
        name="covertree_line_profile",
        dimension=args.dimension,
        metric=args.metric,
    )
    profiler = build_profiler()

    should_profile_build = args.profile_phase in ("build", "both")
    should_profile_search = args.profile_phase in ("search", "both")

    if should_profile_build:
        profiled_build = profiler(tree.build_index)
        profiled_build(train)
    else:
        tree.build_index(train)

    if should_profile_search:
        profiled_batch_search = profiler(tree.batch_search)
        for _ in range(args.search_runs):
            profiled_batch_search(queries, k=effective_k)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# CoverTreeV2_2 Line Profiling Report\n")
        handle.write(f"phase={args.profile_phase}\n")
        handle.write(
            f"metric={args.metric} train_size={args.train_size} query_size={args.query_size} "
            f"dimension={args.dimension} k={effective_k} search_runs={args.search_runs}\n\n"
        )
        profiler.print_stats(stream=handle)

    raw_path = Path(args.dump_raw)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(str(raw_path))

    print(f"Line profile text report written to: {output_path}")
    print(f"Raw line profile data written to: {raw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
