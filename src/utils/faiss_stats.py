"""Utilities to read FAISS runtime stats for distance evaluations.

This module abstracts over FAISS global stats objects to provide a safe API
for resetting and reading distance evaluation counters (ndis) across IVF and
HNSW index families. It also classifies common index types and annotates
whether counters correspond to raw vector-to-vector distances or code-distance
evaluations (for compressed indexes).
"""
from __future__ import annotations

from typing import Optional, Tuple

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
except Exception:  # pragma: no cover - allow importing without faiss installed
    faiss = None  # type: ignore


def _downcast(index):  # pragma: no cover - thin wrapper
    if faiss is None:
        return index
    try:
        return faiss.downcast_index(index)
    except Exception:
        return index


def classify_index(index) -> str:
    """Return a coarse type label for a FAISS index.

    Possible return values: flat | hnsw | ivf_flat | ivf_compressed | ivf_other | unknown
    """
    if faiss is None:
        return "unknown"
    idx = _downcast(index)
    name = type(idx).__name__
    if name in ("IndexFlat", "IndexFlatL2", "IndexFlatIP"):
        return "flat"
    if name in ("IndexHNSW", "IndexHNSWFlat", "IndexHNSW2Level"):
        return "hnsw"
    if name in ("IndexIVFFlat",):
        return "ivf_flat"
    if name in (
        "IndexIVFPQ",
        "IndexIVFScalarQuantizer",
        "IndexIVFResidualQuantizer",
        "IndexIVFProdQuantizer",
    ):
        return "ivf_compressed"
    # Feature-based fallbacks
    if hasattr(idx, "hnsw"):
        return "hnsw"
    try:
        if faiss.extract_index_ivf(index) is not None:
            return "ivf_other"
    except Exception:
        pass
    return "unknown"


def _maybe_enable_stats() -> None:  # pragma: no cover - best-effort
    if faiss is None:
        return
    for flag in ("indexIVF_stats_enabled", "hnsw_stats_enabled"):
        if hasattr(faiss.cvar, flag):
            try:
                setattr(faiss.cvar, flag, True)
            except Exception:
                pass


def reset_runtime_stats(index) -> None:
    """Reset FAISS global runtime stats for IVF and HNSW families.

    This is safe to call before any search on a FAISS index. For mixed indexes
    (e.g., IVF with HNSW coarse quantizer), both families are reset to ensure
    counters reflect only the upcoming search calls.
    """
    if faiss is None:
        return
    _maybe_enable_stats()
    # Reset both when available to be robust to hybrid indices
    if hasattr(faiss.cvar, "indexIVF_stats"):
        try:
            faiss.cvar.indexIVF_stats.reset()
        except Exception:
            pass
    if hasattr(faiss.cvar, "hnsw_stats"):
        try:
            faiss.cvar.hnsw_stats.reset()
        except Exception:
            pass


def read_distance_counts(index) -> Tuple[Optional[int], Optional[int]]:
    """Return (ivf_ndis, hnsw_ndis) from FAISS global stats.

    Values are None if unavailable for the current build or index type. For
    Flat indexes, both will be None; callers should compute nq * ntotal.
    """
    if faiss is None:
        return (None, None)
    ivf_ndis = None
    hnsw_ndis = None
    try:
        if hasattr(faiss.cvar, "indexIVF_stats") and hasattr(faiss.cvar.indexIVF_stats, "ndis"):
            ivf_ndis = int(faiss.cvar.indexIVF_stats.ndis)
    except Exception:
        ivf_ndis = None
    try:
        if hasattr(faiss.cvar, "hnsw_stats") and hasattr(faiss.cvar.hnsw_stats, "ndis"):
            hnsw_ndis = int(faiss.cvar.hnsw_stats.ndis)
    except Exception:
        hnsw_ndis = None
    return (ivf_ndis, hnsw_ndis)


def counters_are_v2v(index) -> Optional[bool]:
    """Whether FAISS ndis counters correspond to raw vector-to-vector distances.

    Returns True for: flat, ivf_flat, hnsw
    Returns False for: ivf_compressed, ivf_other (code distances)
    Returns None for unknown
    """
    kind = classify_index(index)
    if kind in ("flat", "ivf_flat", "hnsw"):
        return True
    if kind in ("ivf_compressed", "ivf_other"):
        return False
    return None
