"""Pytest fixtures and helpers for the vectordb-retrieval project."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is importable for test modules.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def is_faiss_available() -> bool:
    """Return True if faiss is importable."""
    try:
        import faiss  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True
