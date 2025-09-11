"""Utility functions for environment compatibility checks.

This module currently provides helpers to ensure that the
benchmark uses ARM-compatible BLAS libraries when executed on
Apple Silicon or other ARM-based machines.
"""

from __future__ import annotations

import platform
import subprocess
import sys


def ensure_arm_compatible_blas() -> None:
    """Ensure numeric libraries are ARM compatible.

    On Apple Silicon or other ARM architectures, using Intel's MKL
    (which targets x86 instruction sets) can cause segmentation faults.
    This function detects such a configuration and attempts to reinstall
    ``numpy``, ``scipy`` and ``scikit-learn`` with OpenBLAS or Apple's
    Accelerate backend via ``pip`` if MKL is detected.

    The function performs a best-effort fix: if the installation fails the
    benchmark will continue to run, but a warning will be emitted.
    """
    arch = platform.machine().lower()
    if arch not in {"arm64", "aarch64"}:
        return

    try:
        import numpy as np  # type: ignore

        info = {}
        for key in ["blas_opt_info", "lapack_opt_info"]:
            try:
                info.update(np.__config__.get_info(key))
            except Exception:
                continue

        mkl_detected = any("mkl" in str(v).lower() for v in info.values())
        if mkl_detected:
            print(
                "Detected Intel MKL on an ARM machine; attempting to install "
                "OpenBLAS/Accelerate based wheels for numpy, scipy and scikit-learn..."
            )
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--upgrade",
                        "--force-reinstall",
                        "numpy",
                        "scipy",
                        "scikit-learn",
                    ],
                    check=True,
                )
                print("Reinstallation completed. Please rerun the benchmark.")
                raise SystemExit(0)
            except subprocess.CalledProcessError as exc:  # pragma: no cover - best effort
                print(f"Warning: failed to install ARM compatible libraries: {exc}")
    except Exception as exc:  # pragma: no cover - best effort
        print(f"Warning: could not check BLAS configuration: {exc}")
