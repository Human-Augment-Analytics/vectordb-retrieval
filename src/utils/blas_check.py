import platform
import logging

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy required elsewhere
    np = None


def ensure_arm_compatible_blas() -> None:
    """Ensure BLAS backend is compatible with ARM architectures.

    On Apple Silicon and other ARM platforms, using Intel's MKL backend may
    cause segmentation faults. This function checks the current architecture
    and the BLAS libraries linked with NumPy. If an incompatible combination
    is detected, an informative ``RuntimeError`` is raised so that the user can
    install an ARM-optimized build (OpenBLAS or Apple's Accelerate framework).
    """
    machine = platform.machine().lower()
    if machine not in {"arm64", "aarch64"} or np is None:
        return

    info = np.__config__.get_info("blas_opt_info")
    libraries = [lib.lower() for lib in info.get("libraries", [])]

    if any("mkl" in lib for lib in libraries):
        raise RuntimeError(
            "Intel MKL BLAS detected on ARM architecture. "
            "Please reinstall NumPy/Scipy with OpenBLAS or Apple's Accelerate." 
        )

    if not any(lib in libraries for lib in {"openblas", "accelerate", "veclib"}):
        logging.warning(
            "BLAS backend %s may not be optimized for ARM. "
            "Consider using OpenBLAS or Apple's Accelerate.", libraries
        )
