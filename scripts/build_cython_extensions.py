"""Build optional Cython extensions for algorithm accelerators."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from setuptools import Extension, setup


def main() -> None:
    try:
        from Cython.Build import cythonize
    except ImportError as exc:
        raise SystemExit("Cython is not installed. Run `pip install cython`.") from exc

    repo_root = Path(__file__).resolve().parents[1]
    ext_modules = [
        Extension(
            name="src.algorithms._covertree_cython",
            sources=[str(repo_root / "src" / "algorithms" / "_covertree_cython.pyx")],
            include_dirs=[np.get_include()],
        )
    ]

    setup(
        name="vectordb-retrieval-cython-ext",
        ext_modules=cythonize(ext_modules, compiler_directives={"language_level": "3"}),
        script_args=["build_ext", "--inplace"],
    )


if __name__ == "__main__":
    main()
