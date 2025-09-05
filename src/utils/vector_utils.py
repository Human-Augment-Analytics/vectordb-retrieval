import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Literal

def normalize_vectors(vectors: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Normalize vectors to unit length.

    Args:
        vectors: Array of vectors to normalize
        axis: Axis along which to normalize

    Returns:
        Normalized vectors
    """
    norms = np.linalg.norm(vectors, axis=axis, keepdims=True)
    # Avoid division by zero
    norms = np.maximum(norms, 1e-10)
    return vectors / norms

def compute_distance(x: np.ndarray, y: np.ndarray, metric: str = "l2") -> np.ndarray:
    """
    Compute distance between vectors using specified metric.

    Args:
        x: First set of vectors (m, dimension)
        y: Second set of vectors (n, dimension)
        metric: Distance metric ('l2', 'cosine', 'dot')

    Returns:
        Distance matrix (m, n)
    """
    if metric == "l2":
        # Euclidean distance using broadcasting
        m = x.shape[0]
        n = y.shape[0]
        xx = np.sum(x**2, axis=1).reshape(m, 1)
        yy = np.sum(y**2, axis=1).reshape(1, n)
        xy = np.dot(x, y.T)
        distances = np.sqrt(xx + yy - 2 * xy)
    elif metric == "cosine":
        # Cosine distance = 1 - cosine similarity
        x_norm = normalize_vectors(x)
        y_norm = normalize_vectors(y)
        distances = 1 - np.dot(x_norm, y_norm.T)
    elif metric == "dot":
        # Negative dot product (higher is better, so we negate for distance)
        distances = -np.dot(x, y.T)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return distances

def random_unit_vectors(n: int, dim: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate random unit vectors.

    Args:
        n: Number of vectors to generate
        dim: Dimensionality of vectors
        seed: Random seed for reproducibility

    Returns:
        Array of random unit vectors (n, dim)
    """
    if seed is not None:
        np.random.seed(seed)

    vectors = np.random.randn(n, dim).astype(np.float32)
    return normalize_vectors(vectors)

def vector_to_string(vector: np.ndarray, precision: int = 4) -> str:
    """
    Convert a vector to a compact string representation.

    Args:
        vector: Vector to convert
        precision: Number of decimal places

    Returns:
        String representation of the vector
    """
    return '[' + ', '.join(f"{x:.{precision}f}" for x in vector) + ']'
