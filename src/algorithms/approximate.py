import numpy as np
from typing import Dict, List, Any, Optional
import logging
from .base import BaseAlgorithm

class ApproximateSearch(BaseAlgorithm):
    """
    Approximate nearest neighbor search using IVF (Inverted File Index).
    This is a simplified implementation for educational purposes.
    """

    def __init__(self, dimension: int, index_type: str = "IVF100,Flat", 
                 metric: str = "l2", nprobe: int = 10, **kwargs):
        """
        Initialize the approximate search algorithm.

        Args:
            dimension: Dimensionality of vectors
            index_type: Type of index (e.g., "IVF100,Flat")
            metric: Distance metric to use (e.g., "l2", "ip")
            nprobe: Number of clusters to visit during search
            **kwargs: Additional parameters
        """
        super().__init__(dimension, metric, index_type=index_type, nprobe=nprobe, **kwargs)

        # Parse index type
        parts = index_type.split(",")
        if len(parts) != 2 or not parts[0].startswith("IVF"):
            raise ValueError(f"Invalid index_type: {index_type}. Expected format: IVF<nlist>,<quantizer>")

        try:
            self.nlist = int(parts[0][3:])  # Number of clusters
        except ValueError:
            raise ValueError(f"Invalid nlist in {parts[0]}. Expected integer after 'IVF'.")

        self.quantizer = parts[1]  # Quantizer type (Flat, PQ, etc.)
        self.nprobe = nprobe  # Number of clusters to visit during search

        # Initialize index components
        self.centroids = None  # Cluster centroids
        self.assignments = None  # Vector to cluster assignments
        self.vectors = None  # Original vectors

        self.logger = logging.getLogger("approximate_search")

    def build_index(self, vectors: np.ndarray) -> None:
        """
        Build the search index with IVF clustering.

        Args:
            vectors: Vector data to index (n_vectors × dimension)
        """
        self.vectors = vectors
        n_vectors = vectors.shape[0]

        # Adjust nlist if needed
        self.nlist = min(self.nlist, n_vectors // 10)
        if self.nlist < 1:
            self.nlist = 1

        self.logger.info(f"Building IVF index with {self.nlist} clusters")

        # Simple k-means clustering to create centroids
        # For simplicity, we'll just use a random subset of vectors as initial centroids
        np.random.seed(42)  # For reproducibility
        centroid_indices = np.random.choice(n_vectors, self.nlist, replace=False)
        self.centroids = vectors[centroid_indices].copy()

        # Assign vectors to clusters
        self.assignments = np.zeros(n_vectors, dtype=np.int32)

        # Simple k-means iterations
        MAX_ITERATIONS = 10
        for iteration in range(MAX_ITERATIONS):
            # Assign vectors to nearest centroid
            for i in range(n_vectors):
                distances = np.linalg.norm(self.centroids - vectors[i], axis=1)
                self.assignments[i] = np.argmin(distances)

            # Update centroids
            new_centroids = np.zeros_like(self.centroids)
            counts = np.zeros(self.nlist, dtype=np.int32)

            for i in range(n_vectors):
                cluster = self.assignments[i]
                new_centroids[cluster] += vectors[i]
                counts[cluster] += 1

            # Avoid division by zero
            for j in range(self.nlist):
                if counts[j] > 0:
                    new_centroids[j] /= counts[j]
                else:
                    # If a cluster is empty, keep the old centroid
                    new_centroids[j] = self.centroids[j]

            # Check for convergence
            centroid_change = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids

            if centroid_change < 0.001:
                self.logger.info(f"K-means converged after {iteration+1} iterations")
                break

        self.logger.info(f"Index built with {self.nlist} clusters")

    def search(self, query: np.ndarray, k: int) -> np.ndarray:
        """
        Find the k nearest neighbors for a single query vector.

        Args:
            query: Query vector
            k: Number of nearest neighbors to retrieve

        Returns:
            Array of indices of the nearest neighbors
        """
        if self.centroids is None or self.vectors is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Find nearest clusters
        centroid_distances = np.linalg.norm(self.centroids - query, axis=1)
        nearest_clusters = np.argsort(centroid_distances)[:self.nprobe]

        # Collect candidate vectors from selected clusters
        candidate_indices = []
        for cluster in nearest_clusters:
            cluster_members = np.where(self.assignments == cluster)[0]
            candidate_indices.extend(cluster_members)

        if not candidate_indices:
            # If no candidates found, fall back to some random vectors
            candidate_indices = np.random.choice(len(self.vectors), min(k*10, len(self.vectors)), replace=False)

        # Compute distances for candidates
        candidates = self.vectors[candidate_indices]
        distances = np.linalg.norm(candidates - query, axis=1)

        # Get top k indices among candidates
        k = min(k, len(distances))  # Handle case where k > number of candidates
        top_k_local = np.argsort(distances)[:k]

        # Map back to original indices
        top_k_global = np.array([candidate_indices[i] for i in top_k_local])

        return top_k_global

    def get_memory_usage(self) -> float:
        """
        Get the memory usage of the index in MB.

        Returns:
            Memory usage in MB
        """
        if self.vectors is None or self.centroids is None:
            return 0.0

        # Calculate memory usage: vectors + centroids + assignments
        vector_bytes = self.vectors.nbytes
        centroid_bytes = self.centroids.nbytes
        assignment_bytes = self.assignments.nbytes

        total_bytes = vector_bytes + centroid_bytes + assignment_bytes
        return total_bytes / (1024 * 1024)
