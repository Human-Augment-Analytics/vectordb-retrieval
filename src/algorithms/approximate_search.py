import numpy as np
import faiss
from typing import List, Dict, Any, Tuple, Optional
from .base_algorithm import BaseAlgorithm

class ApproximateSearch(BaseAlgorithm):
    """
    Approximate search algorithm using FAISS.
    This implements various approximate nearest neighbor search techniques.
    """

    def __init__(self, dimension: int, index_type: str = "IVF100,Flat", metric: str = "l2", **kwargs):
        """
        Initialize the approximate search algorithm.

        Args:
            dimension: Dimensionality of the vectors
            index_type: Type of FAISS index to use
            metric: Distance metric ('l2', 'cosine', 'dot')
            **kwargs: Additional parameters like nprobe
        """
        super().__init__("ApproximateSearch", dimension, index_type=index_type, metric=metric, **kwargs)
        self.index_type = index_type
        self.metric = metric
        self.nprobe = kwargs.get('nprobe', 10)
        self.index = None

    def build_index(self, vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Build FAISS index for the given vectors.

        Args:
            vectors: Vectors to index (n_vectors, dimension)
            metadata: Optional metadata for each vector
        """
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected vectors of dimension {self.dimension}, got {vectors.shape[1]}")

        n_vectors = vectors.shape[0]
        self.vectors = vectors
        self.metadata = metadata

        # Determine the index based on the metric
        if self.metric == "cosine":
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(vectors)
            # Use inner product for normalized vectors
            metric_type = faiss.METRIC_INNER_PRODUCT
        elif self.metric == "dot":
            metric_type = faiss.METRIC_INNER_PRODUCT
        else:  # Default to L2
            metric_type = faiss.METRIC_L2

        # Create the index
        if self.index_type == "Flat":
            if metric_type == faiss.METRIC_INNER_PRODUCT:
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type.startswith("IVF"):
            # Extract the number of centroids from the index type string
            parts = self.index_type.split(',')
            nlist = int(parts[0].replace('IVF', ''))

            # Create the base index
            quantizer = faiss.IndexFlatL2(self.dimension)
            if "PQ" in self.index_type:
                # Product Quantization index
                m = int(parts[1].replace('PQ', ''))  # Number of subquantizers
                self.index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, m, 8, metric_type)
            else:
                # Standard IVF index
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, metric_type)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

        # Train the index if needed
        if not self.index.is_trained and n_vectors > 0:
            self.index.train(vectors)

        # Set the number of probes for search
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.nprobe

        # Add vectors to the index
        self.index.add(vectors)
        self.index_built = True

    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the index for the k nearest neighbors of the query vector.

        Args:
            query: Query vector (dimension,)
            k: Number of nearest neighbors to return

        Returns:
            Tuple of (distances, indices)
        """
        if not self.index_built:
            raise RuntimeError("Index not built. Call build_index() first.")

        # Reshape query to (1, dimension) if needed
        if len(query.shape) == 1:
            query = query.reshape(1, -1)

        # Normalize query for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(query)

        # Perform the search
        distances, indices = self.index.search(query, k)

        return distances[0], indices[0]

    def batch_search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the index for the k nearest neighbors of multiple query vectors.

        Args:
            queries: Query vectors (n_queries, dimension)
            k: Number of nearest neighbors to return

        Returns:
            Tuple of (distances, indices)
        """
        if not self.index_built:
            raise RuntimeError("Index not built. Call build_index() first.")

        # Normalize queries for cosine similarity
        if self.metric == "cosine":
            queries_copy = queries.copy()
            faiss.normalize_L2(queries_copy)
            distances, indices = self.index.search(queries_copy, k)
        else:
            distances, indices = self.index.search(queries, k)

        return distances, indices
