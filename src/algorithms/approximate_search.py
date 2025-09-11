import faiss
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from .base_algorithm import BaseAlgorithm

class ApproximateSearch(BaseAlgorithm):
    """
    Approximate nearest neighbor search using Faiss.
    This class supports various index types available in Faiss.
    """

    def __init__(self, name: str, dimension: int, index_type: str, metric: str = 'l2', **kwargs):
        """
        Initialize the ApproximateSearch algorithm.

        Args:
            name: Name of the algorithm instance
            dimension: Dimensionality of the vectors
            index_type: Faiss index string (e.g., 'IVF100,Flat')
            metric: Distance metric ('l2' or 'ip')
            **kwargs: Additional parameters for the index
        """
        super().__init__(name, dimension, **kwargs)
        self.index_type = index_type
        self.metric = faiss.METRIC_L2 if metric == 'l2' else faiss.METRIC_INNER_PRODUCT
        self.index = None

    def build_index(self, vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Build the Faiss index.

        Args:
            vectors: Vectors to index (n_vectors, dimension)
            metadata: Optional metadata (not used by this algorithm)
        """
        self.vectors = vectors.astype(np.float32)
        
        # Create the index
        self.index = faiss.index_factory(self.dimension, self.index_type, self.metric)
        
        # Train the index if necessary
        if not self.index.is_trained:
            self.index.train(self.vectors)
            
        # Add vectors to the index
        self.index.add(self.vectors)
        self.index_built = True

        # Set search-time parameters if provided
        if 'nprobe' in self.config:
            self.index.nprobe = self.config['nprobe']

    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the index for the k nearest neighbors.

        Args:
            query: Query vector (dimension,)
            k: Number of nearest neighbors to return

        Returns:
            Tuple of (distances, indices)
        """
        if not self.index_built:
            raise RuntimeError("Index has not been built yet.")
            
        # Faiss expects a 2D array for queries
        query_vector = np.array([query], dtype=np.float32)
        distances, indices = self.index.search(query_vector, k)
        
        return distances[0], indices[0]

    def batch_search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch search for k nearest neighbors.

        Args:
            queries: Query vectors (n_queries, dimension)
            k: Number of nearest neighbors to return

        Returns:
            Tuple of (distances, indices)
        """
        if not self.index_built:
            raise RuntimeError("Index has not been built yet.")
            
        return self.index.search(queries.astype(np.float32), k)
