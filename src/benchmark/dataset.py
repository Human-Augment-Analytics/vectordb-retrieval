import numpy as np
import os
import pickle
import requests
import zipfile
from typing import Dict, List, Tuple, Any, Optional
import logging
from tqdm import tqdm

class Dataset:
    """
    Class for handling benchmark datasets for vector retrieval.
    """

    AVAILABLE_DATASETS = {
        "sift1m": {
            "description": "SIFT1M dataset with 1M 128-dimensional SIFT vectors",
            "dimensions": 128,
            "size": 1_000_000,
            "url": "http://corpus-texmex.irisa.fr/"
        },
        "glove50": {
            "description": "GloVe word embeddings from glove.6B.zip (50d)",
            "dimensions": 50,
            "size": 400000,
            "url": "http://nlp.stanford.edu/data/glove.6B.zip"
        },
        "random": {
            "description": "Randomly generated vectors for testing",
            "dimensions": 128,
            "size": 10_000,
            "url": None
        }
    }

    def __init__(self, name: str, data_dir: str = "data"):
        """
        Initialize the dataset.

        Args:
            name: Name of the dataset
            data_dir: Directory to store the dataset
        """
        if name not in self.AVAILABLE_DATASETS and name != "random":
            raise ValueError(f"Unknown dataset: {name}. Available datasets: {list(self.AVAILABLE_DATASETS.keys())}")

        self.name = name
        self.data_dir = os.path.join(data_dir, name)
        self.info = self.AVAILABLE_DATASETS.get(name, {})

        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)

        self.train_vectors = None
        self.test_vectors = None
        self.ground_truth = None
        self.loaded = False

    def download(self):
        """
        Download the dataset if it's not already available.
        """
        if self.name == "random":
            return

        # Implement dataset-specific download logic
        if self.name == "sift1m":
            self._download_sift1m()
        elif self.name == "glove50":
            self._download_glove()

    def _download_sift1m(self):
        """
        Download SIFT1M dataset from the official source.
        """
        base_url = "ftp://ftp.irisa.fr/local/texmex/corpus/"
        files = [
            "sift_base.fvecs",  # 1M base vectors
            "sift_query.fvecs", # 10K query vectors
            "sift_groundtruth.ivecs"  # Ground truth
        ]
        
        for filename in files:
            file_path = os.path.join(self.data_dir, filename)
            if os.path.exists(file_path):
                print(f"File {filename} already exists, skipping download")
                continue
                
            url = base_url + filename
            print(f"Downloading {filename} from {url}")
            
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                with open(file_path, 'wb') as f, tqdm(
                    desc=filename,
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
                            
                print(f"Successfully downloaded {filename}")
                
            except Exception as e:
                print(f"Error downloading {filename}: {str(e)}")
                if os.path.exists(file_path):
                    os.remove(file_path)
                raise

    def _download_glove(self):
        """
        Download GloVe 50d dataset.
        """
        url = "http://nlp.stanford.edu/data/glove.6B.zip"
        zip_path = os.path.join(self.data_dir, "glove.6B.zip")
        
        if os.path.exists(zip_path):
            print(f"File {zip_path} already exists, skipping download.")
        else:
            print(f"Downloading GloVe dataset from {url}")
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                with open(zip_path, 'wb') as f, tqdm(
                    desc="glove.6B.zip",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
                print(f"Successfully downloaded {zip_path}")
            except Exception as e:
                print(f"Error downloading GloVe dataset: {str(e)}")
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                raise

        # Unzip the file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            print(f"Extracting {zip_path}...")
            zip_ref.extractall(self.data_dir)
            print("Extraction complete.")

    def load(self, force_download: bool = False) -> None:
        """
        Load the dataset into memory.

        Args:
            force_download: Whether to force download even if files exist
        """
        if self.loaded:
            return

        cache_file = os.path.join(self.data_dir, f"{self.name}_processed.pkl")

        # Check if processed file exists
        if os.path.exists(cache_file) and not force_download:
            print(f"Loading processed dataset from {cache_file}")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                self.train_vectors = data['train']
                self.test_vectors = data['test']
                self.ground_truth = data['ground_truth']
        else:
            # For random dataset, generate it
            if self.name == "random":
                print("Generating random dataset")
                self._generate_random_dataset()
            else:
                # Download and process real dataset
                self.download()
                self._process_dataset()

            # Save processed data
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'train': self.train_vectors,
                    'test': self.test_vectors,
                    'ground_truth': self.ground_truth
                }, f)

        self.loaded = True
        print(f"Dataset loaded: {self.name}")
        print(f"  Train vectors: {self.train_vectors.shape}")
        print(f"  Test vectors: {self.test_vectors.shape}")
        print(f"  Ground truth: {self.ground_truth.shape}")

    def _generate_random_dataset(self, dimensions: int = 128, 
                                 train_size: int = 10_000, 
                                 test_size: int = 1_000,
                                 k: int = 100) -> None:
        """
        Generate a random dataset for testing.

        Args:
            dimensions: Dimensionality of vectors
            train_size: Number of training vectors
            test_size: Number of test vectors
            k: Number of ground truth neighbors
        """
        np.random.seed(42)  # For reproducibility

        # Generate random vectors
        self.train_vectors = np.random.randn(train_size, dimensions).astype(np.float32)
        self.test_vectors = np.random.randn(test_size, dimensions).astype(np.float32)

        # Compute ground truth with brute force
        self.ground_truth = np.zeros((test_size, k), dtype=np.int32)
        distances = np.zeros((test_size, train_size))

        for i in tqdm(range(test_size), desc="Computing ground truth"):
            # L2 distance
            distances[i] = np.linalg.norm(self.train_vectors - self.test_vectors[i:i+1], axis=1)
            self.ground_truth[i] = np.argsort(distances[i])[:k]

    def _process_dataset(self) -> None:
        """
        Process the raw dataset files into numpy arrays.
        This needs to be implemented specifically for each dataset.
        """
        # Implementation depends on dataset format
        if self.name == "sift1m":
            self._process_sift1m()
        elif self.name == "glove50":
            self._process_glove()

    def _read_fvecs(self, filename: str) -> np.ndarray:
        """
        Read .fvecs file format used by SIFT1M dataset.
        
        Args:
            filename: Path to .fvecs file
            
        Returns:
            numpy array of vectors
        """
        with open(filename, 'rb') as f:
            # Read dimension (first 4 bytes)
            dim = np.frombuffer(f.read(4), dtype=np.int32)[0]
            
            # Reset to beginning and read all data
            f.seek(0)
            data = np.frombuffer(f.read(), dtype=np.int32)
            
            # Reshape: each vector is (dim+1) int32 values (dim + dimension value)
            n_vectors = len(data) // (dim + 1)
            data = data.reshape(n_vectors, dim + 1)
            
            # Remove dimension column and convert to float32
            vectors = data[:, 1:].astype(np.float32)
            
            return vectors

    def _read_ivecs(self, filename: str) -> np.ndarray:
        """
        Read .ivecs file format used for ground truth.
        
        Args:
            filename: Path to .ivecs file
            
        Returns:
            numpy array of integer vectors
        """
        with open(filename, 'rb') as f:
            # Read dimension (first 4 bytes)
            dim = np.frombuffer(f.read(4), dtype=np.int32)[0]
            
            # Reset to beginning and read all data
            f.seek(0)
            data = np.frombuffer(f.read(), dtype=np.int32)
            
            # Reshape: each vector is (dim+1) int32 values
            n_vectors = len(data) // (dim + 1)
            data = data.reshape(n_vectors, dim + 1)
            
            # Remove dimension column
            vectors = data[:, 1:]
            
            return vectors

    def _process_sift1m(self) -> None:
        """
        Process SIFT1M dataset files.
        """
        print("Processing SIFT1M dataset...")
        
        # Read base vectors (training data)
        base_file = os.path.join(self.data_dir, "sift_base.fvecs")
        if not os.path.exists(base_file):
            raise FileNotFoundError(f"Base vectors file not found: {base_file}")
        
        print("Reading base vectors...")
        self.train_vectors = self._read_fvecs(base_file)
        
        # Read query vectors (test data)
        query_file = os.path.join(self.data_dir, "sift_query.fvecs")
        if not os.path.exists(query_file):
            raise FileNotFoundError(f"Query vectors file not found: {query_file}")
            
        print("Reading query vectors...")
        self.test_vectors = self._read_fvecs(query_file)
        
        # Read ground truth
        gt_file = os.path.join(self.data_dir, "sift_groundtruth.ivecs")
        if not os.path.exists(gt_file):
            raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
            
        print("Reading ground truth...")
        self.ground_truth = self._read_ivecs(gt_file)
        
        print(f"SIFT1M dataset processed:")
        print(f"  Base vectors: {self.train_vectors.shape}")
        print(f"  Query vectors: {self.test_vectors.shape}")
        print(f"  Ground truth: {self.ground_truth.shape}")

    def _process_glove(self) -> None:
        """
        Process GloVe dataset.
        """
        txt_file = os.path.join(self.data_dir, "glove.6B.50d.txt")
        if not os.path.exists(txt_file):
            raise FileNotFoundError(f"glove.6B.50d.txt not found. Please run download first.")

        print("Processing GloVe dataset from text file...")
        vectors = []
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading GloVe vectors"):
                parts = line.split()
                vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                vectors.append(vector)
        
        all_vectors = np.array(vectors)
        
        # Use a random subset as queries and the rest as the base set
        np.random.seed(42)
        test_size = 1000
        if all_vectors.shape[0] <= test_size:
            raise ValueError("Dataset is too small to create a test split.")
            
        test_indices = np.random.choice(all_vectors.shape[0], test_size, replace=False)
        train_indices = np.setdiff1d(np.arange(all_vectors.shape[0]), test_indices)

        self.test_vectors = all_vectors[test_indices]
        self.train_vectors = all_vectors[train_indices]

        # Compute ground truth for the test set
        print("Computing ground truth for GloVe...")
        k = 100  # Number of nearest neighbors for ground truth
        self.ground_truth = np.zeros((self.test_vectors.shape[0], k), dtype=np.int32)
        for i in tqdm(range(self.test_vectors.shape[0]), desc="Computing GloVe ground truth"):
            distances = np.linalg.norm(self.train_vectors - self.test_vectors[i], axis=1)
            self.ground_truth[i] = np.argsort(distances)[:k]

    def get_train_test_split(self, test_ratio: float = 0.1, 
                           seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split the dataset into training and testing sets.

        Args:
            test_ratio: Ratio of data to use for testing
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_vectors, test_vectors)
        """
        if not self.loaded:
            self.load()

        return self.train_vectors, self.test_vectors

    def get_ground_truth(self) -> np.ndarray:
        """
        Get the ground truth nearest neighbors for test vectors.

        Returns:
            Array of indices of ground truth neighbors
        """
        if not self.loaded:
            self.load()

        return self.ground_truth
