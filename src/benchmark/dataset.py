import json
import hashlib
import logging
import os
import pickle
import zipfile
from ftplib import FTP
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import requests
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
        },
        "msmarco": {
            "description": "MS MARCO passage ranking dataset (TF-IDF projection)",
            "dimensions": None,
            "size": None,
            "url": None
        }
    }

    def __init__(self, name: str, data_dir: str = "data", options: Optional[Dict[str, Any]] = None):
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
        self.options = options or {}

        cache_dir_option = self.options.get("cache_dir")
        if cache_dir_option:
            if os.path.isabs(cache_dir_option):
                self.cache_dir = cache_dir_option
            else:
                self.cache_dir = os.path.join(self.data_dir, cache_dir_option)
        else:
            self.cache_dir = self.data_dir

        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

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
        Download SIFT1M dataset from the official source using FTP.
        """
        base_url = self.AVAILABLE_DATASETS['sift1m']['url']
        files = [
            "sift_base.fvecs",
            "sift_query.fvecs",
            "sift_groundtruth.ivecs"
        ]

        try:
            parsed_url = urlparse(base_url)
            ftp_host = parsed_url.hostname
            ftp_path = parsed_url.path

            with FTP(ftp_host) as ftp:
                ftp.login()  # Anonymous login
                ftp.cwd(ftp_path)

                for filename in files:
                    file_path = os.path.join(self.data_dir, filename)
                    if os.path.exists(file_path):
                        print(f"File {filename} already exists, skipping download")
                        continue

                    print(f"Downloading {filename} from {base_url + filename}")

                    try:
                        # Try to get file size for progress bar
                        try:
                            total_size = ftp.size(filename)
                            with open(file_path, 'wb') as f, tqdm(
                                desc=filename,
                                total=total_size,
                                unit='B',
                                unit_scale=True,
                                unit_divisor=1024,
                            ) as pbar:
                                def callback(chunk):
                                    f.write(chunk)
                                    pbar.update(len(chunk))
                                
                                ftp.retrbinary(f'RETR {filename}', callback)
                        except Exception as size_error:
                            # If SIZE command fails, download without progress bar
                            print(f"Warning: Could not get file size for {filename} ({str(size_error)}). Downloading without progress bar...")
                            with open(file_path, 'wb') as f:
                                ftp.retrbinary(f'RETR {filename}', f.write)

                        print(f"Successfully downloaded {filename}")

                    except Exception as e:
                        print(f"Error downloading {filename}: {str(e)}")
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        raise
        except Exception as e:
            print(f"FTP connection failed for {base_url}: {str(e)}")
            raise

    def _download_glove(self):
        """
        Download GloVe 50d dataset.
        """
        url = self.AVAILABLE_DATASETS['glove50']['url']
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
                print(f"Error downloading GloVe dataset from {url}: {str(e)}")
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

        cache_suffix = ""
        if self.options:
            options_key = json.dumps(self.options, sort_keys=True)
            digest = hashlib.md5(options_key.encode("utf-8")).hexdigest()[:8]
            cache_suffix = f"_{digest}"

        cache_file = os.path.join(self.cache_dir, f"{self.name}{cache_suffix}_processed.pkl")

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
        print(f"  Ground truth: {self.ground_truth.shape if self.ground_truth is not None else 'N/A'}")

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
        dims = int(self.options.get("dimensions", dimensions))
        train_sz = int(self.options.get("train_size", train_size))
        test_sz = int(self.options.get("test_size", test_size))
        k_neighbors = int(self.options.get("ground_truth_k", k))

        np.random.seed(self.options.get("seed", 42))  # For reproducibility

        # Generate random vectors
        self.train_vectors = np.random.randn(train_sz, dims).astype(np.float32)
        self.test_vectors = np.random.randn(test_sz, dims).astype(np.float32)

        # Compute ground truth with brute force
        self.ground_truth = np.zeros((test_sz, k_neighbors), dtype=np.int32)
        distances = np.zeros((test_sz, train_sz))

        for i in tqdm(range(test_sz), desc="Computing ground truth"):
            # L2 distance
            distances[i] = np.linalg.norm(self.train_vectors - self.test_vectors[i:i+1], axis=1)
            self.ground_truth[i] = np.argsort(distances[i])[:k_neighbors]

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
        elif self.name == "msmarco":
            self._process_msmarco()
        elif self.name == "msmarco":
            self._process_msmarco()

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

    # ------------------------------------------------------------------
    # MS MARCO processing helpers
    # ------------------------------------------------------------------
    def _process_msmarco(self) -> None:
        """Process MS MARCO parquet files into TF-IDF vector representations."""
        try:
            import pyarrow.parquet as pq  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency managed via requirements
            raise ImportError(
                "pyarrow is required to load the MS MARCO dataset. Install pyarrow>=8.0.0"
            ) from exc

        from sklearn.feature_extraction.text import TfidfVectorizer

        version = self.options.get("version", "v2.1")
        base_split = self.options.get("base_split", "train")
        query_split = self.options.get("query_split", "validation")
        base_limit = int(self.options.get("base_limit", 5000))
        query_limit = int(self.options.get("query_limit", 1000))
        max_passages = int(self.options.get("max_passages_per_query", 5))
        include_unselected = bool(self.options.get("include_unselected", False))
        selected_only = bool(self.options.get("selected_only", True))
        max_features = int(self.options.get("vectorizer_max_features", 512))
        ground_truth_k = int(self.options.get("ground_truth_k", 10))
        batch_size = int(self.options.get("batch_size", 128))

        base_path = os.path.join(self.data_dir, version, f"{base_split}.parquet")
        query_path = os.path.join(self.data_dir, version, f"{query_split}.parquet")

        if not os.path.exists(base_path):
            raise FileNotFoundError(f"MS MARCO base split not found: {base_path}")
        if not os.path.exists(query_path):
            raise FileNotFoundError(f"MS MARCO query split not found: {query_path}")

        def iter_parquet_rows(parquet_path: str, columns: List[str], limit: Optional[int]) -> List[Dict[str, Any]]:
            pf = pq.ParquetFile(parquet_path)
            collected: List[Dict[str, Any]] = []
            remaining = limit

            for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
                data = batch.to_pydict()
                if not data:
                    continue

                batch_len = len(next(iter(data.values())))
                for i in range(batch_len):
                    row = {col: data[col][i] for col in columns}
                    collected.append(row)
                    if remaining is not None:
                        remaining -= 1
                        if remaining <= 0:
                            return collected

            return collected

        def normalise_text(value: Optional[str]) -> Optional[str]:
            if value is None:
                return None
            text = value.strip()
            return text if text else None

        doc_texts: List[str] = []
        doc_lookup: Dict[str, int] = {}

        def add_document(text: Optional[str]) -> Optional[int]:
            canonical = normalise_text(text)
            if not canonical:
                return None

            if canonical not in doc_lookup:
                doc_lookup[canonical] = len(doc_texts)
                doc_texts.append(canonical)

            return doc_lookup[canonical]

        def extract_passages(entry: Dict[str, Any]) -> Tuple[List[str], List[int]]:
            passages = entry.get("passages", {}) or {}
            texts = passages.get("passage_text") or []
            selections = passages.get("is_selected") or []

            # Ensure selections aligns with texts
            if len(selections) < len(texts):
                selections = list(selections) + [0] * (len(texts) - len(selections))

            return list(texts), list(selections)

        # ------------------------------------------------------------------
        # Build document collection from base split
        # ------------------------------------------------------------------
        base_rows = iter_parquet_rows(base_path, ["passages"], base_limit)

        for entry in base_rows:
            texts, selections = extract_passages(entry)
            if not texts:
                continue

            added = 0

            # Always add selected passages first
            for idx, text in enumerate(texts):
                is_selected = bool(selections[idx])
                if is_selected:
                    prev_count = len(doc_texts)
                    if add_document(text) is not None and len(doc_texts) > prev_count:
                        added += 1

            # Optionally include additional non-selected passages
            if not selected_only or include_unselected:
                for idx, text in enumerate(texts):
                    if added >= max_passages:
                        break

                    is_selected = bool(selections[idx])
                    if is_selected and selected_only:
                        continue

                    prev_count = len(doc_texts)
                    if add_document(text) is not None and len(doc_texts) > prev_count:
                        added += 1

        # ------------------------------------------------------------------
        # Prepare queries and ensure relevant passages are present
        # ------------------------------------------------------------------
        query_rows = iter_parquet_rows(query_path, ["query", "passages"], query_limit)

        queries: List[str] = []
        positives: List[List[int]] = []

        for entry in query_rows:
            query_text = normalise_text(entry.get("query"))
            if not query_text:
                continue

            texts, selections = extract_passages(entry)
            relevant_indices: List[int] = []

            for idx, text in enumerate(texts):
                is_selected = bool(selections[idx])
                if is_selected or not selected_only:
                    doc_idx = add_document(text)
                    if doc_idx is not None:
                        relevant_indices.append(doc_idx)
                if len(relevant_indices) >= ground_truth_k:
                    break

            if not relevant_indices and texts:
                doc_idx = add_document(texts[0])
                if doc_idx is not None:
                    relevant_indices.append(doc_idx)

            if not relevant_indices:
                continue  # Skip queries without identifiable positives

            queries.append(query_text)
            positives.append(relevant_indices)

        if not doc_texts:
            raise ValueError("No passages were collected for the MS MARCO dataset. Check dataset options.")
        if not queries:
            raise ValueError("No queries with positive passages found for the MS MARCO dataset.")

        # ------------------------------------------------------------------
        # Vectorise documents and queries using TF-IDF
        # ------------------------------------------------------------------
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        doc_matrix = vectorizer.fit_transform(doc_texts)
        query_matrix = vectorizer.transform(queries)

        self.train_vectors = doc_matrix.astype(np.float32).toarray()
        self.test_vectors = query_matrix.astype(np.float32).toarray()

        # ------------------------------------------------------------------
        # Construct ground truth matrix ensuring consistent width
        # ------------------------------------------------------------------
        max_relevant = max((len(p) for p in positives), default=0)
        effective_k = max(1, min(ground_truth_k, max_relevant))

        ground_truth = np.zeros((len(positives), effective_k), dtype=np.int32)

        for i, relevant_docs in enumerate(positives):
            for j in range(effective_k):
                ground_truth[i, j] = relevant_docs[j % len(relevant_docs)]

        self.ground_truth = ground_truth

        print("MS MARCO dataset processed:")
        print(f"  Documents: {self.train_vectors.shape}")
        print(f"  Queries: {self.test_vectors.shape}")
        print(f"  Ground truth width: {self.ground_truth.shape[1]}")

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
