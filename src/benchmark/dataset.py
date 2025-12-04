import hashlib
import logging
import os
import pickle
import zipfile
import json
import tempfile
from ftplib import FTP
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import numpy as np
import requests
from tqdm import tqdm

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - faiss is required for brute-force GT
    faiss = None

logger = logging.getLogger(__name__)

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
            "description": "MS MARCO passage ranking dataset (TF-IDF projection or pre-embedded vectors)",
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
        self._use_memmap_cache = bool(self.options.get("use_memmap_cache", False))
        self._cache_suffix = ""
        self._train_memmap_path: Optional[str] = None
        self._test_cache_path: Optional[str] = None
        self._ground_truth_cache_path: Optional[str] = None
        self._train_cache_format: Optional[str] = None
        self._ground_truth_suffix: str = "groundtruth"

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
        target_txt = os.path.join(self.data_dir, "glove.6B.50d.txt")

        # If the desired text file already exists, skip download/extraction entirely.
        if os.path.exists(target_txt):
            print(f"File {target_txt} already exists, skipping download and extraction.")
            return
        
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

        if self.name == "msmarco" and "_ground_truth_method" not in self.options:
            # Ensure cache keys change when the ground-truth construction method changes.
            self.options["_ground_truth_method"] = "bruteforce_v3"

        cache_suffix = ""
        if self.options:
            options_key = json.dumps(self.options, sort_keys=True)
            digest = hashlib.md5(options_key.encode("utf-8")).hexdigest()[:8]
            cache_suffix = f"_{digest}"
        self._cache_suffix = cache_suffix

        cache_file = os.path.join(self.cache_dir, f"{self.name}{cache_suffix}_processed.pkl")
        memmap_meta_path = self._memmap_meta_path()

        if self._use_memmap_cache and not force_download and os.path.exists(memmap_meta_path):
            print(f"Loading processed dataset from {memmap_meta_path}")
            self._load_memmap_cache(memmap_meta_path)
        elif os.path.exists(cache_file) and not force_download:
            print(f"Loading processed dataset from {cache_file}")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                self.train_vectors = data['train']
                self.test_vectors = data['test']
                self.ground_truth = data['ground_truth']
        else:
            if self.name == "random":
                print("Generating random dataset")
                self._generate_random_dataset()
            else:
                self.download()
                self._process_dataset()

            if self._use_memmap_cache:
                self._save_memmap_cache(memmap_meta_path)
            else:
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

    def _memmap_meta_path(self) -> str:
        filename = f"{self.name}{self._cache_suffix}_memmap.json"
        return os.path.join(self.cache_dir, filename)

    def _train_memmap_path_for_cache(self) -> str:
        filename = f"{self.name}{self._cache_suffix}_train.memmap"
        return os.path.join(self.cache_dir, filename)

    def _resolve_cache_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(self.cache_dir, path)

    def _write_array_cache(self, array: Optional[np.ndarray], suffix: str) -> Optional[str]:
        if array is None:
            return None

        filename = f"{self.name}{self._cache_suffix}_{suffix}.npy"
        final_path = os.path.join(self.cache_dir, filename)
        with tempfile.NamedTemporaryFile(
            dir=self.cache_dir,
            prefix=f"{self.name}{self._cache_suffix}_{suffix}_",
            suffix=".npy",
            delete=False,
        ) as tmp:
            np.save(tmp, array)
            tmp_path = tmp.name
        os.replace(tmp_path, final_path)
        return final_path

    def _save_memmap_cache(self, meta_path: str) -> None:
        if self.train_vectors is None:
            raise ValueError("Cannot cache dataset before it is loaded.")

        meta: Dict[str, Any] = {"version": 1}

        train_entry: Dict[str, Any]
        if self._train_cache_format == "memmap" and self._train_memmap_path:
            train_entry = {
                "path": os.path.relpath(self._train_memmap_path, self.cache_dir),
                "dtype": str(self.train_vectors.dtype),
                "shape": list(self.train_vectors.shape),
                "format": "memmap",
            }
        else:
            train_cache = self._write_array_cache(self.train_vectors, "train")
            if train_cache is None:
                raise ValueError("Failed to materialise training vectors for caching.")
            self._train_memmap_path = train_cache
            self._train_cache_format = "numpy"
            train_entry = {
                "path": os.path.relpath(train_cache, self.cache_dir),
                "dtype": str(self.train_vectors.dtype),
                "shape": list(self.train_vectors.shape),
                "format": "numpy",
            }

        test_cache = self._write_array_cache(self.test_vectors, "test")
        if test_cache is None:
            raise ValueError("Test vectors must be available for caching.")
        self._test_cache_path = test_cache

        ground_truth_cache = None
        if self.ground_truth is not None:
            suffix = self._ground_truth_suffix or "groundtruth"
            ground_truth_cache = self._write_array_cache(self.ground_truth, suffix)
            self._ground_truth_cache_path = ground_truth_cache

        meta["train"] = train_entry
        meta["test"] = {
            "path": os.path.relpath(test_cache, self.cache_dir),
            "format": "numpy",
        }
        if ground_truth_cache is not None:
            meta["ground_truth"] = {
                "path": os.path.relpath(ground_truth_cache, self.cache_dir),
                "format": "numpy",
            }

        tmp_meta_fd, tmp_meta_path = tempfile.mkstemp(dir=self.cache_dir, prefix=f"{self.name}{self._cache_suffix}_meta_", suffix=".json")
        os.close(tmp_meta_fd)
        try:
            with open(tmp_meta_path, "w", encoding="utf-8") as tmp_f:
                json.dump(meta, tmp_f, indent=2)
            os.replace(tmp_meta_path, meta_path)
        finally:
            if os.path.exists(tmp_meta_path):
                os.remove(tmp_meta_path)

    def _load_memmap_cache(self, meta_path: str) -> None:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        train_meta = meta.get("train")
        if not train_meta:
            raise ValueError(f"Invalid metadata cache: missing train entry ({meta_path})")

        train_path = self._resolve_cache_path(train_meta["path"])
        dtype = np.dtype(train_meta.get("dtype", "float32"))
        shape = tuple(train_meta.get("shape", []))
        fmt = train_meta.get("format", "memmap")

        if fmt == "memmap":
            self._train_memmap_path = train_path
            self.train_vectors = np.memmap(train_path, dtype=dtype, mode="r", shape=shape)
            self._train_cache_format = "memmap"
        elif fmt == "numpy":
            self.train_vectors = np.load(train_path, allow_pickle=False)
            self._train_memmap_path = train_path
            self._train_cache_format = "numpy"
        else:
            raise ValueError(f"Unsupported train cache format: {fmt}")

        test_meta = meta.get("test")
        if not test_meta:
            raise ValueError(f"Invalid metadata cache: missing test entry ({meta_path})")
        test_path = self._resolve_cache_path(test_meta["path"])
        test_format = test_meta.get("format", "numpy")
        if test_format != "numpy":
            raise ValueError(f"Unsupported test cache format: {test_format}")
        self.test_vectors = np.load(test_path, allow_pickle=False)
        self._test_cache_path = test_path

        ground_meta = meta.get("ground_truth")
        if ground_meta:
            ground_path = self._resolve_cache_path(ground_meta["path"])
            ground_format = ground_meta.get("format", "numpy")
            if ground_format != "numpy":
                raise ValueError(f"Unsupported ground truth cache format: {ground_format}")
            self.ground_truth = np.load(ground_path, allow_pickle=False)
            self._ground_truth_cache_path = ground_path
        else:
            self.ground_truth = None

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
            if self.options.get("use_preembedded", False) or self.options.get("preembedded_passage_dir"):
                self._process_msmarco_preembedded()
            else:
                self._process_msmarco_tfidf()

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
        Process GloVe dataset with optional subsampling controls.
        """
        txt_file = os.path.join(self.data_dir, "glove.6B.50d.txt")
        if not os.path.exists(txt_file):
            raise FileNotFoundError(f"glove.6B.50d.txt not found. Please run download first.")

        print("Processing GloVe dataset from text file...")
        options = self.options or {}
        seed = int(options.get("seed", 42))
        rng = np.random.default_rng(seed)
        requested_test_size = int(options.get("test_size", 1000))
        train_limit_raw = options.get("train_limit")
        test_limit_raw = options.get("test_limit")
        ground_truth_k = int(options.get("ground_truth_k", 100))

        vectors = []
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading GloVe vectors"):
                parts = line.split()
                vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                vectors.append(vector)
        
        all_vectors = np.array(vectors)
        available = all_vectors.shape[0]

        test_size = min(requested_test_size, available - 1)
        if test_size <= 0:
            raise ValueError("Dataset is too small to create a test split.")
            
        test_indices = rng.choice(available, test_size, replace=False)
        train_indices = np.setdiff1d(np.arange(available), test_indices)

        if test_limit_raw is not None:
            test_limit = min(int(test_limit_raw), test_indices.size)
            test_indices = test_indices[:test_limit]

        if train_limit_raw is not None:
            train_limit = min(int(train_limit_raw), train_indices.size)
            if train_limit < train_indices.size:
                train_indices = rng.choice(train_indices, train_limit, replace=False)

        self.test_vectors = all_vectors[test_indices]
        self.train_vectors = all_vectors[train_indices]

        # Compute ground truth for the test set
        print("Computing ground truth for GloVe...")
        k = min(ground_truth_k, self.train_vectors.shape[0])
        self.ground_truth = np.zeros((self.test_vectors.shape[0], k), dtype=np.int32)
        for i in tqdm(range(self.test_vectors.shape[0]), desc="Computing GloVe ground truth"):
            distances = np.linalg.norm(self.train_vectors - self.test_vectors[i], axis=1)
            self.ground_truth[i] = np.argsort(distances)[:k]

    # ------------------------------------------------------------------
    # MS MARCO processing helpers
    # ------------------------------------------------------------------
    def _process_msmarco_tfidf(self) -> None:
        """Process MS MARCO parquet files into TF-IDF vector representations."""
        try:
            import pyarrow.parquet as pq  # type: ignore
            import pyarrow as pa  # type: ignore
            import pyarrow as pa  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency managed via requirements
            raise ImportError(
                "pyarrow is required to load the MS MARCO dataset. Install pyarrow>=8.0.0"
            ) from exc

        from sklearn.feature_extraction.text import TfidfVectorizer

        version = self.options.get("version", "v2.1")
        base_split = self.options.get("base_split", "train")
        query_split = self.options.get("query_split", "validation")
        base_limit_raw = self.options.get("base_limit")
        base_limit = int(base_limit_raw) if base_limit_raw is not None else None
        query_limit_raw = self.options.get("query_limit")
        query_limit = int(query_limit_raw) if query_limit_raw is not None else None
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

    def _compute_bruteforce_ground_truth(
        self,
        train_vectors: np.ndarray,
        query_vectors: np.ndarray,
        requested_k: int,
        metric: str,
        normalize_cosine: bool,
    ) -> np.ndarray:
        """
        Build exact ground truth by performing a brute-force search using FAISS.

        Args:
            train_vectors: Passage embedding matrix of shape (n_train, dim)
            query_vectors: Query embedding matrix of shape (n_queries, dim)
            requested_k: Desired number of neighbours per query
            metric: Distance/similarity metric ('cosine', 'l2', or 'ip')

        Returns:
            Ground-truth index matrix of shape (n_queries, effective_k)
        """
        if faiss is None:  # pragma: no cover - dependency managed via requirements
            raise ImportError(
                "faiss-cpu is required to compute brute-force ground truth. Install faiss-cpu>=1.7.4."
            )

        if requested_k <= 0:
            raise ValueError("requested_k must be a positive integer.")

        train_array = np.asarray(train_vectors, dtype=np.float32)
        query_array = np.asarray(query_vectors, dtype=np.float32)

        if train_array.ndim != 2 or query_array.ndim != 2:
            raise ValueError("Training and query embeddings must both be 2D arrays.")

        n_train, dim = train_array.shape
        if query_array.shape[1] != dim:
            raise ValueError(
                f"Dimensional mismatch between passages ({dim}) and queries ({query_array.shape[1]})."
            )

        effective_k = min(requested_k, n_train)
        effective_k = max(1, effective_k)

        metric_name = metric.lower()
        logger.debug(
            "Preparing FAISS ground truth: metric=%s, requested_k=%s, train_dtype=%s, query_dtype=%s",
            metric,
            requested_k,
            train_array.dtype,
            query_array.dtype,
        )
        if metric_name in {"cosine", "cos"}:
            train_search = train_array if train_array.flags["C_CONTIGUOUS"] else np.ascontiguousarray(train_array)
            query_search = query_array if query_array.flags["C_CONTIGUOUS"] else np.ascontiguousarray(query_array)
            if normalize_cosine:
                train_search = np.ascontiguousarray(train_search.copy())
                query_search = np.ascontiguousarray(query_search.copy())
                faiss.normalize_L2(train_search)
                faiss.normalize_L2(query_search)
            index = faiss.IndexFlatIP(dim)
        elif metric_name in {"ip", "inner_product"}:
            train_search = train_array if train_array.flags["C_CONTIGUOUS"] else np.ascontiguousarray(train_array)
            query_search = query_array if query_array.flags["C_CONTIGUOUS"] else np.ascontiguousarray(query_array)
            index = faiss.IndexFlatIP(dim)
        elif metric_name in {"l2", "euclidean"}:
            train_search = train_array if train_array.flags["C_CONTIGUOUS"] else np.ascontiguousarray(train_array)
            query_search = query_array if query_array.flags["C_CONTIGUOUS"] else np.ascontiguousarray(query_array)
            index = faiss.IndexFlatL2(dim)
        else:
            raise ValueError(
                f"Unsupported ground-truth metric '{metric}'. Expected one of: cosine, ip, l2."
            )

        logger.debug(
            "Running faiss search: n_train=%s, n_queries=%s, dim=%s, metric=%s, normalize_cosine=%s, train_contig=%s, query_contig=%s",
            n_train,
            query_array.shape[0],
            dim,
            metric,
            normalize_cosine,
            (train_search.flags["C_CONTIGUOUS"], train_search.flags["F_CONTIGUOUS"]),
            (query_search.flags["C_CONTIGUOUS"], query_search.flags["F_CONTIGUOUS"]),
        )
        try:
            train_ptr = int(train_search.ctypes.data)
            query_ptr = int(query_search.ctypes.data)
        except AttributeError:
            train_ptr = query_ptr = None
        logger.debug("train_ptr=%s query_ptr=%s", train_ptr, query_ptr)
        logger.debug("index ntotal before add=%s", index.ntotal)
        index.add(train_search)
        logger.debug("index ntotal after add=%s", index.ntotal)
        distances, indices = index.search(query_search, effective_k)
        logger.debug(
            "Faiss search completed (metric=%s, normalize_cosine=%s): indices dtype=%s sample=%s",
            metric,
            normalize_cosine,
            indices.dtype,
            indices[0, : min(5, effective_k)].tolist() if indices.size else [],
        )
        logger.debug(
            "Distance sample: %s",
            distances[0, : min(5, effective_k)].tolist() if distances.size else [],
        )
        del index  # ensure resources are reclaimed promptly

        return indices.astype(np.int32, copy=False)

    def _process_msmarco_preembedded(self) -> None:
        """Load pre-embedded MS MARCO passages and queries from parquet files."""
        embedded_dir_value = self.options.get("embedded_dataset_dir") or self.options.get("embedding_dir")
        passage_embeddings_path_opt = self.options.get("passage_embeddings_path")
        query_embeddings_path_opt = self.options.get("query_embeddings_path")
        ground_truth_path_opt = self.options.get("ground_truth_path")

        if embedded_dir_value or passage_embeddings_path_opt or query_embeddings_path_opt or ground_truth_path_opt:
            embedded_dir = Path(embedded_dir_value) if embedded_dir_value else None

            def resolve_path(option_key: str, default_filename: str) -> Path:
                explicit = self.options.get(option_key)
                if explicit:
                    return Path(explicit)
                if embedded_dir is not None:
                    return embedded_dir / default_filename
                raise ValueError(
                    f"Dataset option '{option_key}' is required when 'embedded_dataset_dir' is not provided."
                )

            passage_path = resolve_path("passage_embeddings_path", "passage_embeddings.npy")
            query_path = resolve_path("query_embeddings_path", "query_embeddings.npy")

            if ground_truth_path_opt:
                logger.warning(
                    "Ignoring provided 'ground_truth_path' option (%s); ground truth will be recomputed "
                    "via brute-force search over the supplied embeddings.",
                    self.options.get("ground_truth_path"),
                )

            if not passage_path.exists():
                raise FileNotFoundError(f"Passage embeddings file not found: {passage_path}")
            if not query_path.exists():
                raise FileNotFoundError(f"Query embeddings file not found: {query_path}")

            use_memmap_cache = bool(self.options.get("use_memmap_cache", False))
            mmap_mode = "r" if use_memmap_cache else None

            train_vectors = np.load(passage_path, mmap_mode=mmap_mode, allow_pickle=False)
            if train_vectors.dtype != np.float32:
                raise ValueError(
                    f"Expected passage embeddings to be float32 but found {train_vectors.dtype} at {passage_path}"
                )
            if train_vectors.ndim != 2:
                raise ValueError(f"Passage embeddings must be 2D; found shape {train_vectors.shape} at {passage_path}")

            test_vectors = np.load(query_path, mmap_mode=None, allow_pickle=False)
            if test_vectors.dtype != np.float32:
                raise ValueError(
                    f"Expected query embeddings to be float32 but found {test_vectors.dtype} at {query_path}"
                )
            if test_vectors.ndim != 2:
                raise ValueError(f"Query embeddings must be 2D; found shape {test_vectors.shape} at {query_path}")
            if test_vectors.shape[1] != train_vectors.shape[1]:
                raise ValueError(
                    "Dimensional mismatch between passages and queries: "
                    f"{train_vectors.shape[1]} vs {test_vectors.shape[1]}"
                )

            self.train_vectors = train_vectors
            self.test_vectors = test_vectors
            self._ground_truth_cache_path = None

            if use_memmap_cache and isinstance(train_vectors, np.memmap):
                self._train_memmap_path = str(passage_path)
                self._train_cache_format = "memmap"
            else:
                self._train_memmap_path = None
                self._train_cache_format = None

            metric_value = (
                self.options.get("metric")
                or self.options.get("distance_metric")
                or self.options.get("similarity")
                or "cosine"
            )
            metric = str(metric_value).lower()

            ground_truth_k_opt = (
                self.options.get("ground_truth_k")
                or self.options.get("GROUND_TRUTH_K")
                or self.options.get("topk")
            )
            try:
                requested_k = int(ground_truth_k_opt) if ground_truth_k_opt is not None else 100
            except (TypeError, ValueError) as exc:
                raise ValueError("ground_truth_k must be an integer") from exc

            normalize_cosine = bool(self.options.get("normalize_cosine_groundtruth", False))
            ground_truth = self._compute_bruteforce_ground_truth(
                train_vectors,
                test_vectors,
                requested_k,
                metric,
                normalize_cosine,
            )
            self.ground_truth = ground_truth
            effective_k = ground_truth.shape[1]
            self._ground_truth_suffix = f"top{effective_k}groundtruth"

            if not self._use_memmap_cache:
                cache_path = self._write_array_cache(ground_truth, self._ground_truth_suffix)
                if cache_path is not None:
                    self._ground_truth_cache_path = cache_path

            logger.info(
                "Ground-truth index range: min=%s max=%s",
                int(ground_truth.min()) if ground_truth.size else None,
                int(ground_truth.max()) if ground_truth.size else None,
            )

            logger.info(
                "Computed brute-force ground truth for MS MARCO using metric=%s (requested_k=%s, effective_k=%s, normalize_cosine=%s).",
                metric,
                requested_k,
                effective_k,
                normalize_cosine,
            )

            print("MS MARCO embedded dataset loaded from pre-generated numpy arrays:")
            print(f"  Documents: {self.train_vectors.shape}")
            print(f"  Queries: {self.test_vectors.shape}")
            print(f"  Ground truth width: {self.ground_truth.shape[1]}")
            return

        try:
            import pyarrow.parquet as pq  # type: ignore
            import pyarrow as pa  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency managed via requirements
            raise ImportError(
                "pyarrow is required to load the MS MARCO dataset. Install pyarrow>=8.0.0"
            ) from exc

        batch_size = int(self.options.get("batch_size", 128))
        base_limit_raw = self.options.get("base_limit")
        base_limit = int(base_limit_raw) if base_limit_raw is not None else 0
        query_limit_raw = self.options.get("query_limit")
        query_limit = int(query_limit_raw) if query_limit_raw is not None else 0
        ground_truth_k = int(self.options.get("ground_truth_k", 10))
        candidate_limit = int(self.options.get("relevance_candidates_limit", max(ground_truth_k, 100)))
        max_passage_scan_raw = self.options.get("max_passage_scan")
        max_passage_scan = int(max_passage_scan_raw) if max_passage_scan_raw is not None else 0
        strict_resolution = bool(self.options.get("strict_relevance_resolution", True))
        progress_interval = int(self.options.get("progress_log_interval", 200_000))

        if base_limit < 0:
            base_limit = 0
        if query_limit < 0:
            query_limit = 0
        if candidate_limit <= 0:
            candidate_limit = max(ground_truth_k, 1)
        if max_passage_scan < 0:
            max_passage_scan = 0
        if progress_interval < 0:
            progress_interval = 0

        passage_dir = self.options.get("preembedded_passage_dir")
        query_dir = self.options.get("preembedded_query_dir")

        if passage_dir is None:
            root = self.options.get("preembedded_root", self.data_dir)
            passage_dir = os.path.join(root, "passages_parquet")
        if query_dir is None:
            root = self.options.get("preembedded_root", self.data_dir)
            query_dir = os.path.join(root, "queries_parquet")

        passage_dir_path = Path(passage_dir)
        query_dir_path = Path(query_dir)

        if not passage_dir_path.exists():
            raise FileNotFoundError(f"Pre-embedded passage directory not found: {passage_dir_path}")
        if not query_dir_path.exists():
            raise FileNotFoundError(f"Pre-embedded query directory not found: {query_dir_path}")

        passage_paths = sorted(passage_dir_path.rglob("*.parquet"))
        query_paths = sorted(query_dir_path.rglob("*.parquet"))

        if not passage_paths:
            raise FileNotFoundError(f"No parquet files found in {passage_dir_path}")
        if not query_paths:
            raise FileNotFoundError(f"No parquet files found in {query_dir_path}")

        def is_vector_field(field: pa.Field) -> bool:
            field_type = field.type

            if pa.types.is_fixed_size_list(field_type):
                return pa.types.is_floating(field_type.value_type)

            if pa.types.is_list(field_type) or pa.types.is_large_list(field_type):
                value_field = field_type.value_field
                value_type = value_field.type

                if pa.types.is_fixed_size_list(value_type):
                    return pa.types.is_floating(value_type.value_type)

                return pa.types.is_floating(value_type)

            return False

        def select_column(
            paths: Sequence[Path],
            requested: Optional[Any],
            fallbacks: Sequence[str],
            required: bool,
            context: str,
            require_vector: bool = False,
        ) -> Optional[str]:
            def normalise(value: Optional[Any]) -> List[str]:
                if value is None:
                    return []
                if isinstance(value, str):
                    return [value]
                if isinstance(value, Sequence):
                    return [str(v) for v in value]
                return [str(value)]

            candidates: List[str] = []
            candidates.extend(normalise(requested))
            for fb in fallbacks:
                if fb not in candidates:
                    candidates.append(fb)

            first_available: Optional[List[str]] = None

            for path in paths:
                pf = pq.ParquetFile(path)
                arrow_schema = getattr(pf, "schema_arrow", None)
                if arrow_schema is None:
                    arrow_schema = pf.schema.to_arrow_schema()


                field_lookup = {field.name: field for field in arrow_schema}
                schema_names = set(field_lookup.keys())

                if first_available is None:
                    first_available = list(field_lookup.keys())

                logger.debug(
                    "select_column context=%s path=%s candidates=%s schema=%s",
                    context,
                    path,
                    candidates,
                    list(field_lookup.keys()),
                )

                for candidate in candidates:
                    logger.debug("evaluating candidate=%s for context=%s", candidate, context)
                    if candidate in schema_names:
                        field = field_lookup.get(candidate)
                        if not require_vector:
                            return candidate
                        if field is None or is_vector_field(field):
                            return candidate

                    field = field_lookup.get(candidate)
                    if field is not None and pa.types.is_struct(field.type):
                        for child in field.type:
                            if child.name in {"values", "list", "array"} and is_vector_field(child):
                                return f"{candidate}.{child.name}"

            if required and require_vector:
                for path in paths:
                    pf = pq.ParquetFile(path)
                    arrow_schema = getattr(pf, "schema_arrow", None)
                    if arrow_schema is None:
                        arrow_schema = pf.schema.to_arrow_schema()

                    for field in arrow_schema:
                        if is_vector_field(field):
                            return field.name

                        if pa.types.is_struct(field.type):
                            for child in field.type:
                                if is_vector_field(child):
                                    return f"{field.name}.{child.name}"

                message = (
                    "Could not locate required column for "
                    f"{context}. Checked candidates: {candidates}. "
                    f"Available top-level columns include: {first_available or []}"
                )
                logger.error(message)
                raise ValueError(message)
            return None

        passage_embedding_column = select_column(
            passage_paths,
            self.options.get("passage_embedding_column"),
            ["emb", "embedding", "vector"],
            True,
            "passage embeddings",
            require_vector=True,
        )

        passage_id_column = select_column(
            passage_paths,
            self.options.get("passage_id_column"),
            ["_id", "id", "doc_id", "passage_id"],
            False,
            "passage identifiers",
        )

        query_embedding_column = select_column(
            query_paths,
            self.options.get("query_embedding_column"),
            ["emb", "embedding", "vector"],
            True,
            "query embeddings",
            require_vector=True,
        )

        query_relevance_column = select_column(
            query_paths,
            self.options.get("query_relevance_column"),
            [
                "top1k_passage_ids",
                "positive_passage_ids",
                "doc_ids",
                "positive_passages",
                "qrels",
            ],
            False,
            "query relevance passage identifiers",
        )

        query_relevance_offsets_column = select_column(
            query_paths,
            self.options.get("query_relevance_offsets_column"),
            ["top1k_offsets", "positive_passage_offsets", "offsets"],
            False,
            "query relevance passage offsets",
        )

        if query_relevance_column is None:
            query_relevance_column = select_column(
                query_paths,
                None,
                ["top1k_passage_ids", "positive_passages", "qrels"],
                False,
                "query relevance identifiers",
            )

        if query_relevance_offsets_column is None:
            query_relevance_offsets_column = select_column(
                query_paths,
                None,
                ["top1k_offsets", "positive_passage_offsets"],
                False,
                "query relevance offsets",
            )

        logger.debug(
            "MS MARCO pre-embedded query columns -> embeddings: %s ids: %s offsets: %s",
            query_embedding_column,
            query_relevance_column,
            query_relevance_offsets_column,
        )

        if query_relevance_column is None and query_relevance_offsets_column is None:
            available = []
            for path in query_paths:
                pf = pq.ParquetFile(path)
                arrow_schema = getattr(pf, "schema_arrow", None)
                if arrow_schema is None:
                    arrow_schema = pf.schema.to_arrow_schema()
                available.extend(arrow_schema.names)
                break
            message = (
                "MS MARCO pre-embedded queries require either a relevance id column or an offset column."
                f" Observed columns: {available}"
            )
            logger.error(message)
            raise ValueError(message)

        # ------------------------------------------------------------------
        # Pass 1: Read queries to collect embeddings and relevance references
        # ------------------------------------------------------------------
        queries_raw: List[Tuple[np.ndarray, List[str], List[int]]] = []
        needed_doc_ids: set[str] = set()
        needed_offsets: set[int] = set()

        for path in query_paths:
            pf = pq.ParquetFile(path)
            columns = [query_embedding_column]
            if query_relevance_column:
                columns.append(query_relevance_column)
            if query_relevance_offsets_column and query_relevance_offsets_column not in columns:
                columns.append(query_relevance_offsets_column)

            for batch in pf.iter_batches(columns=columns, batch_size=batch_size):
                data = batch.to_pydict()
                if not data:
                    continue

                batch_len = len(next(iter(data.values())))
                for i in range(batch_len):
                    embedding = data[query_embedding_column][i]
                    if embedding is None:
                        continue

                    vector = np.asarray(embedding, dtype=np.float32)
                    if vector.ndim == 2 and vector.shape[0] == 1:
                        vector = vector[0]
                    if vector.ndim != 1:
                        raise ValueError(f"Unexpected embedding shape for query row: {vector.shape}")

                    id_candidates: List[str] = []
                    if query_relevance_column:
                        raw_ids = data.get(query_relevance_column, [None])[i]

                        if isinstance(raw_ids, dict):
                            raw_ids = list(raw_ids.keys())

                        candidates_iterable = list(raw_ids or [])

                        for entry in candidates_iterable[:candidate_limit]:
                            if entry is None:
                                continue

                            doc_id = None

                            if isinstance(entry, (list, tuple)):
                                doc_id = entry[0] if entry else None
                            elif isinstance(entry, dict):
                                doc_id = entry.get("doc_id") or entry.get("passage_id")
                            else:
                                doc_id = entry

                            if doc_id is None:
                                continue

                            doc_str = str(doc_id)
                            id_candidates.append(doc_str)
                            needed_doc_ids.add(doc_str)

                    offset_candidates: List[int] = []
                    if query_relevance_offsets_column:
                        raw_offsets = data.get(query_relevance_offsets_column, [None])[i]

                        candidates_iterable = list(raw_offsets or [])

                        for entry in candidates_iterable[:candidate_limit]:
                            offset_value = None

                            if isinstance(entry, (list, tuple)):
                                offset_value = entry[0] if entry else None
                            elif isinstance(entry, dict):
                                offset_value = entry.get("offset") or entry.get("passage_offset")
                            else:
                                offset_value = entry

                            try:
                                offset_int = int(offset_value)
                            except (TypeError, ValueError):
                                continue

                            offset_candidates.append(offset_int)
                            needed_offsets.add(offset_int)

                    queries_raw.append((vector, id_candidates, offset_candidates))

                    if query_limit and len(queries_raw) >= query_limit:
                        break

                if query_limit and len(queries_raw) >= query_limit:
                    break
            if query_limit and len(queries_raw) >= query_limit:
                break

        if not queries_raw:
            raise ValueError("No queries were loaded from the pre-embedded dataset.")

        # ------------------------------------------------------------------
        # Pass 2: Read passages, ensuring coverage for required doc ids/offsets
        # ------------------------------------------------------------------
        use_memmap_cache = self._use_memmap_cache
        train_memmap_target = self._train_memmap_path_for_cache() if use_memmap_cache else None
        memmap_tmp_path: Optional[str] = None
        memmap_fp: Optional[BinaryIO] = None

        if use_memmap_cache:
            if train_memmap_target is None:
                raise ValueError("Cache directory is not configured for memmap storage.")
            memmap_tmp_path = f"{train_memmap_target}.tmp"
            os.makedirs(os.path.dirname(train_memmap_target), exist_ok=True)
            if os.path.exists(train_memmap_target):
                os.remove(train_memmap_target)
            if os.path.exists(memmap_tmp_path):
                os.remove(memmap_tmp_path)
            memmap_fp = open(memmap_tmp_path, "wb")

        doc_vectors: List[np.ndarray] = []
        doc_id_to_index: Dict[str, int] = {}
        offset_to_index: Dict[int, int] = {}
        doc_count = 0
        doc_dim: Optional[int] = None

        global_offset = 0
        last_progress_logged = 0

        def store_vector(vector: np.ndarray) -> int:
            nonlocal doc_dim, doc_count

            contiguous = np.ascontiguousarray(vector, dtype=np.float32)

            if doc_dim is None:
                doc_dim = contiguous.shape[0]
            elif contiguous.shape[0] != doc_dim:
                raise ValueError(
                    f"Inconsistent embedding dimension for MS MARCO passages: "
                    f"expected {doc_dim}, observed {contiguous.shape[0]}"
                )

            local_idx = doc_count
            if use_memmap_cache:
                assert memmap_fp is not None
                memmap_fp.write(contiguous.tobytes())
            else:
                doc_vectors.append(contiguous)

            doc_count += 1
            return local_idx

        def add_passage(vector: np.ndarray, doc_identifier: Optional[str], offset: int) -> None:
            local_idx = store_vector(vector)
            offset_to_index[offset] = local_idx
            if doc_identifier is not None and doc_identifier not in doc_id_to_index:
                doc_id_to_index[doc_identifier] = local_idx

        try:
            for path in passage_paths:
                pf = pq.ParquetFile(path)
                columns = [passage_embedding_column]
                if passage_id_column:
                    columns.append(passage_id_column)

                for batch in pf.iter_batches(columns=columns, batch_size=batch_size):
                    data = batch.to_pydict()
                    if not data:
                        continue

                    embeddings = data[passage_embedding_column]
                    ids = data.get(passage_id_column) if passage_id_column else None

                    batch_len = len(embeddings)
                    for i in range(batch_len):
                        embedding = embeddings[i]
                        doc_identifier = None
                        if passage_id_column and ids is not None:
                            doc_identifier = ids[i]
                            doc_identifier = str(doc_identifier) if doc_identifier is not None else None

                        vector = np.asarray(embedding, dtype=np.float32) if embedding is not None else None
                        if vector is not None and vector.ndim == 2 and vector.shape[0] == 1:
                            vector = vector[0]

                        should_add = False
                        if vector is not None and vector.ndim == 1:
                            if base_limit <= 0 or doc_count < base_limit:
                                should_add = True
                        elif (
                            doc_identifier is not None
                            and doc_identifier in needed_doc_ids
                            and doc_identifier not in doc_id_to_index
                        ):
                            should_add = True
                        elif global_offset in needed_offsets and global_offset not in offset_to_index:
                            should_add = True

                    if should_add and vector is not None:
                        add_passage(vector, doc_identifier, global_offset)

                    global_offset += 1
                    if (
                        progress_interval > 0
                        and global_offset - last_progress_logged >= progress_interval
                    ):
                        logger.info(
                            "MS MARCO loader progress: processed %s rows, retained %s vectors "
                            "(base_limit=%s, max_passage_scan=%s)",
                            f"{global_offset:,}",
                            f"{doc_count:,}",
                            base_limit or "unbounded",
                            max_passage_scan or "unbounded",
                        )
                        last_progress_logged = global_offset

                    should_stop = False
                    if base_limit > 0 and doc_count >= base_limit:
                        if strict_resolution:
                            should_stop = needed_doc_ids.issubset(doc_id_to_index.keys()) and needed_offsets.issubset(
                                offset_to_index.keys()
                            )
                        else:
                            should_stop = True
                    if not should_stop and max_passage_scan > 0 and global_offset >= max_passage_scan:
                        should_stop = True

                    if should_stop:
                        break

                should_stop_outer = False
                if base_limit > 0 and doc_count >= base_limit:
                    if strict_resolution:
                        should_stop_outer = needed_doc_ids.issubset(doc_id_to_index.keys()) and needed_offsets.issubset(
                            offset_to_index.keys()
                        )
                    else:
                        should_stop_outer = True
                if not should_stop_outer and max_passage_scan > 0 and global_offset >= max_passage_scan:
                    should_stop_outer = True

                if should_stop_outer:
                    break
        finally:
            if memmap_fp is not None:
                memmap_fp.flush()
                memmap_fp.close()

        if doc_count == 0:
            if memmap_tmp_path and os.path.exists(memmap_tmp_path):
                os.remove(memmap_tmp_path)
            raise ValueError("No passages with embeddings were loaded from the pre-embedded dataset.")

        if use_memmap_cache:
            assert train_memmap_target is not None
            assert memmap_tmp_path is not None
            if doc_dim is None:
                raise ValueError("Unable to infer MS MARCO embedding dimensionality.")
            os.replace(memmap_tmp_path, train_memmap_target)
            self._train_memmap_path = train_memmap_target
            self.train_vectors = np.memmap(train_memmap_target, dtype=np.float32, mode="r", shape=(doc_count, doc_dim))
            self._train_cache_format = "memmap"
        else:
            self.train_vectors = np.vstack(doc_vectors)
            self._train_memmap_path = None
            self._train_cache_format = None

        missing_ids = needed_doc_ids.difference(doc_id_to_index.keys())
        missing_offsets = needed_offsets.difference(offset_to_index.keys())

        if missing_ids or missing_offsets:
            message = (
                "Warning: Could not load all requested ground-truth passages. "
                f"Missing ids: {len(missing_ids)}, missing offsets: {len(missing_offsets)}"
            )
            if not strict_resolution:
                message += " (strict_relevance_resolution is disabled; continuing with partial coverage.)"
            elif max_passage_scan > 0 and global_offset >= max_passage_scan:
                message += f" (Reached max_passage_scan={max_passage_scan:,} during processing.)"
            print(message)

        # ------------------------------------------------------------------
        # Pass 3: Build query vectors and align ground truth indices
        # ------------------------------------------------------------------
        query_vectors: List[np.ndarray] = []
        positives: List[List[int]] = []

        for vector, id_candidates, offset_candidates in queries_raw:
            relevant_indices: List[int] = []
            seen: set[int] = set()

            for doc_identifier in id_candidates:
                idx = doc_id_to_index.get(doc_identifier)
                if idx is None or idx in seen:
                    continue
                relevant_indices.append(idx)
                seen.add(idx)
                if len(relevant_indices) >= ground_truth_k:
                    break

            if len(relevant_indices) < ground_truth_k:
                for offset in offset_candidates:
                    idx = offset_to_index.get(offset)
                    if idx is None or idx in seen:
                        continue
                    relevant_indices.append(idx)
                    seen.add(idx)
                    if len(relevant_indices) >= ground_truth_k:
                        break

            if not relevant_indices:
                continue

            query_vectors.append(vector)
            positives.append(relevant_indices)

        if not query_vectors:
            missing_ids = needed_doc_ids.difference(doc_id_to_index.keys())
            missing_offsets = needed_offsets.difference(offset_to_index.keys())
            raise ValueError(
                "No queries with matching ground-truth passages were loaded. "
                f"Loaded passages: {doc_count} (base_limit={base_limit}, "
                f"max_passage_scan={max_passage_scan or 'unbounded'}). "
                f"Resolved doc ids: {len(doc_id_to_index)}/{len(needed_doc_ids)}, "
                f"offsets: {len(offset_to_index)}/{len(needed_offsets)}. "
                "Increase base_limit or max_passage_scan, raise progress_log_interval, "
                "or re-enable strict_relevance_resolution to continue scanning for required passages."
            )

        self.test_vectors = np.vstack(query_vectors)

        max_relevant = max((len(p) for p in positives), default=0)
        effective_k = max(1, min(ground_truth_k, max_relevant))

        ground_truth = np.zeros((len(positives), effective_k), dtype=np.int32)
        for i, relevant_docs in enumerate(positives):
            for j in range(effective_k):
                idx = relevant_docs[j] if j < len(relevant_docs) else relevant_docs[-1]
                ground_truth[i, j] = idx

        self.ground_truth = ground_truth

        print("MS MARCO pre-embedded dataset processed:")
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
