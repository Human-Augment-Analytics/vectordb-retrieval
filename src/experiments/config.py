import yaml
import copy
from typing import Dict, Any

class ExperimentConfig:
    """
    Configuration for vector retrieval experiments.
    """

    def __init__(self, **kwargs):
        """
        Initialize configuration with default values.

        Args:
            **kwargs: Override default values
        """
        # Dataset configuration
        self.dataset = kwargs.get("dataset", "random")
        self.data_dir = kwargs.get("data_dir", "data")
        self.force_download = kwargs.get("force_download", False)
        self.dataset_options = copy.deepcopy(kwargs.get("dataset_options", {}))

        # Experiment parameters
        self.n_queries = kwargs.get("n_queries", 1000)  # Number of test queries to run
        self.topk = kwargs.get("topk", 100)  # Number of nearest neighbors to retrieve
        self.repeat = kwargs.get("repeat", 1)  # Number of times to repeat experiment

        # Algorithm configurations
        default_algorithms = {
            "exact": {"type": "ExactSearch", "metric": "l2"},
            "approx": {"type": "ApproximateSearch", "index_type": "IVF100,Flat", "metric": "l2", "nprobe": 10},
            "hnsw": {"type": "HNSW", "M": 16, "efConstruction": 200, "efSearch": 100, "metric": "l2"}
        }
        self.algorithms = copy.deepcopy(kwargs.get("algorithms", default_algorithms))

        # Dataset-wide metric preference (optional)
        self.metric = kwargs.get("metric")
        if self.metric is not None:
            for alg_config in self.algorithms.values():
                if isinstance(alg_config, dict):
                    alg_config.setdefault("metric", self.metric)

        # Additional parameters
        self.seed = kwargs.get("seed", 42)  # Random seed for reproducibility
        self.output_prefix = kwargs.get("output_prefix", "experiment")  # Prefix for output files

    @classmethod
    def from_yaml(cls, yaml_file: str) -> 'ExperimentConfig':
        """
        Load configuration from a YAML file.

        Args:
            yaml_file: Path to YAML configuration file

        Returns:
            ExperimentConfig instance
        """
        with open(yaml_file, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        config_dict = {
            "dataset": self.dataset,
            "data_dir": self.data_dir,
            "force_download": self.force_download,
            "dataset_options": self.dataset_options,
            "n_queries": self.n_queries,
            "topk": self.topk,
            "repeat": self.repeat,
            "algorithms": self.algorithms,
            "seed": self.seed,
            "output_prefix": self.output_prefix
        }

        if self.metric is not None:
            config_dict["metric"] = self.metric

        return config_dict

    def save(self, output_file: str) -> None:
        """
        Save configuration to a YAML file.

        Args:
            output_file: Path to save configuration
        """
        with open(output_file, 'w') as f:
            yaml.dump(self.to_dict(), f)

    def __str__(self) -> str:
        return yaml.dump(self.to_dict())
