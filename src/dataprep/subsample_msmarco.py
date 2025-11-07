import os
import random
import logging
from typing import Optional

import csv
import yaml

from pathlib import Path
from tqdm import tqdm
from box import ConfigBox

import ir_datasets

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "ms_marco_subset_embed.yaml"

logger = logging.getLogger(__name__)


def _load_config() -> ConfigBox:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return ConfigBox(yaml.safe_load(file))


config = _load_config()


def _ensure_ir_datasets_home(cfg: ConfigBox) -> Optional[Path]:
    """Ensure IR_DATASETS_HOME is set, falling back to config defaults."""
    configured = os.environ.get("IR_DATASETS_HOME")
    if configured:
        return Path(configured)

    default_home = cfg.common.get("IR_DATASETS_HOME")
    if default_home:
        default_path = Path(default_home)
        os.environ["IR_DATASETS_HOME"] = str(default_path)
        return default_path

    return None


def _resolve_output_dir(ir_home: Optional[Path], cfg: ConfigBox) -> Path:
    subset_cfg = cfg.subset
    embeddings_cfg = cfg.embeddings

    explicit = (
        subset_cfg.get("OUTPUT_DIR")
        or embeddings_cfg.get("SUBSAMPLED_DATA_DIR")
        or embeddings_cfg.get("INPUT_DIR")
    )
    if explicit:
        return Path(explicit)

    if ir_home:
        return ir_home.parent / "msmarco_v1_subsampled"

    return Path.cwd() / "msmarco_v1_subsampled"


def sample_corpus(output_dir: Path, sample_size: int, seed: int) -> None:
    """
    Loads and samples the full MSMARCO v1 passage corpus.
    Assumes data is pre-downloaded based on IR_DATASETS_HOME.
    """
    logger.info("Loading 'msmarco-passage' corpus via ir_datasets...")

    try:
        dataset = ir_datasets.load("msmarco-passage")
        n_total_docs = dataset.docs_count()
        logger.info("Total documents in corpus: %s", f"{n_total_docs:,}")

    except Exception as e:
        logger.error(
            "Error loading dataset. Is IR_DATASETS_HOME set correctly and data downloaded? (%s)",
            e,
        )
        raise

    if sample_size > n_total_docs:
        raise ValueError(
            f"Requested {sample_size:,} passages but dataset only contains {n_total_docs:,} documents."
        )

    logger.info("Generating %s random indices (seed=%s)...", f"{sample_size:,}", seed)
    random.seed(seed)
    indices_to_keep = set(random.sample(range(n_total_docs), sample_size))

    out_path = output_dir / "corpus.tsv"
    logger.info("Streaming corpus and writing subsample to %s...", out_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    with out_path.open('w', encoding='utf-8', newline='') as f_out:
        writer = csv.writer(f_out, delimiter='\t')
        writer.writerow(["doc_id", "text"])

        count = 0
        with tqdm(total=sample_size, desc="Sampling corpus") as pbar:
            if not hasattr(dataset, 'docs_iter'):
                raise AttributeError("docs_iter not found on the dataset object.")

            for i, doc in enumerate(dataset.docs_iter()):
                if i in indices_to_keep:
                    writer.writerow([doc.doc_id, doc.text])
                    count += 1
                    pbar.update(1)

                if count == sample_size:
                    break

    logger.info("Successfully sampled %s documents.", f"{count:,}")


def sample_queries(output_dir: Path, sample_size: int, seed: int) -> None:
    """
    Loads and samples the MSMARCO v1 dev queries.
    Assumes data is pre-downloaded based on IR_DATASETS_HOME.
    """
    logger.info("Loading 'msmarco-passage/dev' queries via ir_datasets...")
    try:
        dataset = ir_datasets.load("msmarco-passage/dev")
        n_total_queries = dataset.queries_count()
        logger.info("Total queries in dev set: %s", f"{n_total_queries:,}")

    except Exception as e:
        logger.error(
            "Error loading dataset. Is IR_DATASETS_HOME set correctly and data downloaded? (%s)",
            e,
        )
        raise

    if sample_size > n_total_queries:
        raise ValueError(
            f"Requested {sample_size:,} queries but dataset only contains {n_total_queries:,} queries."
        )

    logger.info("Generating %s random indices (seed=%s)...", f"{sample_size:,}", seed)
    random.seed(seed)
    indices_to_keep = set(random.sample(range(n_total_queries), sample_size))

    out_path = output_dir / "queries.tsv"
    logger.info("Streaming queries and writing subsample to %s...", out_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    with out_path.open('w', encoding='utf-8', newline='') as f_out:
        writer = csv.writer(f_out, delimiter='\t')
        writer.writerow(["query_id", "text"])

        count = 0
        with tqdm(total=sample_size, desc="Sampling queries") as pbar:
            if not hasattr(dataset, 'queries_iter'):
                raise AttributeError("queries_iter not found on the dataset object.")

            for i, query in enumerate(dataset.queries_iter()):
                if i in indices_to_keep:
                    writer.writerow([query.query_id, query.text])
                    count += 1
                    pbar.update(1)

                if count == sample_size:
                    break

    logger.info("Successfully sampled %s queries.", f"{count:,}")

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger.info("--- MSMARCO v1 Subsampler ---")

    ir_home = _ensure_ir_datasets_home(config)

    if ir_home:
        logger.info("Using IR_DATASETS_HOME=%s", ir_home)
    else:
        logger.warning(
            "IR_DATASETS_HOME not set. ir_datasets will fall back to its default cache location."
        )

    output_dir = _resolve_output_dir(ir_home, config)
    logger.info("Output directory resolved to: %s", output_dir)

    seed = int(config.subset.SEED)
    corpus_sample_size = int(config.subset.CORPUS_SAMPLE_SIZE)
    query_sample_size = int(config.subset.QUERY_SAMPLE_SIZE)

    sample_corpus(output_dir, corpus_sample_size, seed)
    sample_queries(output_dir, query_sample_size, seed)

    logger.info("Subsampled data written to: %s", output_dir)

if __name__ == "__main__":
    main()
