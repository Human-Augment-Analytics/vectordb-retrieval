import csv
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import yaml
from box import ConfigBox
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "ms_marco_subset_embed.yaml"

logger = logging.getLogger(__name__)


def _load_config() -> ConfigBox:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return ConfigBox(yaml.safe_load(file))


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


def _resolve_paths(cfg: ConfigBox) -> Tuple[Path, Path]:
    embeddings_cfg = cfg.embeddings

    input_dir = (
        embeddings_cfg.get("INPUT_DIR")
        or embeddings_cfg.get("SUBSAMPLED_DATA_DIR")
        or cfg.subset.get("OUTPUT_DIR")
    )
    output_dir = embeddings_cfg.get("OUTPUT_DIR") or embeddings_cfg.get("OUTPUT_EMBEDDING_DIR")

    if not input_dir:
        raise ValueError("Missing INPUT_DIR/SUBSAMPLED_DATA_DIR/subset.OUTPUT_DIR in configuration.")
    if not output_dir:
        raise ValueError("Missing OUTPUT_DIR/OUTPUT_EMBEDDING_DIR in configuration.")

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    return input_path, output_path


def load_tsv_data(file_path: Path) -> Tuple[List[str], List[str]]:
    """Loads IDs and text from a TSV file."""
    ids: List[str] = []
    texts: List[str] = []

    if not file_path.is_file():
        raise FileNotFoundError(f"File not found at {file_path}")

    logger.info("Loading data from %s ...", file_path)

    with file_path.open('r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        try:
            next(reader)  # Skip header
        except StopIteration as exc:
            raise ValueError(f"File {file_path} appears to be empty.") from exc

        for row in tqdm(reader, desc=f"Reading {file_path.name}"):
            if len(row) == 2:
                ids.append(row[0])
                texts.append(row[1])
            else:
                logger.warning("Skipping malformed row in %s: %s", file_path.name, row)

    logger.info("Loaded %s items from %s.", f"{len(ids):,}", file_path.name)
    return ids, texts


def _save_metadata(
    output_dir: Path,
    *,
    passage_count: int,
    query_count: int,
    embedding_dim: int,
    model_name: str,
    ground_truth_precomputed: bool,
    ir_datasets_home: Optional[str],
    config_path: Path,
) -> None:
    metadata = {
        "passage_count": passage_count,
        "query_count": query_count,
        "embedding_dim": embedding_dim,
        "model_name": model_name,
        "ground_truth_precomputed": ground_truth_precomputed,
        "ir_datasets_home": ir_datasets_home,
        "config_path": str(config_path),
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Wrote metadata to %s", metadata_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger.info("--- MSMARCO v1 Embedding Generation ---")

    cfg = _load_config()
    ir_home = _ensure_ir_datasets_home(cfg)
    if ir_home:
        logger.info("Using IR_DATASETS_HOME=%s", ir_home)
    else:
        logger.warning(
            "IR_DATASETS_HOME not set. ir_datasets will fall back to its default cache location."
        )

    input_dir, output_dir = _resolve_paths(cfg)
    logger.info("Input directory: %s", input_dir)
    logger.info("Output directory: %s", output_dir)

    corpus_file = input_dir / "corpus.tsv"
    queries_file = input_dir / "queries.tsv"

    passage_ids, passage_texts = load_tsv_data(corpus_file)

    query_ids_raw, query_texts_raw = load_tsv_data(queries_file)
    query_ids = query_ids_raw
    query_texts = query_texts_raw
    logger.info(
        "Encoding all %s queries from queries.tsv without filtering.",
        f"{len(query_ids):,}",
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info("Using device: %s", device)
    if device == 'cpu':
        logger.warning("No GPU detected. Encoding will proceed on CPU.")

    model_name = cfg.embeddings.MODEL_NAME
    logger.info("Loading sentence transformer model: %s...", model_name)
    model = SentenceTransformer(model_name, device=device)

    logger.info("Encoding %s passages...", f"{len(passage_texts):,}")
    passage_embeddings = model.encode(
        passage_texts,
        batch_size=int(cfg.embeddings.BATCH_SIZE),
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32, copy=False)

    logger.info("Encoding %s queries...", f"{len(query_texts):,}")
    query_embeddings = model.encode(
        query_texts,
        batch_size=int(cfg.embeddings.BATCH_SIZE),
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32, copy=False)

    passage_emb_path = output_dir / "passage_embeddings.npy"
    passage_ids_path = output_dir / "passage_ids.npy"
    query_emb_path = output_dir / "query_embeddings.npy"
    query_ids_path = output_dir / "query_ids.npy"

    logger.info("Saving passage embeddings to %s", passage_emb_path)
    np.save(passage_emb_path, passage_embeddings)
    np.save(passage_ids_path, np.array(passage_ids, dtype=np.str_))

    logger.info("Saving query embeddings to %s", query_emb_path)
    np.save(query_emb_path, query_embeddings)
    np.save(query_ids_path, np.array(query_ids, dtype=np.str_))

    _save_metadata(
        output_dir,
        passage_count=len(passage_ids),
        query_count=len(query_ids),
        embedding_dim=passage_embeddings.shape[1],
        model_name=model_name,
        ground_truth_precomputed=False,
        ir_datasets_home=os.environ.get("IR_DATASETS_HOME"),
        config_path=CONFIG_PATH,
    )

    logger.info(
        "Completed embedding generation. Artifacts written to %s",
        output_dir,
    )


if __name__ == "__main__":
    main()
