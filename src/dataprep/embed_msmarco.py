import csv
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ir_datasets
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


def _build_ground_truth(
    query_ids: List[str],
    query_texts: List[str],
    passage_ids: List[str],
    ground_truth_k: int,
) -> Tuple[List[str], List[str], np.ndarray, Dict[str, List[str]], int]:
    """Return filtered queries, ground-truth indices (padded), and doc id mapping."""
    if ground_truth_k <= 0:
        raise ValueError("GROUND_TRUTH_K must be a positive integer.")

    dataset = ir_datasets.load("msmarco-passage/dev")
    id_to_index = {pid: idx for idx, pid in enumerate(passage_ids)}
    query_id_set = set(query_ids)

    index_map: Dict[str, List[int]] = defaultdict(list)
    docid_map: Dict[str, List[str]] = defaultdict(list)

    logger.info("Collecting qrels for %s sampled queries...", f"{len(query_ids):,}")
    for qrel in dataset.qrels_iter():
        if qrel.relevance <= 0:
            continue
        qid = qrel.query_id
        if qid not in query_id_set:
            continue
        doc_idx = id_to_index.get(qrel.doc_id)
        if doc_idx is None:
            continue
        index_map[qid].append(doc_idx)
        docid_map[qid].append(qrel.doc_id)

    filtered_ids: List[str] = []
    filtered_texts: List[str] = []
    ground_truth_rows: List[List[int]] = []
    docid_rows: Dict[str, List[str]] = {}
    dropped = 0

    for qid, text in zip(query_ids, query_texts):
        positives = index_map.get(qid, [])
        positive_doc_ids = docid_map.get(qid, [])
        if not positives:
            dropped += 1
            continue

        unique_indices: List[int] = []
        unique_doc_ids: List[str] = []
        seen: set[int] = set()

        for idx, doc_id in zip(positives, positive_doc_ids):
            if idx in seen:
                continue
            seen.add(idx)
            unique_indices.append(idx)
            unique_doc_ids.append(doc_id)
            if len(unique_indices) >= ground_truth_k:
                break

        if not unique_indices:
            dropped += 1
            continue

        padded = list(unique_indices)
        while len(padded) < ground_truth_k:
            padded.append(padded[-1])

        filtered_ids.append(qid)
        filtered_texts.append(text)
        ground_truth_rows.append(padded[:ground_truth_k])
        docid_rows[qid] = unique_doc_ids[:ground_truth_k]

    if not ground_truth_rows:
        raise ValueError(
            "No queries retained after aligning qrels with the subsampled corpus. "
            "Increase the corpus sample size or adjust the query subset."
        )

    ground_truth = np.asarray(ground_truth_rows, dtype=np.int32)
    logger.info(
        "Retained %s queries with positives (dropped %s).",
        f"{len(filtered_ids):,}",
        f"{dropped:,}",
    )
    return filtered_ids, filtered_texts, ground_truth, docid_rows, dropped


def _save_metadata(
    output_dir: Path,
    *,
    passage_count: int,
    query_count: int,
    embedding_dim: int,
    model_name: str,
    ground_truth_k: int,
    dropped_queries: int,
    ir_datasets_home: Optional[str],
    config_path: Path,
) -> None:
    metadata = {
        "passage_count": passage_count,
        "query_count": query_count,
        "embedding_dim": embedding_dim,
        "model_name": model_name,
        "ground_truth_k": ground_truth_k,
        "dropped_queries": dropped_queries,
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

    ground_truth_k = int(cfg.embeddings.get("GROUND_TRUTH_K", 100))
    query_ids_raw, query_texts_raw = load_tsv_data(queries_file)

    (
        query_ids,
        query_texts,
        ground_truth,
        ground_truth_doc_ids,
        dropped_queries,
    ) = _build_ground_truth(
        query_ids_raw,
        query_texts_raw,
        passage_ids,
        ground_truth_k,
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

    if query_embeddings.shape[0] != ground_truth.shape[0]:
        raise ValueError(
            "Mismatch between number of encoded queries and ground-truth rows: "
            f"{query_embeddings.shape[0]} vs {ground_truth.shape[0]}"
        )

    passage_emb_path = output_dir / "passage_embeddings.npy"
    passage_ids_path = output_dir / "passage_ids.npy"
    query_emb_path = output_dir / "query_embeddings.npy"
    query_ids_path = output_dir / "query_ids.npy"
    ground_truth_path = output_dir / "ground_truth.npy"
    ground_truth_doc_ids_path = output_dir / "ground_truth_doc_ids.json"

    logger.info("Saving passage embeddings to %s", passage_emb_path)
    np.save(passage_emb_path, passage_embeddings)
    np.save(passage_ids_path, np.array(passage_ids, dtype=np.str_))

    logger.info("Saving query embeddings to %s", query_emb_path)
    np.save(query_emb_path, query_embeddings)
    np.save(query_ids_path, np.array(query_ids, dtype=np.str_))

    logger.info("Saving ground truth to %s", ground_truth_path)
    np.save(ground_truth_path, ground_truth)
    ground_truth_doc_ids_path.write_text(
        json.dumps(ground_truth_doc_ids, indent=2),
        encoding="utf-8",
    )

    _save_metadata(
        output_dir,
        passage_count=len(passage_ids),
        query_count=len(query_ids),
        embedding_dim=passage_embeddings.shape[1],
        model_name=model_name,
        ground_truth_k=ground_truth_k,
        dropped_queries=dropped_queries,
        ir_datasets_home=os.environ.get("IR_DATASETS_HOME"),
        config_path=CONFIG_PATH,
    )

    logger.info(
        "Completed embedding generation. Artifacts written to %s",
        output_dir,
    )


if __name__ == "__main__":
    main()
