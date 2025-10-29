import numpy as np
import csv
import torch
import yaml

from sentence_transformers import SentenceTransformer
from pathlib import Path
from tqdm import tqdm
from box import ConfigBox

# load config -----
with open("config/config_a3c.yml", "r") as file:
    config = ConfigBox(yaml.safe_load(file))


def load_tsv_data(file_path: Path):
    """Loads IDs and text from a TSV file."""
    
    ids = []
    texts = []
    print(f"Loading data from {file_path}...")
    
    if not file_path.is_file():
        print(f"Error: File not found at {file_path}")
        return None, None
        
    with file_path.open('r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        try:
            header = next(reader) # skip header
        except StopIteration:
            print(f"Error: File {file_path} appears to be empty.")
            return None, None
            
        for row in tqdm(reader, desc=f"Reading {file_path.name}"):
            if len(row) == 2: # check for id and text
                 ids.append(row[0])
                 texts.append(row[1])
            else:
                print(f"Warning: Skipping malformed row in {file_path.name}: {row}")
                
    print(f"Loaded {len(ids):,} items.")
    return ids, texts

def main():
    print("--- MSMARCO v1 Embedding Generation ---")
    
    # --- Setup ---
    config.embeddings.OUTPUT_EMBEDDING_DIR.mkdir(exist_ok=True)
    print(f"Output directory: {config.embeddings.OUTPUT_EMBEDDING_DIR}")

    corpus_file = config.embeddings.SUBSAMPLED_DATA_DIR / "corpus.tsv"
    queries_file = config.embeddings.SUBSAMPLED_DATA_DIR / "queries.tsv"

    # --- Determine Device ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cpu':
        print("Warning: No GPU detected, will use CPU.")

    # --- Load Model ---
    print(f"Loading sentence transformer model: {config.embeddings.MODEL_NAME}...")
    model = SentenceTransformer(config.embeddings.MODEL_NAME, device=device)
    print("Sentence Bert loaded.")

    # --- Process Corpus ---
    passage_ids, passage_texts = load_tsv_data(corpus_file)
    if passage_ids is None:
        return # Exit if loading failed

    print("\nGenerating passage embeddings...")
    
    passage_embeddings = model.encode(
        passage_texts,
        batch_size=config.embeddings.BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    print(f"Generated passage embeddings shape: {passage_embeddings.shape}")

    # Save passage embeddings and IDs
    passage_emb_path = config.embeddings.OUTPUT_EMBEDDING_DIR / "passage_embeddings.npy"
    passage_ids_path = config.embeddings.OUTPUT_EMBEDDING_DIR / "passage_ids.npy"
    
    print(f"Saving passage embeddings to {passage_emb_path}...")
    np.save(passage_emb_path, passage_embeddings.astype(np.float32)) # save for faiss, float32
    
    print(f"Saving passage IDs to {passage_ids_path}...")
    np.save(passage_ids_path, np.array(passage_ids))
    
    print("Passage data saved.")
 
    # --- Process Queries ---
    query_ids, query_texts = load_tsv_data(queries_file)
    if query_ids is None:
        return print("Failed loading queries") # Exit if loading failed

    print("\nGenerating query embeddings...")
    query_embeddings = model.encode(
        query_texts,
        batch_size=config.embeddings.BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    print(f"Generated query embeddings shape: {query_embeddings.shape}")

    # Save query embeddings and IDs
    query_emb_path = config.embeddings.OUTPUT_EMBEDDING_DIR / "query_embeddings.npy"
    query_ids_path = config.embeddings.OUTPUT_EMBEDDING_DIR / "query_ids.npy"
    
    print(f"Saving query embeddings to {query_emb_path}...")
    np.save(query_emb_path, query_embeddings.astype(np.float32)) # Save as float32
    
    print(f"Saving query IDs to {query_ids_path}...")
    np.save(query_ids_path, np.array(query_ids))
    
    print("Query data saved.")

    print("Embeddings and IDs saved in:", config.embeddings.OUTPUT_EMBEDDING_DIR)

if __name__ == "__main__":
    main()
