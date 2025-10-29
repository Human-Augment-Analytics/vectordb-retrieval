import ir_datasets
import random
import csv
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
SEED = 42  # Fixed seed for reproducibility
CORPUS_SAMPLE_SIZE = 1_000_000  # 1 million passages
QUERY_SAMPLE_SIZE = 1_000       # 1 thousand dev queries
OUTPUT_DIR = Path("./msmarco_subsample")
# ---------------------

def sample_corpus():
    """
    Downloads and samples the full MSMARCO v1 passage corpus.
    """
    print(f"Loading 'msmarco-passage' corpus...")
    # ir-datasets will handle downloading the 8.8M passage collection
    dataset = ir_datasets.load("msmarco-passage")
    n_total_docs = dataset.docs_count()
    print(f"Total documents in corpus: {n_total_docs:,}")
    
    # 1. Generate a reproducible set of indices to keep
    print(f"Generating {CORPUS_SAMPLE_SIZE:,} random indices (Seed={SEED})...")
    random.seed(SEED)
    indices_to_keep = set(random.sample(range(n_total_docs), CORPUS_SAMPLE_SIZE))
    
    out_path = OUTPUT_DIR / "corpus.tsv"
    print(f"Streaming corpus and writing subsample to {out_path}...")
    
    # 2. Iterate and write the subsample
    # We use csv.writer for robust TSV formatting (handles potential newlines/tabs)
    with out_path.open('w', encoding='utf-8', newline='') as f_out:
        writer = csv.writer(f_out, delimiter='\t')
        writer.writerow(["doc_id", "text"]) # Write header
        
        count = 0
        # Use tqdm for a progress bar
        with tqdm(total=CORPUS_SAMPLE_SIZE, desc="Sampling corpus") as pbar:
            for i, doc in enumerate(dataset.docs_iter()):
                if i in indices_to_keep:
                    writer.writerow([doc.doc_id, doc.text])
                    count += 1
                    pbar.update(1)
                
                # Optimization: Stop iterating after we've found all our samples
                if count == CORPUS_SAMPLE_SIZE:
                    break
                    
    print(f"Successfully sampled {count:,} documents.\n")

def sample_queries():
    """
    Downloads and samples the MSMARCO v1 dev queries.
    """
    print(f"Loading 'msmarco-passage/dev' queries...")
    # This loads the dev query set
    dataset = ir_datasets.load("msmarco-passage/dev")
    n_total_queries = dataset.queries_count()
    print(f"Total queries in dev set: {n_total_queries:,}")

    # 1. Generate reproducible indices
    # We set the *same* seed to ensure that if the script is run
    # in parts, it remains reproducible. The call to random.sample
    # will be different from the corpus one because n_total_queries
    # is different from n_total_docs.
    # For true independence, a different seed could be used,
    # but this is simpler and sufficient.
    print(f"Generating {QUERY_SAMPLE_SIZE:,} random indices (Seed={SEED})...")
    random.seed(SEED) 
    indices_to_keep = set(random.sample(range(n_total_queries), QUERY_SAMPLE_SIZE))

    out_path = OUTPUT_DIR / "queries.tsv"
    print(f"Streaming queries and writing subsample to {out_path}...")

    # 2. Iterate and write
    with out_path.open('w', encoding='utf-8', newline='') as f_out:
        writer = csv.writer(f_out, delimiter='\t')
        writer.writerow(["query_id", "text"]) # Write header
        
        count = 0
        with tqdm(total=QUERY_SAMPLE_SIZE, desc="Sampling queries") as pbar:
            for i, query in enumerate(dataset.queries_iter()):
                if i in indices_to_keep:
                    writer.writerow([query.query_id, query.text])
                    count += 1
                    pbar.update(1)
                
                if count == QUERY_SAMPLE_SIZE:
                    break
                    
    print(f"Successfully sampled {count:,} queries.\n")

def main():
    print("--- MSMARCO v1 Subsampler ---")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    sample_corpus()
    sample_queries()
    
    print(f"All done. Subsampled data is in: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
