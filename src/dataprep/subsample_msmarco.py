import ir_datasets
import random
import csv
import os
import yaml

from pathlib import Path
from tqdm import tqdm
from box import ConfigBox

# load config -----
with open("config/ms_marco_subset_embed.yml", "r") as file:
    config = ConfigBox(yaml.safe_load(file))

# SET UP IR_DATASETS HOME -----
# We load the downloaded dataset from the common drive so that we can save time on downloading it again every time that we want
# to subsample it in a different way

ir_datasets_home_str = os.environ.get('IR_DATASETS_HOME') # this has to be set from the slurm job

if ir_datasets_home_str:
    ir_datasets_home_path = Path(ir_datasets_home_str)
    
    # Go up one level from 'ms_marco_v1_raw' and create 'msmarco_v1_subsampled' there
    OUTPUT_DIR = ir_datasets_home_path.parent / "msmarco_v1_subsampled"

else:
    # Default fallback if IR_DATASETS_HOME is not set 
    # This is not recommended because it will pollute your local home directory on ICE
    
    OUTPUT_DIR = Path("./msmarco_v1_subsampled")

# ----

def sample_corpus():
    """
    Loads and samples the full MSMARCO v1 passage corpus.
    Assumes data is pre-downloaded based on IR_DATASETS_HOME.
    """
    print(f"Loading 'msmarco-passage' corpus...")
    
    # ir-datasets will try to use IR_DATASETS_HOME to find the data
    
    try:
        dataset = ir_datasets.load("msmarco-passage")
        n_total_docs = dataset.docs_count()
        print(f"Total documents in corpus: {n_total_docs:,}")
        
    except Exception as e:
        print(f"Error loading dataset. Is IR_DATASETS_HOME set correctly and data downloaded?")
        print(f"Error details: {e}")
        return # Exit the function if dataset loading fails

    # 1. Set up seed
    print(f"Generating {config.subset.CORPUS_SAMPLE_SIZE:,} random indices (Seed={config.subset.SEED})...")
    random.seed(config.subset.SEED)
    indices_to_keep = set(random.sample(range(n_total_docs), config.subset.CORPUS_SAMPLE_SIZE))

    out_path = OUTPUT_DIR / "corpus.tsv"
    print(f"Streaming corpus and writing subsample to {out_path}...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Iterate and write the subsample
    with out_path.open('w', encoding='utf-8', newline='') as f_out:
        writer = csv.writer(f_out, delimiter='\t')
        writer.writerow(["doc_id", "text"]) # Write header

        count = 0
        with tqdm(total=config.subset.CORPUS_SAMPLE_SIZE, desc="Sampling corpus") as pbar:
            # Check if docs_iter exists before iterating
            if not hasattr(dataset, 'docs_iter'):
                print("Error: docs_iter not found on the dataset object.")
                return

            # go through all the indices and write only the sampled ones
            for i, doc in enumerate(dataset.docs_iter()):
                if i in indices_to_keep:
                    writer.writerow([doc.doc_id, doc.text])
                    count += 1
                    pbar.update(1)
                
                # stop once we hit the desired number of documents
                if count == config.subset.CORPUS_SAMPLE_SIZE:
                    break

    print(f"Successfully sampled {count:,} documents.\n")

def sample_queries():
    """
    Loads and samples the MSMARCO v1 dev queries.
    Assumes data is pre-downloaded based on IR_DATASETS_HOME.
    """
    print(f"Loading 'msmarco-passage/dev' queries...")
    try:
        dataset = ir_datasets.load("msmarco-passage/dev")
        n_total_queries = dataset.queries_count()
        print(f"Total queries in dev set: {n_total_queries:,}")
        
    except Exception as e:
        print(f"Error loading dataset. Is IR_DATASETS_HOME set correctly and data downloaded?")
        print(f"Error details: {e}")
        return

    # 1. Generate reproducible indices
    print(f"Generating {config.subset.QUERY_SAMPLE_SIZE:,} random indices (Seed={config.subset.SEED})...")
    random.seed(config.subset.SEED)
    indices_to_keep = set(random.sample(range(n_total_queries), config.subset.QUERY_SAMPLE_SIZE))

    out_path = OUTPUT_DIR / "queries.tsv"
    print(f"Streaming queries and writing subsample to {out_path}...")

    # Ensure output directory exists before writing
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Iterate and write
    with out_path.open('w', encoding='utf-8', newline='') as f_out:
        writer = csv.writer(f_out, delimiter='\t')
        writer.writerow(["query_id", "text"]) # Write header

        count = 0
        with tqdm(total=config.subset.QUERY_SAMPLE_SIZE, desc="Sampling queries") as pbar:
            if not hasattr(dataset, 'queries_iter'):
                 print("Error: queries_iter not found on the dataset object.")
                 return

            for i, query in enumerate(dataset.queries_iter()):
                if i in indices_to_keep:
                    writer.writerow([query.query_id, query.text])
                    count += 1
                    pbar.update(1)

                if count == config.subset.QUERY_SAMPLE_SIZE:
                    break

    print(f"Successfully sampled {count:,} queries.\n")

def main():
    print("--- MSMARCO v1 Subsampler ---")

    # Check and print the IR_DATASETS_HOME environment variable
    ir_datasets_home_str = os.environ.get('IR_DATASETS_HOME') # this has to be set from the slurm job (hardcoded)
    
    if ir_datasets_home_str:
        print(f"Using IR_DATASETS_HOME: {ir_datasets_home_str}")
        print(f"Output directory will be: {OUTPUT_DIR}") # Show output path
    else:
        print("Warning: IR_DATASETS_HOME environment variable not set.")
        print(f"Output directory will be relative: {OUTPUT_DIR}")

    # Run the random sampling from the documents and the sampling from the queries
    sample_corpus()
    sample_queries()

    print(f"Subsampled data should be in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

