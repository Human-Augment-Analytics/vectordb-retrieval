---
configs:
- config_name: "passages"
  data_files:
  - split: train
    path: passages_parquet/*
- config_name: "queries"
  data_files:
  - split: test
    path: queries_parquet/*
---


# TREC-RAG 2024 Corpus (MSMARCO 2.1) - Encoded with Cohere Embed English v3


This dataset contains the embeddings for the [TREC-RAG Corpus 2024](https://trec-rag.github.io/annoucements/2024-corpus-finalization/) embedded with the [Cohere Embed V3 English](https://cohere.com/blog/introducing-embed-v3) model.

It contains embeddings for 113,520,750 passages, embeddings for 1677 queries from TREC-Deep Learning 2021-2023, as well as top-1000 hits for all queries using a brute-force (flat) index.

## Search over the Index

We have a pre-build index that only requires 300 MB available at [TREC-RAG-2024-index](https://huggingface.co/datasets/Cohere/trec-rag-2024-index). Just pass in your Cohere API key, and you are able to search across 113M passages.

The linked index used PQ-compression with memory-mapped IVF, reducing your memory need to only 300MB, while achieving 97% search quality compared to a float32 flat index (that requires 250+GB memory and is extremely slow).


## Passages 

### Passages - Parquet
113,520,750 passages are embedded. The parquet files can be found in the folder `passages_parquet`. Each row is a passage from the corpus. The column `emb` contains the respective embedding.

You can stream the dataset for example like this:
```python
from datasets import load_dataset

dataset = load_dataset("Cohere/msmarco-v2.1-embed-english-v3", "passages", split="train", streaming=True)

for row in dataset:
    print(row)
    break
```

### Passages - JSONL and Numpy
The folder `passages_jsonl` contain the `.json.gz` files for the passages as distributed by the task organizers.

The folder `passages_npy` contains a numpy matrix with all the embeddings for the respective `.json.gz` file.

When your server has enough memory, you can load all doc embeddings like this:
```python
import numpy as np
import glob

emb_paths = sorted(glob.glob("passages_npy/*.npy"))

for e_path in emb_paths:
  doc_emb = np.load(e_path)
```

## Queries 

For 1677 queries from TREC-Deep Learning 2021, 2022 and 2023 we compute the embedding and the respective top-1k hits from a brute-force (flat) index. 
These queries can e.g. be used to test different ANN setting, e.g. in Recall@10 scenarios.

We also added annotations from NIST for the 215 queries that received an annotation. These queries have a non-empty qrel column.

The format is the following:
- "_id": The query ID
- "text": Query text
- "trec-year": TREC-Deep Learning year 
- "emb": Cohere Embed V3 embedding
- "top1k_offsets": Passage ID (int) when the numpy matrices are loaded sequentially and vertically stacked
- "top1k_passage_ids": Passage ID (string) as they appear in the dataset 
- "top1k_cossim": Cosine similarities
- "qrels": Relevance annotations for the 215 annotated queries by NIST. The **document relevance** scores are provided. You can get the doc_id for a passage via `row['_id'].split("#")[0]` 

### Queries - JSONL
The folder `queries_jsonl/` contains the queries in a `.jsonl.gz` format.

Note: qrels are provided here as a dictionary lookup, while in the parquet format as a list in the format `[doc_id, score]` due to the limited support for dictionaries in parquet.

### Queries - Parquet

If you want to use the parquet file or the HF datasets library, the folder `queries_parquet/` contains the respective parquet file.

You can load the queries with the following command in HF datasets

```python
from datasets import load_dataset

dataset = load_dataset("Cohere/msmarco-v2.1-embed-english-v3", "queries", split="test")

for row in dataset:
    print(row)
    break
```


# License

The embeddings are provided as Apache 2.0. The text data, qrels etc. are provided following the license of MSMARCO v2.1
