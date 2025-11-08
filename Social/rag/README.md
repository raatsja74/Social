# RAG Ingestion Pipeline

This module turns the markdown knowledge base into a searchable vector store.

## Setup

Install dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the ingestion script from the repository root:

```bash
python -m rag.ingest
```

Key options:

- `--source-dirs`: Specific folders to index. Defaults to the curated knowledge base directories.
- `--include-root-files` / `--exclude-root-files`: Toggle inclusion of root-level markdown files.
- `--file-patterns`: Glob patterns for selecting files (defaults to `*.md`).
- `--chunk-size` and `--chunk-overlap`: Control chunk granularity (characters).
- `--model-name`: SentenceTransformer model used for embeddings.
- `--database-path`: Where the persistent Chroma database is stored (default `data/vector_store`).
- `--collection-name`: Name of the Chroma collection.
- `--dry-run`: Analyse the corpus without generating embeddings or touching the database.

The resulting vector store can be consumed by retrieval-augmented generation
workflows using Chroma's standard client APIs.
