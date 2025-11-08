# Social Knowledge Base Toolkit

This repository contains tooling for building a retrieval-augmented knowledge base
from markdown source material.

## Project Requirements Overview

- Maintain a searchable vector store backed by [Chroma](https://docs.trychroma.com/).
- Support ingesting curated markdown directories as well as root-level documents.
- Provide configurable chunking so downstream RAG applications can tune recall
  and latency for their workloads.
- Supply automated tests that cover the ingestion primitives to ensure future
  enhancements remain stable.

## Development Environment

1. Create an isolated Python environment (e.g. `python -m venv .venv`).
2. Activate the environment and install dependencies:
   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install pytest
   ```
3. Run the ingestion pipeline in dry-run mode to verify configuration:
   ```bash
   python -m rag.ingest --dry-run
   ```
4. Execute the automated test suite:
   ```bash
   pytest
   ```

## Repository Layout

- `rag/`: Ingestion pipeline and helpers for preparing the vector store.
- `tests/`: Automated tests that exercise the ingestion utilities.
- `TODO.md`: High-level checklist used to track recent project onboarding work.
