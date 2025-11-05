"""Build a local vector store from the markdown knowledge base.

This script reads markdown documents across the repository, extracts the
YAML front matter as metadata, chunks the markdown body, generates
embeddings and stores the chunks in a Chroma vector database. The
resulting database can be consumed by custom RAG applications.
"""
from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import chromadb
import yaml
from bs4 import BeautifulSoup
from markdown_it import MarkdownIt
from sentence_transformers import SentenceTransformer


@dataclass
class Document:
    """A markdown file with its metadata."""

    path: Path
    metadata: Dict[str, str]
    text: str


@dataclass
class Chunk:
    """A chunk of text prepared for vector storage."""

    document: Document
    index: int
    text: str

    @property
    def id(self) -> str:
        digest = hashlib.md5(str(self.document.path).encode("utf-8")).hexdigest()
        return f"{digest}-{self.index}"


def extract_front_matter(raw_text: str) -> tuple[Dict[str, str], str]:
    """Split YAML front matter from the rest of the markdown document."""

    front_matter_pattern = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)
    match = front_matter_pattern.match(raw_text)
    if not match:
        return {}, raw_text

    raw_front_matter, body = match.groups()
    metadata = yaml.safe_load(raw_front_matter) or {}
    return metadata, body


def markdown_to_text(markdown: str) -> str:
    """Convert markdown to clean plain text."""

    markdown_parser = MarkdownIt()
    html = markdown_parser.render(markdown)
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n")
    # Normalise whitespace and collapse multiple newlines.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text into overlapping chunks measured in characters."""

    if not text:
        return []

    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    chunks: List[str] = []
    start = 0
    text_length = len(normalized)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            boundary = normalized.rfind(" ", start, end)
            if boundary == -1 or boundary <= start:
                boundary = normalized.find(" ", end)
                if boundary == -1:
                    boundary = text_length
            end = boundary

        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def sanitize_metadata(metadata: Dict[str, object]) -> Dict[str, str]:
    """Ensure metadata is serialisable by the vector store."""

    serialisable: Dict[str, str] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            serialisable[key] = str(value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            serialisable[key] = ", ".join(str(item) for item in value)
        else:
            serialisable[key] = str(value)
    return serialisable


def load_markdown_documents(paths: Iterable[Path]) -> List[Document]:
    documents: List[Document] = []
    for path in paths:
        raw_text = path.read_text(encoding="utf-8")
        metadata, body = extract_front_matter(raw_text)
        plain_text = markdown_to_text(body)
        document = Document(path=path, metadata=sanitize_metadata(metadata), text=plain_text)
        documents.append(document)
    return documents


def collect_markdown_files(root: Path, source_dirs: Sequence[str], include_root: bool) -> List[Path]:
    """Gather markdown files from the configured folders."""

    markdown_files: List[Path] = []

    for directory in source_dirs:
        dir_path = (root / directory).resolve()
        if not dir_path.exists():
            continue
        markdown_files.extend(sorted(dir_path.glob("**/*.md")))

    if include_root:
        markdown_files.extend(sorted(root.glob("*.md")))

    # Deduplicate while preserving order.
    seen = set()
    unique_files = []
    for path in markdown_files:
        if path in seen:
            continue
        seen.add(path)
        unique_files.append(path)
    return unique_files


def build_chunks(documents: Iterable[Document], chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    for document in documents:
        text_chunks = chunk_text(document.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for index, text in enumerate(text_chunks):
            chunks.append(Chunk(document=document, index=index, text=text))
    return chunks


def embed_chunks(chunks: Sequence[Chunk], model_name: str) -> List[List[float]]:
    """Generate embeddings for each chunk."""

    if not chunks:
        return []

    model = SentenceTransformer(model_name)
    payload = [chunk.text for chunk in chunks]
    return model.encode(payload, show_progress_bar=True, convert_to_numpy=False).tolist()


def store_chunks(
    chunks: Sequence[Chunk],
    embeddings: Sequence[Sequence[float]],
    collection_name: str,
    database_path: Path,
    repository_root: Path,
) -> None:
    database_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(database_path))
    collection = client.get_or_create_collection(collection_name)

    ids = [chunk.id for chunk in chunks]
    documents = [chunk.text for chunk in chunks]
    metadatas = []
    for chunk in chunks:
        metadata = dict(chunk.document.metadata)
        metadata.update({
            "source": str(chunk.document.path.relative_to(repository_root)),
            "chunk_index": str(chunk.index),
        })
        metadatas.append(metadata)

    collection.upsert(ids=ids, metadatas=metadatas, documents=documents, embeddings=embeddings)


def run_ingestion(
    repository_root: Path,
    source_dirs: Sequence[str],
    include_root_files: bool,
    chunk_size: int,
    chunk_overlap: int,
    model_name: str,
    collection_name: str,
    database_path: Path,
) -> None:
    markdown_files = collect_markdown_files(repository_root, source_dirs, include_root_files)
    if not markdown_files:
        raise SystemExit("No markdown files found for ingestion.")

    documents = load_markdown_documents(markdown_files)
    chunks = build_chunks(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise SystemExit("No chunks were produced from the provided documents.")

    embeddings = embed_chunks(chunks, model_name=model_name)
    store_chunks(
        chunks,
        embeddings,
        collection_name=collection_name,
        database_path=database_path,
        repository_root=repository_root,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest markdown documents into a Chroma vector store.")
    parser.add_argument(
        "--source-dirs",
        nargs="*",
        default=[
            "RAG",
            "Prompts",
            "Templates",
            "Automations",
            "Market Research",
            "Outreach",
            "Case Studies",
            "mega_prompts_linkedin",
        ],
        help="Folders (relative to the repository root) that contain markdown files.",
    )
    parser.add_argument(
        "--include-root-files",
        dest="include_root_files",
        action="store_true",
        help="Include markdown files located at the repository root (default).",
    )
    parser.add_argument(
        "--exclude-root-files",
        dest="include_root_files",
        action="store_false",
        help="Skip markdown files located at the repository root.",
    )
    parser.set_defaults(include_root_files=True)
    parser.add_argument("--chunk-size", type=int, default=900, help="Target chunk size in characters.")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Overlap between consecutive chunks in characters.")
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model used for embeddings.",
    )
    parser.add_argument(
        "--collection-name",
        default="social-knowledge-base",
        help="Name of the Chroma collection that will store the chunks.",
    )
    parser.add_argument(
        "--database-path",
        default="data/vector_store",
        help="Directory where the Chroma database will be stored.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    repository_root = Path(__file__).resolve().parent.parent
    database_path = repository_root / args.database_path

    run_ingestion(
        repository_root=repository_root,
        source_dirs=args.source_dirs,
        include_root_files=args.include_root_files,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        model_name=args.model_name,
        collection_name=args.collection_name,
        database_path=database_path,
    )


if __name__ == "__main__":
    main()
