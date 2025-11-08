from pathlib import Path
import sys
import textwrap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.ingest import (
    IngestionStats,
    Chunk,
    collect_markdown_files,
    run_ingestion,
    sanitize_metadata,
)


def test_collect_markdown_files_respects_patterns(tmp_path):
    root: Path = tmp_path
    docs_dir = root / "docs"
    docs_dir.mkdir()
    notes_dir = root / "notes"
    notes_dir.mkdir()

    (docs_dir / "guide.md").write_text("content")
    (docs_dir / "ignore.txt").write_text("skip")
    (notes_dir / "summary.markdown").write_text("content")
    (root / "overview.md").write_text("content")
    (root / "overview.txt").write_text("skip")

    collected = collect_markdown_files(
        root=root,
        source_dirs=["docs", "notes"],
        include_root=True,
        file_patterns=["*.md", "*.markdown"],
    )

    relative = [path.relative_to(root) for path in collected]
    assert relative == [Path("docs/guide.md"), Path("notes/summary.markdown"), Path("overview.md")]


def test_run_ingestion_returns_stats_and_chunks(tmp_path):
    root: Path = tmp_path
    docs_dir = root / "docs"
    docs_dir.mkdir()

    markdown = textwrap.dedent(
        """
        ---
        title: Sample Doc
        tags:
          - example
          - ingestion
        ---

        # Heading

        This is a paragraph that should be chunked appropriately by the ingestion
        pipeline. It contains multiple sentences to ensure the chunking logic has
        enough material to work with.
        """
    ).lstrip()
    (docs_dir / "sample.md").write_text(markdown)

    stats, chunks = run_ingestion(
        repository_root=root,
        source_dirs=["docs"],
        include_root_files=False,
        file_patterns=["*.md"],
        chunk_size=120,
        chunk_overlap=20,
    )

    assert isinstance(stats, IngestionStats)
    assert stats.document_count == 1
    assert stats.chunk_count == len(chunks)
    assert [path.relative_to(root) for path in stats.source_files] == [Path("docs/sample.md")]
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert {chunk.document.metadata.get("title") for chunk in chunks} == {"Sample Doc"}


def test_sanitize_metadata_converts_sequences():
    metadata = {
        "title": "Example",
        "tags": ["one", "two"],
        "count": 3,
        "ignore": None,
        "mapping": {"a": 1},
    }

    sanitized = sanitize_metadata(metadata)

    assert sanitized["title"] == "Example"
    assert sanitized["tags"] == "one, two"
    assert sanitized["count"] == "3"
    assert sanitized["mapping"] == "{'a': 1}"
    assert "ignore" not in sanitized
