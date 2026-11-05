import hashlib
import json
from pathlib import Path
from typing import Sequence

from course_qa.chunking import chunk_text
from course_qa.cleaning import clean_text
from course_qa.config import AppConfig, SourceConfig
from course_qa.contracts import Chunk
from course_qa.embeddings import Embedder
from course_qa.loaders import load_text_file
from course_qa.store import SnapshotStore

def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _included_segments(
    normalized_text: str,
    excluded_lines: tuple[tuple[int, int], ...],
):
    """
    Yield contigous included text with its original starting line.
    """
    lines = normalized_text.splitlines(keepends=True)

    if excluded_lines and excluded_lines[-1][1] > len(lines):
        raise ValueError("An excluded line range exceeds the document.")

    cursor = 0

    for first, last in excluded_lines:
        segment = "".join(lines[cursor:first - 1])

        if segment.strip():
            yield cursor + 1, segment

        cursor = last

    remaining = "".join(lines[cursor:])

    if remaining.strip():
        yield cursor + 1, remaining


def prepare_chunks(
    sources: Sequence[SourceConfig],
    config: AppConfig,
) -> list[Chunk]:
    """Validate approved text sources and create traceable chunks."""
    if not sources:
        raise ValueError("Select at least one source.")

    if len(sources) > config.max_source_files:
        raise ValueError("Too many source files.")

    source_ids = [source.source_id for source in sources]

    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Each source_id must be unique.")

    resolved_sources: list[tuple[SourceConfig, Path]] = []
    seen_paths = set()
    total_bytes = 0

    # Validate the whole selection before reading document text.
    for source in sources:
        path = source.resolve_path()

        if path in seen_paths:
            raise ValueError("The same source file was selected twice.")

        seen_paths.add(path)
        size = path.stat().st_size

        if size > config.max_file_bytes:
            raise ValueError("A selected source exceeds the file-size limit.")

        total_bytes += size
        resolved_sources.append((source, path))

    if total_bytes > config.max_total_file_bytes:
        raise ValueError("Selected files exceed the total-size limit.")

    chunks = []
    total_characters = 0
    step = config.chunk_size - config.chunk_overlap

    for source, path in resolved_sources:
        raw_text = load_text_file(str(path))
        normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n")

        total_characters += len(normalized)

        if total_characters > config.max_corpus_characters:
            raise ValueError("The corpus exceeds the character limit.")

        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        relative_path = path.relative_to(
            source.approved_root.resolve()
        ).as_posix()

        source_chunk_count = 0

        for first_line, segment in _included_segments(
            normalized,
            source.excluded_lines,
        ):
            cleaned = clean_text(segment)

            # clean_text removes outer newline characters.
            removed_prefix = len(segment) - len(segment.lstrip("\n"))
            first_cleaned_line = (
                first_line + segment[:removed_prefix].count("\n")
            )

            pieces = chunk_text(
                cleaned,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )

            for position, piece in enumerate(pieces):
                if not piece.strip():
                    continue

                start = position * step
                end = start + len(piece)

                line_start = (
                    first_cleaned_line + cleaned[:start].count("\n")
                )
                line_end = (
                    first_cleaned_line + cleaned[:end].count("\n")
                )

                # A trailing newline does not include the following line.
                if piece.endswith("\n"):
                    line_end -= 1

                metadata = {
                    "source_id": source.source_id,
                    "title": source.title,
                    "project": source.project,
                    "document_type": source.document_type,
                    "source_path": relative_path,
                    "source_hash": source_hash,
                    "content_hash": _hash_text(piece),
                    "line_start": line_start,
                    "line_end": line_end,
                    "chunk_index": source_chunk_count,
                    "processing_version": "text-character-chunks-v1",
                }

                identity = json.dumps(
                    {
                        "metadata": metadata,
                        "chunk_size": config.chunk_size,
                        "chunk_overlap": config.chunk_overlap,
                    },
                    sort_keys=True,
                )

                chunks.append(
                    Chunk(
                        chunk_id=_hash_text(identity),
                        text=piece,
                        metadata=metadata,
                    )
                )

                source_chunk_count += 1

        if source_chunk_count == 0:
            raise ValueError(
                f"No usable text remains in source: {source.source_id}"
            )

    return chunks


def ingest(
    sources: Sequence[SourceConfig],
    config: AppConfig,
    embedder: Embedder | None = None,
    dry_run: bool = False,
) -> dict:
    """Prepare, embed, and publish a complete knowledge-base snapshot."""
    chunks = prepare_chunks(sources, config)

    if dry_run:
        return {
            "status": "preview",
            "source_count": len(sources),
            "chunk_count": len(chunks),
            "chunks": chunks,
        }

    if embedder is None:
        raise ValueError("An explicit embedding adapter is required.")

    build_identity = {
        "processing_version": "text-character-chunks-v1",
        "embedding_fingerprint": embedder.fingerprint,
        "dimension": embedder.dimension,
        "chunk_ids": sorted(chunk.chunk_id for chunk in chunks),
    }

    build_key = _hash_text(
        json.dumps(build_identity, sort_keys=True)
    )

    store = SnapshotStore(config.storage_dir)

    with store.locked():
        active = store.read_active_manifest()

        if active is not None and active["build_key"] == build_key:
            # Verify the existing index before declaring it reusable.
            store.open_active(embedder.fingerprint, embedder.dimension)

            return {
                "status": "unchanged",
                **active,
            }

        vectors = []

        for start in range(
            0,
            len(chunks),
            config.embedding_batch_size,
        ):
            batch = chunks[start:start + config.embedding_batch_size]

            vectors.extend(
                embedder.embed_documents(
                    [chunk.text for chunk in batch]
                )
            )

        manifest = store.create_snapshot(
            chunks=chunks,
            vectors=vectors,
            embedding_fingerprint=embedder.fingerprint,
            dimension=embedder.dimension,
            build_key=build_key,
            batch_size=config.embedding_batch_size,
        )

        store.publish(manifest)

        return {
            **manifest,
            "status": "published",
        }
