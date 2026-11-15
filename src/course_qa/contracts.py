"""Shared records and the shapes of data crossing file/database boundaries."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict


class ChunkMetadata(TypedDict):
    source_id: str
    title: str
    project: str
    document_type: str
    source_path: str
    source_hash: str
    content_hash: str
    line_start: int
    line_end: int
    chunk_index: int
    processing_version: str
    page_number: NotRequired[int]


class SnapshotDetails(TypedDict):
    schema_version: int
    index_id: str
    build_key: str
    embedding_fingerprint: str
    dimension: int
    distance_metric: str
    chunk_count: int


class SnapshotManifest(SnapshotDetails):
    status: Literal["ready"]


class IngestSnapshotResult(SnapshotDetails):
    status: Literal["published", "unchanged"]


@dataclass(frozen=True)
class SourceDocument:
    """A source document for the course notes QA assistant."""

    source_id: str
    title: str
    source_path: str
    project: str
    text: str


@dataclass(frozen=True)
class Chunk:
    """A passage and its exact source metadata."""

    chunk_id: str
    text: str
    metadata: ChunkMetadata


@dataclass(frozen=True)
class RetrievedPassage:
    chunk: Chunk
    distance: float
    rank: int


@dataclass(frozen=True)
class ContextBundle:
    index_id: str
    text: str
    sources: dict[str, Chunk]


class IngestPreview(TypedDict):
    status: Literal["preview"]
    source_count: int
    chunk_count: int
    chunks: list[Chunk]


def require_string(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid text field: {key}")
    return value


def require_integer(data: Mapping[str, object], key: str, minimum: int = 0) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"Missing or invalid integer field: {key}")
    return value


def parse_chunk_metadata(data: Mapping[str, object]) -> ChunkMetadata:
    """Validate database values before exposing them as application records."""
    result: ChunkMetadata = {
        "source_id": require_string(data, "source_id"),
        "title": require_string(data, "title"),
        "project": require_string(data, "project"),
        "document_type": require_string(data, "document_type"),
        "source_path": require_string(data, "source_path"),
        "source_hash": require_string(data, "source_hash"),
        "content_hash": require_string(data, "content_hash"),
        "line_start": require_integer(data, "line_start", 1),
        "line_end": require_integer(data, "line_end", 1),
        "chunk_index": require_integer(data, "chunk_index"),
        "processing_version": require_string(data, "processing_version"),
    }
    if result["line_end"] < result["line_start"]:
        raise ValueError("A chunk's ending line precedes its starting line.")
    if "page_number" in data:
        result["page_number"] = require_integer(data, "page_number", 1)
    return result
