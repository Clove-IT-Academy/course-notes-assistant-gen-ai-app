"""Local Chroma snapshots with validated application records."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable
from uuid import uuid4

from course_qa.contracts import (
    Chunk,
    RetrievedPassage,
    SnapshotManifest,
    parse_chunk_metadata,
    require_integer,
    require_string,
)
from course_qa.embeddings import validate_vectors

if TYPE_CHECKING:
    from chromadb import Collection
    from chromadb.api import ClientAPI
    from chromadb.api.types import Metadata, Where


@runtime_checkable
class _ClosableClient(Protocol):
    def close(self) -> None: ...


def _read_object(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Stored JSON must be an object.")
    # JSON object keys are always strings; field values are validated below.
    return cast(dict[str, object], value)


def _parse_manifest(data: Mapping[str, object]) -> SnapshotManifest:
    if (
        data.get("schema_version") != 1
        or data.get("status") != "ready"
        or data.get("distance_metric") != "cosine"
    ):
        raise ValueError("The index manifest is invalid or incompatible.")
    return {
        "schema_version": 1,
        "status": "ready",
        "index_id": require_string(data, "index_id"),
        "build_key": require_string(data, "build_key"),
        "embedding_fingerprint": require_string(data, "embedding_fingerprint"),
        "dimension": require_integer(data, "dimension", 1),
        "distance_metric": "cosine",
        "chunk_count": require_integer(data, "chunk_count", 1),
    }


class SnapshotStore:
    COLLECTION_NAME = "course_notes"

    def __init__(self, storage_dir: Path) -> None:
        self.root = storage_dir.resolve()
        self._clients: list[_ClosableClient] = []

    def close(self) -> None:
        """Release clients before unlocking the embedded database."""
        try:
            for client in self._clients:
                client.close()
        finally:
            self._clients.clear()

    @contextmanager
    def locked(self) -> Generator[None]:
        """Use every read/build/query operation inside this process lock."""
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / ".lock"
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            raise RuntimeError(
                "Storage is locked. After an abnormal stop, confirm no process "
                "is using the index before removing a stale lock."
            ) from None
        try:
            os.write(descriptor, str(os.getpid()).encode("ascii"))
            yield
        finally:
            try:
                self.close()
            finally:
                os.close(descriptor)
                lock_path.unlink(missing_ok=True)

    def _snapshot_dir(self, index_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", index_id):
            raise ValueError("Invalid snapshot identifier.")
        directory = (self.root / "indexes" / index_id).resolve()
        if not directory.is_relative_to(self.root):
            raise ValueError("Snapshot path escapes storage.")
        return directory

    def _client(self, directory: Path) -> ClientAPI:
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(
            path=str(directory / "chroma"),
            settings=Settings(anonymized_telemetry=False),
        )
        # Chroma 1.5.9 implements close(), but ClientAPI omits it.
        if not isinstance(client, _ClosableClient):
            raise TypeError("Installed Chroma client does not support close().")
        self._clients.append(client)
        return client

    def read_active_manifest(self) -> SnapshotManifest | None:
        pointer = self.root / "active.json"
        if not pointer.exists():
            return None
        index_id = require_string(_read_object(pointer), "index_id")
        manifest = _parse_manifest(
            _read_object(self._snapshot_dir(index_id) / "manifest.json")
        )
        if manifest["index_id"] != index_id:
            raise ValueError("The active pointer does not match its manifest.")
        return manifest

    def create_snapshot(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        embedding_fingerprint: str,
        dimension: int,
        build_key: str,
        batch_size: int,
    ) -> SnapshotManifest:
        """Build and verify an isolated snapshot without publishing it."""
        if not chunks:
            raise ValueError("Cannot create an empty index.")
        if batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("Chunk IDs must be unique.")
        validate_vectors(vectors, len(chunks), dimension)
        index_id = uuid4().hex
        directory = self._snapshot_dir(index_id)
        directory.mkdir(parents=True, exist_ok=False)
        client = self._client(directory)
        collection = client.create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={
                "embedding_fingerprint": embedding_fingerprint,
                "dimension": dimension,
            },
        )
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            # Convert our narrow TypedDict into Chroma's wider Metadata type.
            metadatas: list[Metadata] = []
            for chunk in batch:
                checked = parse_chunk_metadata(chunk.metadata)
                metadatas.append(
                    {
                        "source_id": checked["source_id"],
                        "title": checked["title"],
                        "project": checked["project"],
                        "document_type": checked["document_type"],
                        "source_path": checked["source_path"],
                        "source_hash": checked["source_hash"],
                        "content_hash": checked["content_hash"],
                        "line_start": checked["line_start"],
                        "line_end": checked["line_end"],
                        "chunk_index": checked["chunk_index"],
                        "processing_version": checked["processing_version"],
                    }
                )
            collection.add(
                ids=[chunk.chunk_id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                metadatas=metadatas,
                embeddings=[
                    list(vector) for vector in vectors[start : start + batch_size]
                ],
            )
        if collection.count() != len(chunks):
            raise RuntimeError("Stored chunk count is incorrect.")
        probe = collection.query(
            query_embeddings=[list(vectors[0])],
            n_results=1,
            include=["distances"],
        )
        if not probe["ids"][0]:
            raise RuntimeError("Snapshot retrieval verification failed.")
        manifest: SnapshotManifest = {
            "schema_version": 1,
            "index_id": index_id,
            "status": "ready",
            "build_key": build_key,
            "embedding_fingerprint": embedding_fingerprint,
            "dimension": dimension,
            "distance_metric": "cosine",
            "chunk_count": len(chunks),
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return manifest

    def publish(self, manifest: SnapshotManifest) -> None:
        temporary = self.root / f"active-{uuid4().hex}.tmp"
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump({"index_id": manifest["index_id"]}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.root / "active.json")
        finally:
            temporary.unlink(missing_ok=True)

    def open_active(
        self,
        embedding_fingerprint: str,
        dimension: int,
    ) -> tuple[SnapshotManifest, Collection]:
        manifest = self.read_active_manifest()
        if manifest is None:
            raise FileNotFoundError("No active index. Ingest notes first.")
        if (
            manifest["embedding_fingerprint"] != embedding_fingerprint
            or manifest["dimension"] != dimension
        ):
            raise ValueError("Embedding configuration changed. Rebuild the index.")
        directory = self._snapshot_dir(manifest["index_id"])
        if not (directory / "chroma").is_dir():
            raise FileNotFoundError("The active Chroma directory is missing.")
        collection = self._client(directory).get_collection(
            name=self.COLLECTION_NAME,
            embedding_function=None,
        )
        metadata = collection.metadata or {}
        if (
            metadata.get("embedding_fingerprint") != embedding_fingerprint
            or metadata.get("dimension") != dimension
            or collection.count() != manifest["chunk_count"]
        ):
            raise ValueError("Stored collection does not match its manifest.")
        return manifest, collection

    def query(
        self,
        collection: Collection,
        vector: list[float],
        top_k: int,
        project: str | None = None,
    ) -> list[RetrievedPassage]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        count = collection.count()
        if count == 0:
            return []
        where: Where | None = {"project": project} if project is not None else None
        result = collection.query(
            query_embeddings=[vector],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
            where=where,
        )
        documents = result["documents"]
        metadatas = result["metadatas"]
        distances = result["distances"]
        if documents is None or metadatas is None or distances is None:
            raise ValueError("Chroma returned an incomplete result.")
        passages: list[RetrievedPassage] = []
        for rank, (chunk_id, text, metadata, distance) in enumerate(
            zip(
                result["ids"][0], documents[0], metadatas[0], distances[0], strict=True
            ),
            1,
        ):
            if not math.isfinite(distance):
                raise ValueError("Chroma returned an invalid passage.")
            passages.append(
                RetrievedPassage(
                    chunk=Chunk(chunk_id, text, parse_chunk_metadata(metadata)),
                    distance=distance,
                    rank=rank,
                )
            )
        return passages


# Preserve the original teaching spelling for existing callers.
SnapShotStore = SnapshotStore
