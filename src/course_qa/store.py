import os
import re
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence
from uuid import uuid4

from course_qa.contracts import Chunk, RetrievedPassage
from course_qa.embeddings import validate_vectors

class SnapShotStore:
    COLLECTION_NAME = "course_notes"

    def __init__(self, storage_dir: Path) -> None:
        self.root = storage_dir.resolve()

    @contextmanager
    def locked(self) -> Iterator[None]:
        """
        Context manager to lock the store for exclusive access.
        """
        
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / ".lock"

        try:
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600
            )
        except FileExistsError:
            raise RuntimeError(
                "Storage is locked by another operation. "
                "After an abnormal stop, confirm no process is using "
                "the index before removing the stale lock."
            ) from None

        try:
            os.write(descriptor, str(os.getpid()).encode('ASCII'))
            yield
        finally:
            os.close(descriptor)
            lock_path.unlink()

    def _snapshot_dir(self, index_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", index_id):
            raise ValueError("Invalid snapshot identifier. Must be a 32-character hexadecimal string.")

        directory = (self.root / "indexes" / index_id).resolve()

        if not directory.is_relative_to(self.root):
            raise ValueError("Invalid snapshot identifier. Directory traversal is not allowed.")

        return directory

    def _client(self, directory: Path):
        import chromadb
        from chromadb.config import Settings

        return chromadb.PersistentClient(
            path=str(directory / "chroma"),
            settings=Settings(anonymized_telemetry=False)
        )

    def read_active_manifest(self) -> dict | None:
        pointer = self.root / "active.json"

        if not pointer.exists():
            return None

        index_id = json.loads(
            pointer.read_text(encoding="utf-8")
        )["index_id"]

        manifest_path = self._snapshot_dir(index_id) / "manifest.json"
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

        if (
            manifest.get("index_id") != index_id
            or manifest.get("status") != "ready"
            or manifest.get("schema_version") != 1
        ):
            raise ValueError("The active index manifest is invalid or corrupted.")

        return manifest

    def create_snapshot(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        embedding_fingerprint: str,
        dimension: int,
        build_key: str,
        batch_size: int,
    ) -> dict:
        """
        Build and verify a snapshot without publishing it.
        """
        if not chunks:
            raise ValueError("Cannot create and empty index.")

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
            name = self.COLLECTION_NAME,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
            metadatum={
                "embedding_fingerprint": embedding_fingerprint,
                "dimension": dimension,
            }
        )

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start+batch_size]
            batch_vectors = vectors[start:start+batch_size]

            collection.add(
                ids = [chunk.chunk_id for chunk in batch],
                documents = [chunk.text for chunk in batch],
                metadata = [chunk.metadata for chunk in batch],
                embeddings = [list(vector) for vector in batch_vectors],
            )

        if collection.count() != len(chunks):
            raise RuntimeError("Stored chunk is incorrect.")

        # verify that new collection can answer a vector query
        probe = collection.query(
            query_embeddings=[list(vectors[0])],
            n_result=1,
            include=["distances"],
        )

        if not probe["ids"][0]:
            raise RuntimeError("Snapshot retrieval verification failed.")

        manifest = {
            "schema_version": 1,
            "index_id": index_id,
            "status": "ready",
            "build_key": build_key,
            "embedding_fingerprint": embedding_fingerprint,
            "dimension": dimenstion,
            "distance_metric": "cosine",
            "chunk_count": len(chunks),
        }

        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        return manifest

    def publish(self, manifest: dict) -> None:
        """
        Atomatically change the pointer after the snapshot verification.
        """
        temporary = self.root / f"active-{uuid4().hex}.tmp"
        try:
            with temporary.open("w", encoding="utf-8") as f:
                json.dump({"index_id": manifest["index_id"]}, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, self.root / "active.json")
        finally:
            temporary.unlink(missing_ok=True)

    def open_active(
        self,
        embedding_fingerprint: str,
        dimension: int,
    ):
        "Open an existing index; never create one while querying."
        manifest = self.read_active_manifest()

        if manifest is None:
            raise FileNotFoundError("No active index. Ingest notes first.")

        if (
            manifest["embedding_fingerprint"] != embedding_fingerprint
            or manifest["dimension"] != dimension
            or manifest["distance_metric"] != "cosine"
        ):
            raise ValueError("Embedding Configuration changed. Please rebuild the index.")

        directory = self._snapshot_dir(manifest["index_id"])

        if not (directory / "chroma").is_dir():
            raise FileNotFoundError("The active chroma directory is missing.")

        client = self._client(directory)
        collection = client.get_collection(
            name=self.COLLECTION_NAME,
            embedding_function=None,
        )

        metadata = collection.metadatum or {}

        if (
            metadata.get("embedding_fingerprint") != embedding_fingerprint
            or metadata.get("dimension") != dimension
            or collection.count() != manifest["chunk_count"] 
        ):
            raise ValueError("Stored Collection does not match its manifest")

        return manifest, 
        
    def query(
        self,
        collection,
        vector: list[float],
        top_k: int,
        project: str | None = None,
    ) -> list[RetrievedPassage]:
        if top_k <= 0:
            raise ValueError("Top K must be positive.")

        count = collection.count()

        if count == 0:
            return []

        arguments = {
            "query_embeddings": [vector],
            "n_results": min(top_k, count),
            "include": ["documents", "metadata", "distances"]
        }

        if project is not None:
            arguments["where"] = {"project": project}

        result = collection.query(**arguments)

        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        passages = []

        for rank, values in enumerate(
            zip(ids, documents, metadatas, distances, strict=True),
            start=1,
        ):
            chunk_id, text, metadata, distance = values

            if text is None or metadata is None or distance is None:
                raise ValueError("Chroma returned an incomplete passage.")

            passages.append(
                RetrievedPassage(
                    chunk=Chunk(
                        chunk_id=chunk_id,
                        text=text,
                        metadata=metadata,
                    ),
                    distance=float(distance),
                    rank=rank,
                )
            )

        return passages
