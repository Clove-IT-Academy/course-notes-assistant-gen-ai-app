import math
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Protocol, Sequence

class Embedder(Protocol):
    """
    This is an interface used by ingestion and retrieval.
    """

    fingerprint: str
    dimension: int

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a given text.
        """
        raise NotImplementedError("count_tokens method must be implemented by subclasses.")

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
       """
        Embed a list of documents and return their vector representations.
        """
        raise NotImplementedError("embed_documents method must be implemented by subclasses.")

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query and return its vector representation.
        """
        raise NotImplementedError("embed_query method must be implemented by subclasses.")


def validate_vectors(
    vectors: Sequence[Sequence[float]],
    expected_count: int,
    dimension: int,
) -> None:
    """
    Validate the shape of the vectors returned by the embedder.
    """
    if len(vectors) != expected_count:
        raise ValueError(
            "Embedding Count does not match the expected count. "
            f"Expected {expected_count} vectors, but got {len(vectors)}."
        )

    for vector in vectors:
        if len(vector) != dimension:
            raise ValueError(
                "Embedding dimension does not match the model dimension. "
                f"Expected vector dimension {dimension}, but got {len(vector)}."
            )

        if not all(math.isfinite(value) for value in vector):
            raise ValueError("Embedding contains a non-finite number.")

        if not any(value != 0 for value in vector):
            raise ValueError("A zero vector cannot support cosine retrieval.")


def hash_model_directory(model_dir: Path) -> str:
    """
    Generate a hash for the model directory to uniquely identify the model.
    Fingerprint the local model files so that the changed weights, change identity of the model.
    This is useful for caching and ensuring that the correct model is used.
    """

    digest = hashlib.sha256()
    files = sorted(path for path in model_dir.rglob("*") if path.is_file())

    if not files:
        raise ValueError(f"No files found in the model directory: {model_dir}")

    for path in files:
        # Use as_posix() to ensure consistent path separators across platforms
        relative_name = path.relative_to(model_dir).as_posix()

        # Update the hash with the file's relative path and a null byte to separate entries
        digest.update(relative_name.encode("utf-8"))
        digest.update(b"\0")

        file_digest = hashlib.sha256()

        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                file_digest.update(block)

        # Update the main digest with the file's hash and a null byte to separate entries
        digest.update(file_digest.digest())
    
    return digest.hexdigest()
    
