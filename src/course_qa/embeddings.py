import hashlib
import json
import math
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray


class _TextEncoder(Protocol):
    def encode(
        self,
        inputs: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: Literal[True],
        show_progress_bar: bool,
        prompt: str,
    ) -> NDArray[np.float32]: ...


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
        raise NotImplementedError(
            "count_tokens method must be implemented by subclasses."
        )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Embed a list of documents and return their vector representations.
        """
        raise NotImplementedError(
            "embed_documents method must be implemented by subclasses."
        )

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query and return its vector representation.
        """
        raise NotImplementedError(
            "embed_query method must be implemented by subclasses."
        )


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


class LocalSentenceEmbedder:
    """
    Load one local model and use it for documents and questions.
    """

    def __init__(
        self,
        model_dir: Path,
        batch_size: int = 32,
        document_prefix: str = "",
        query_prefix: str = "",
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        model_dir = model_dir.resolve(strict=True)

        if not model_dir.is_dir():
            raise ValueError(f"Model directory does not exist: {model_dir}")

        # Optional dependency: Importing other modules does not load this
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            str(model_dir), local_files_only=True, trust_remote_code=False, device="cpu"
        )

        dimension = self.model.get_embedding_dimension()
        if dimension is None or dimension <= 0:
            raise ValueError("Model must declare a positive embedding dimension.")
        self.dimension = dimension

        max_tokens = self.model.max_seq_length
        if max_tokens is None or max_tokens <= 0:
            raise ValueError("Model must declare a positive token limit.")
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix

        identity: dict[str, str | int | bool] = {
            "adapter": "local-sentence-transformers-v1",
            "model_files": hash_model_directory(model_dir),
            "dimension": self.dimension,
            "normalize_embeddings": True,
            "document_prefix": self.document_prefix,
            "query_prefix": self.query_prefix,
            "sentence_transformers_version": version("sentence-transformers"),
            "transformers_version": version("transformers"),
            "torch_version": version("torch"),
        }

        serialized = json.dumps(identity, sort_keys=True)
        self.fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a given text.
        """
        tokens = self.model.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            # Counting may inspect oversized candidates before ingestion splits
            # them. _encode still rejects oversized inputs before inference.
            verbose=False,
        )

        return len(tokens["input_ids"])

    def _encode(
        self,
        texts: Sequence[str],
        prefix: str = "",
    ) -> list[list[float]]:
        """
        Encode a list of texts into embeddings, applying the specified prefix.
        """
        if not texts:
            return []

        prepared: list[str] = []

        for text in texts:
            if not text.strip():
                raise ValueError("Text cannot be empty.")

            value = prefix + text

            if self.count_tokens(value) > self.max_tokens:
                raise ValueError(
                    f"Text exceeds the maximum token limit of {self.max_tokens} tokens. "
                    "Use smaller chunks or token-aware chunking"
                )

            prepared.append(value)

        # Sentence Transformers has multimodal overloads with unknown input types.
        # Narrow this boundary to the documented text-to-NumPy call we use.
        encoder = cast(_TextEncoder, self.model)
        vectors: list[list[float]] = encoder.encode(
            prepared,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            prompt="",
        ).tolist()

        validate_vectors(
            vectors,
            expected_count=len(prepared),
            dimension=self.dimension,
        )

        return vectors

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        return self._encode(texts, prefix=self.document_prefix)

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        return self._encode([text], prefix=self.query_prefix)[0]
