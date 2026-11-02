from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class SourceConfig:
    source_id: str
    title: str
    project: str
    path: Path
    approved_root: Path
    document_type: str

    excluded_lines: tuple[tuple[int, int], ...] = ()

    def resolve_path(self) -> Path:
        """
        Resolve a source and confirm it stays inside its approved root.
        This is a security measure to prevent path traversal attacks.
        """
        root = self.approved_root.resolve(strict=True)
        path = self.path.resolve(strict=True)

        if not root.is_dir():
            raise ValueError("The approved source root must be a directory.")

        if not path.is_relative_to(root):
            raise ValueError("Source path is outside its approved root.")

        if not path.is_file():
            raise ValueError("Source path must point to a file.")

        if path.suffix.lower() not in {".md", ".txt"}:
            raise ValueError("This increment supports only Markdown and text.")

        if not all(
            value.strip()
            for value in (
                self.source_id,
                self.title,
                self.project,
                self.document_type,
            )
        ):
            raise ValueError("Source identifiers and descriptions are required.")

        previous_end = 0

        for start, end in self.excluded_lines:
            if start <= previous_end or end < start:
                raise ValueError(
                    "Excluded line ranges must be positive, sorted, "
                    "and non-overlapping."
                )

            previous_end = end
        
        return path


@dataclass(frozen=True)
class AppConfig:
    """
    Represents the configuration for the course notes QA assistant application.
    """
    storage_dir: Path

    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_batch_size: int = 32

    top_k: int = 6
    max_context_passages: int = 4

    max_source_files: int = 20
    max_file_bytes: int = 10 * 1024 * 1024
    max_total_file_bytes: int = 100 * 1024 * 1024
    max_corpus_characters: int = 2_000_000

    max_question_tokens: int = 500
    max_context_tokens: int = 4000
    max_output_tokens: int = 800

    def __post_init__(self) -> None:
        positive_values = (
            self.chunk_size,
            self.embedding_batch_size,
            self.top_k,
            self.max_context_passages,
            self.max_source_files,
            self.max_file_bytes,
            self.max_total_file_bytes,
            self.max_corpus_characters,
            self.max_question_tokens,
            self.max_context_tokens,
            self.max_output_tokens,
        )

        if any(value <= 0 for value in positive_values):
            raise ValueError("Size, count, and token limits must be positive.")

        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(
                "chunk_overlap must be between 0 and chunk_size - 1."
            )

        if self.max_context_passages > self.top_k:
            raise ValueError(
                "max_context_passages cannot exceed top_k."
            )
