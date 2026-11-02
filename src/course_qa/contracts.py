from dataclasses import dataclass

@dataclass(frozen=True)
class SourceDocument:
    """
    Represents a source document for the course notes QA assistant.
    """
    source_id: str
    title: str
    source_path: str
    project: str
    text: str


@dataclass(frozen=True)
class Chunk:
    """
    Represents a chunk of text extracted from a source document.
    """
    chunk_id: str
    text: str
    metadata: dict[str, str | int]


@dataclass(frozen=True)
class RetrievedPassage:
    """
    Represents a passage retrieved from the source documents based on a query.
    """
    chunk: Chunk
    distance: float
    rank: int


@dataclass(frozen=True)
class ContextBundle:
    """
    Represents a bundle of context information for the course notes QA assistant.
    """
    index_id: str
    text: str
    sources: dict[str, Chunk]
