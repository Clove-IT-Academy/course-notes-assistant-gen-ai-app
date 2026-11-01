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
