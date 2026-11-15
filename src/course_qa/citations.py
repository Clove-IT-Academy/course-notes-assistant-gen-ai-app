"""Validate citation traceability and render references from indexed metadata.

Label validation cannot establish that a passage actually supports a claim.
Factual support still requires answer-quality evaluation.
"""

import json
import re
from dataclasses import dataclass

from course_qa.contracts import ContextBundle, parse_chunk_metadata
from course_qa.generation import (
    GenerationResult,
    InvalidGenerationError,
    parse_generation_response,
)

_CITATION = re.compile(r"\[(S[1-9][0-9]*)\]")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class SourceReference:
    label: str
    chunk_id: str
    index_id: str
    project: str
    title: str
    source_path: str
    line_start: int
    line_end: int
    source_hash: str
    page_number: int | None = None

    def render(self) -> str:
        """Describe the indexed source; do not claim the live file is unchanged.

        Terminal callers must escape control characters before printing this.
        """
        locator = (
            f"page {self.page_number}, extracted lines {self.line_start}-{self.line_end}"
            if self.page_number is not None
            else f"lines {self.line_start}-{self.line_end}"
        )
        return (
            f"[{self.label}] {self.project} / {self.source_path} / "
            f"{locator} — {self.title}"
        )


def validate_citations(
    result: GenerationResult, context: ContextBundle
) -> tuple[SourceReference, ...]:
    """Reject invalid answers before callers display any generated text.

    The plain-text answer format reserves square brackets for [S1] citations.
    Every nonempty paragraph of an answered response must cite supplied evidence.
    The response envelope and inline labels must agree exactly. Abstention has
    no citations. This function does not inspect or reopen original source files.
    """
    # Also enforce the envelope when a fake/custom adapter constructs the
    # dataclass directly instead of using the shared response parser.
    checked = parse_generation_response(
        json.dumps(
            {
                "status": result.status,
                "answer": result.answer,
                "cited_labels": result.cited_labels,
            }
        )
    )
    inline_labels = _CITATION.findall(checked.answer)
    remainder = _CITATION.sub("", checked.answer)
    if "[" in remainder or "]" in remainder:
        raise InvalidGenerationError("Answer contains malformed citation brackets.")

    if checked.status == "insufficient_evidence":
        if inline_labels:
            raise InvalidGenerationError(
                "An insufficient-evidence answer must not cite sources."
            )
        return ()

    if not context.index_id.strip() or not context.text.strip() or not context.sources:
        raise InvalidGenerationError("An answered response requires indexed evidence.")
    if set(inline_labels) != set(checked.cited_labels):
        raise InvalidGenerationError("Inline citations do not match cited_labels.")
    if any(label not in context.sources for label in inline_labels):
        raise InvalidGenerationError(
            "Answer cites a label absent from the supplied context."
        )
    for paragraph in _PARAGRAPH_BREAK.split(checked.answer):
        if paragraph.strip() and not _CITATION.search(paragraph):
            raise InvalidGenerationError(
                "Every answer paragraph must cite supplied evidence."
            )

    references: list[SourceReference] = []
    # Preserve first use in the answer rather than model-provided list order.
    for label in dict.fromkeys(inline_labels):
        chunk = context.sources[label]
        metadata = parse_chunk_metadata(chunk.metadata)
        references.append(
            SourceReference(
                label=label,
                chunk_id=chunk.chunk_id,
                index_id=context.index_id,
                project=metadata["project"],
                title=metadata["title"],
                source_path=metadata["source_path"],
                line_start=metadata["line_start"],
                line_end=metadata["line_end"],
                source_hash=metadata["source_hash"],
                page_number=metadata.get("page_number"),
            )
        )
    return tuple(references)
