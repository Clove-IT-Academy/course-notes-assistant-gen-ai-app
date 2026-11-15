"""Copy indexed evidence without accepting model-authored explanations."""

import hashlib
import json
import re
from dataclasses import dataclass
from typing import cast

from course_qa.citations import SourceReference
from course_qa.contracts import ContextBundle, parse_chunk_metadata
from course_qa.generation import InvalidGenerationError


@dataclass(frozen=True)
class VerifiedExcerpt:
    text: str
    reference: SourceReference


def verified_source_excerpts(context: ContextBundle) -> tuple[VerifiedExcerpt, ...]:
    """Return whole supplied passages, not model prose or model-chosen citations.

    Verify each passage against its indexed content hash. References describe
    the indexed version; original files are not reopened. Exact copying does not
    establish relevance, completeness, source truth or evidence sufficiency.
    """
    if not context.index_id.strip() or not context.text.strip() or not context.sources:
        raise InvalidGenerationError("Source excerpts require indexed evidence.")
    try:
        payload: object = json.loads(context.text)
    except ValueError as error:
        raise InvalidGenerationError(
            "Source excerpt context must be valid JSON."
        ) from error
    if not isinstance(payload, dict):
        raise InvalidGenerationError("Source excerpt context must be an object.")
    data = cast(dict[str, object], payload)
    entries = data.get("evidence")
    if data.get("index_id") != context.index_id or not isinstance(entries, list):
        raise InvalidGenerationError("Source excerpt context does not match its index.")
    supplied: dict[str, dict[str, object]] = {}
    for entry in cast(list[object], entries):
        if not isinstance(entry, dict):
            raise InvalidGenerationError("Source excerpt evidence must be an object.")
        fields = cast(dict[str, object], entry)
        label = fields.get("source_label")
        if not isinstance(label, str) or label in supplied:
            raise InvalidGenerationError("Source excerpt evidence labels are invalid.")
        supplied[label] = fields
    if set(supplied) != set(context.sources):
        raise InvalidGenerationError(
            "Source excerpt labels do not match supplied evidence."
        )
    excerpts: list[VerifiedExcerpt] = []
    for label, chunk in context.sources.items():
        if re.fullmatch(r"S[1-9][0-9]*", label) is None:
            raise InvalidGenerationError("Source excerpt has an invalid label.")
        metadata = parse_chunk_metadata(chunk.metadata)
        digest = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
        if not chunk.text.strip() or digest != metadata["content_hash"]:
            raise InvalidGenerationError(
                "Source excerpt failed its indexed content hash."
            )
        fields = supplied[label]
        if fields.get("text") != chunk.text or any(
            fields.get(key) != metadata[key]
            for key in (
                "project",
                "title",
                "document_type",
                "source_path",
                "line_start",
                "line_end",
            )
        ):
            raise InvalidGenerationError(
                "Source excerpt differs from supplied evidence."
            )
        excerpts.append(
            VerifiedExcerpt(
                text=chunk.text,
                reference=SourceReference(
                    label=label,
                    chunk_id=chunk.chunk_id,
                    index_id=context.index_id,
                    project=metadata["project"],
                    title=metadata["title"],
                    source_path=metadata["source_path"],
                    line_start=metadata["line_start"],
                    line_end=metadata["line_end"],
                    source_hash=metadata["source_hash"],
                ),
            )
        )
    return tuple(excerpts)
