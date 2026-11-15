import json
from collections.abc import Callable, Sequence

from course_qa.config import AppConfig
from course_qa.contracts import (
    Chunk,
    ContextBundle,
    RetrievedPassage,
)


def _substantially_overlaps(
    candidate: RetrievedPassage,
    selected: RetrievedPassage,
) -> bool:
    """Detect overlapping chunks from the same source snapshot."""
    left = candidate.chunk.metadata
    right = selected.chunk.metadata

    if (
        left["source_id"] != right["source_id"]
        or left["source_hash"] != right["source_hash"]
        or left.get("page_number") != right.get("page_number")
    ):
        return False

    left_start = int(left["line_start"])
    left_end = int(left["line_end"])
    right_start = int(right["line_start"])
    right_end = int(right["line_end"])

    overlap = max(
        0,
        min(left_end, right_end) - max(left_start, right_start) + 1,
    )

    shorter_length = min(
        left_end - left_start + 1,
        right_end - right_start + 1,
    )

    # Avoid treating an entire long single-line document as one duplicate.
    if shorter_length < 3:
        return False

    return overlap / shorter_length >= 0.8


def build_context(
    index_id: str,
    passages: Sequence[RetrievedPassage],
    config: AppConfig,
    count_tokens: Callable[[str], int],
    context_window_tokens: int,
    reserved_input_tokens: int,
) -> ContextBundle:
    """
    Select whole passages that fit the generation model's input budget.

    reserved_input_tokens must cover the question, system instructions,
    message framing, and any text outside this evidence block.
    """
    if context_window_tokens <= 0 or reserved_input_tokens < 0:
        raise ValueError("Invalid generation-model token budget.")

    available_tokens = min(
        config.max_context_tokens,
        context_window_tokens - reserved_input_tokens - config.max_output_tokens,
    )

    if available_tokens <= 0:
        raise ValueError("No token budget remains for evidence.")

    selected: list[RetrievedPassage] = []
    sources: dict[str, Chunk] = {}
    blocks: list[str] = []
    seen_content: set[str] = set()

    for passage in passages:
        if len(selected) >= config.max_context_passages:
            break

        chunk = passage.chunk
        metadata = chunk.metadata
        content_hash = str(metadata["content_hash"])

        if content_hash in seen_content:
            continue

        if any(_substantially_overlaps(passage, previous) for previous in selected):
            continue

        label = f"S{len(selected) + 1}"

        # JSON keeps metadata values and passage text explicitly separated.
        payload: dict[str, str | int] = {
            "source_label": label,
            "project": metadata["project"],
            "title": metadata["title"],
            "document_type": metadata["document_type"],
            "source_path": metadata["source_path"],
            "line_start": metadata["line_start"],
            "line_end": metadata["line_end"],
            "text": chunk.text,
        }
        if "page_number" in metadata:
            payload["page_number"] = metadata["page_number"]

        block = json.dumps(payload, ensure_ascii=False)
        candidate_blocks = [*blocks, block]

        candidate_text = json.dumps(
            {
                "index_id": index_id,
                "evidence": [json.loads(value) for value in candidate_blocks],
            },
            ensure_ascii=False,
            indent=2,
        )

        if count_tokens(candidate_text) > available_tokens:
            continue

        blocks.append(block)
        selected.append(passage)
        sources[label] = chunk
        seen_content.add(content_hash)

    if not selected:
        return ContextBundle(
            index_id=index_id,
            text="",
            sources={},
        )

    context_text = json.dumps(
        {
            "index_id": index_id,
            "evidence": [json.loads(value) for value in blocks],
        },
        ensure_ascii=False,
        indent=2,
    )

    return ContextBundle(
        index_id=index_id,
        text=context_text,
        sources=sources,
    )
