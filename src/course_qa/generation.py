"""Provider-independent generation contract and strict response parsing.

A concrete adapter must enforce its model's token limits, bounded execution and
transport policy. This module neither loads a model nor makes network calls.
Citation membership and paragraph-level citation checks belong to citations.py.
"""

import json
from dataclasses import dataclass
from typing import Literal, Protocol, cast


class GenerationError(RuntimeError):
    """Generation failed; do not present this as insufficient evidence."""


class InvalidGenerationError(GenerationError):
    """The model response does not satisfy the application's output contract."""


@dataclass(frozen=True)
class GenerationResult:
    status: Literal["answered", "insufficient_evidence"]
    answer: str
    cited_labels: tuple[str, ...]


class Generator(Protocol):
    """Interface for one explicitly selected generation model.

    count_tokens counts complete text without truncation. generate must check
    the complete formatted request plus output reservation against the model's
    context limit. It must not run tools or silently switch models/providers.
    Implementations parse model output with parse_generation_response before
    returning it. The Q&A service subsequently validates citations.
    """

    model_id: str
    context_window_tokens: int

    def count_tokens(self, text: str) -> int: ...

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
    ) -> GenerationResult: ...


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidGenerationError(
                "Generation response contains duplicate fields."
            )
        result[key] = value
    return result


def parse_generation_response(raw: str) -> GenerationResult:
    """Accept a single JSON object; reject prose wrappers and unknown fields.

    Expected shape:
    {"status": "answered", "answer": "Supported explanation [S1].",
     "cited_labels": ["S1"]}

    Abstention requires a nonempty explanation and an empty citation list.
    Validation errors never include the raw model response.
    """
    if not raw.strip():
        raise InvalidGenerationError("Generation response is empty.")
    try:
        decoded: object = json.loads(raw, object_pairs_hook=_unique_object)
    except (ValueError, RecursionError) as error:
        raise InvalidGenerationError(
            "Generation response must be valid JSON."
        ) from error
    if not isinstance(decoded, dict):
        raise InvalidGenerationError("Generation response must be a JSON object.")
    data = cast(dict[str, object], decoded)
    if set(data) != {"status", "answer", "cited_labels"}:
        raise InvalidGenerationError(
            "Generation response has missing or unexpected fields."
        )

    status = data["status"]
    if status not in ("answered", "insufficient_evidence"):
        raise InvalidGenerationError("Generation response has an invalid status.")
    answer = data["answer"]
    if not isinstance(answer, str) or not answer.strip():
        raise InvalidGenerationError("Generation response requires a nonempty answer.")
    raw_labels = data["cited_labels"]
    if not isinstance(raw_labels, list):
        raise InvalidGenerationError("cited_labels must be a list.")

    labels: list[str] = []
    for label in cast(list[object], raw_labels):
        if (
            not isinstance(label, str)
            or not label.startswith("S")
            or not label[1:]
            or not label[1:].isascii()
            or not label[1:].isdigit()
            or label[1] == "0"
        ):
            raise InvalidGenerationError(
                "Citation labels must use the form S1, S2, ..."
            )
        if label in labels:
            raise InvalidGenerationError(
                "Generation response repeats a citation label."
            )
        labels.append(label)

    if status == "answered":
        if not labels:
            raise InvalidGenerationError("An answered response must cite evidence.")
        return GenerationResult("answered", answer.strip(), tuple(labels))
    if labels:
        raise InvalidGenerationError(
            "An insufficient-evidence response must not cite labels."
        )
    return GenerationResult("insufficient_evidence", answer.strip(), ())
