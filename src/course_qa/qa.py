"""Coordinate one independent question without terminal I/O or silent ingestion."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from course_qa.citations import SourceReference, validate_citations
from course_qa.config import AppConfig
from course_qa.context import build_context
from course_qa.contracts import ContextBundle
from course_qa.embeddings import Embedder
from course_qa.excerpts import VerifiedExcerpt, verified_source_excerpts
from course_qa.generation import GenerationError, Generator
from course_qa.prompts import PROMPT_VERSION, build_prompt
from course_qa.retrieval import retrieve_passages


@dataclass(frozen=True)
class AnswerResult:
    status: Literal["answered", "insufficient_evidence", "evidence_selected"]
    answer: str
    sources: tuple[SourceReference, ...]
    index_id: str
    prompt_version: str
    generation_model_id: str
    generation_called: bool
    answer_mode: Literal["generated", "verified_excerpts"] = "generated"
    excerpts: tuple[VerifiedExcerpt, ...] = ()


def answer_question(
    question: str,
    config: AppConfig,
    embedder: Embedder,
    generator: Generator,
    *,
    count_request_tokens: Callable[[str, str], int],
    project: str | None = None,
    max_distance: float | None = None,
    answer_mode: Literal["generated", "verified_excerpts"] = "generated",
) -> AnswerResult:
    """Retrieve, budget, generate and validate; propagate operational failures.

    count_request_tokens(system_prompt, user_prompt) must count the generation
    model's FULL chat template using its actual tokenizer, without truncation.
    The same formatter must be used by the adapter. It is injected explicitly
    because counting two raw strings is not equivalent to counting chat tokens.
    There is no automatic generation retry, provider switch or answer repair.
    Citation validation checks traceability, not factual entailment.
    verified_excerpts discards generated explanations and returns hash-checked
    context passages as evidence_selected. It does not verify their relevance.
    """
    if answer_mode not in ("generated", "verified_excerpts"):
        raise ValueError("Unknown answer mode.")
    question = question.strip()
    if not question:
        raise ValueError("Enter a nonempty question.")
    if generator.count_tokens(question) > config.max_question_tokens:
        raise ValueError("The question exceeds the generation question-token limit.")

    index_id, passages = retrieve_passages(
        question, config, embedder, project=project, max_distance=max_distance
    )

    def insufficient(message: str) -> AnswerResult:
        return AnswerResult(
            status="insufficient_evidence",
            answer=message,
            sources=(),
            index_id=index_id,
            prompt_version=PROMPT_VERSION,
            generation_model_id=generator.model_id,
            generation_called=False,
            answer_mode=answer_mode,
        )

    if not passages:
        return insufficient("No passages matched the selected retrieval settings.")

    empty = build_prompt(question, ContextBundle(index_id, "", {}))
    reserved = count_request_tokens(empty.system_prompt, empty.user_prompt)
    if reserved < 0:
        raise ValueError("Request token counter returned a negative count.")
    if reserved + config.max_output_tokens >= generator.context_window_tokens:
        raise GenerationError(
            "Instructions, question and output reservation leave no evidence budget."
        )

    # Token counts are not additive across text boundaries. Recount the complete
    # formatted request and shrink the evidence budget if necessary. This loop
    # makes no model calls and removes whole passages rather than truncating them.
    while True:
        if reserved + config.max_output_tokens >= generator.context_window_tokens:
            return insufficient("No source passages fit the generation context budget.")
        context = build_context(
            index_id=index_id,
            passages=passages,
            config=config,
            count_tokens=generator.count_tokens,
            context_window_tokens=generator.context_window_tokens,
            reserved_input_tokens=reserved,
        )
        if not context.sources:
            return insufficient("No source passages fit the generation context budget.")
        prompt = build_prompt(question, context)
        request_tokens = count_request_tokens(prompt.system_prompt, prompt.user_prompt)
        if request_tokens < 0:
            raise ValueError("Request token counter returned a negative count.")
        overflow = (
            request_tokens + config.max_output_tokens - generator.context_window_tokens
        )
        if overflow <= 0:
            break
        reserved += max(overflow, 1)

    generated = generator.generate(
        system_prompt=prompt.system_prompt,
        user_prompt=prompt.user_prompt,
        max_output_tokens=config.max_output_tokens,
    )
    references = validate_citations(generated, context)
    if answer_mode == "verified_excerpts" and generated.status == "answered":
        # Keep all bounded context passages so a model's citation selection
        # cannot silently omit a supplied definition. Generated prose is discarded.
        excerpts = verified_source_excerpts(context)
        return AnswerResult(
            status="evidence_selected",
            answer="Selected source excerpts; no generated explanation is shown.",
            sources=tuple(excerpt.reference for excerpt in excerpts),
            index_id=index_id,
            prompt_version=prompt.version,
            generation_model_id=generator.model_id,
            generation_called=True,
            answer_mode=answer_mode,
            excerpts=excerpts,
        )
    return AnswerResult(
        status=generated.status,
        answer=(
            "The model did not identify sufficient evidence in the supplied notes."
            if answer_mode == "verified_excerpts"
            else generated.answer
        ),
        sources=references,
        index_id=index_id,
        prompt_version=prompt.version,
        generation_model_id=generator.model_id,
        generation_called=True,
        answer_mode=answer_mode,
    )
