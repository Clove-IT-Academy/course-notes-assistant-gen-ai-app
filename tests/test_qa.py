"""Q&A orchestration regressions using fakes, never model downloads/inference.

These tests verify control flow and traceability. They do not measure factual
support; development answer reports and instructor review cover that separately.
"""

import hashlib
import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from course_qa.config import AppConfig
from course_qa.context import build_context
from course_qa.contracts import Chunk, RetrievedPassage
from course_qa.generation import (
    GenerationError,
    GenerationResult,
    InvalidGenerationError,
)
from course_qa.qa import AnswerResult, answer_question


class FakeEmbedder:
    fingerprint = "test-embedding-space"
    dimension = 2

    def count_tokens(self, text: str) -> int:
        return len(text)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise AssertionError("Answering must not ingest documents.")

    def embed_query(self, text: str) -> list[float]:
        raise AssertionError("Retrieval is replaced at the service boundary here.")


class FakeGenerator:
    model_id = "test-generator"
    context_window_tokens = 12_000

    def __init__(self) -> None:
        self.response = GenerationResult(
            "answered", "The switch is green [S1].", ("S1",)
        )
        self.error: GenerationError | None = None
        self.calls: list[tuple[str, str, int]] = []

    def count_tokens(self, text: str) -> int:
        # Character counts are a deterministic stand-in, not a real tokenizer.
        return len(text)

    def count_request_tokens(self, system: str, user: str) -> int:
        return len(f"<system>{system}</system><user>{user}</user><assistant>")

    def generate(
        self, *, system_prompt: str, user_prompt: str, max_output_tokens: int
    ) -> GenerationResult:
        self.calls.append((system_prompt, user_prompt, max_output_tokens))
        if self.error is not None:
            raise self.error
        return self.response


def passage(text: str = "The switch is green.", number: int = 1) -> RetrievedPassage:
    digest = hashlib.sha256(text.encode()).hexdigest()
    return RetrievedPassage(
        chunk=Chunk(
            chunk_id=f"chunk-{number}",
            text=text,
            metadata={
                "source_id": f"notes-{number}",
                "title": "Sample notes",
                "project": "course",
                "document_type": "student_notes",
                "source_path": f"notes-{number}.md",
                "source_hash": digest,
                "content_hash": digest,
                "processing_version": "test-v1",
                "line_start": 1,
                "line_end": 2,
                "chunk_index": 0,
            },
        ),
        distance=0.1,
        rank=number,
    )


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    return AppConfig(storage_dir=tmp_path, max_output_tokens=40)


@pytest.fixture
def retrieval() -> Iterator[Mock]:
    with patch(
        "course_qa.qa.retrieve_passages", return_value=("snapshot", [passage()])
    ) as mocked:
        yield mocked


def ask(
    config: AppConfig,
    generator: FakeGenerator,
    *,
    question: str = "What color is the switch?",
    counter: Callable[[str, str], int] | None = None,
) -> AnswerResult:
    return answer_question(
        question,
        config,
        FakeEmbedder(),
        generator,
        count_request_tokens=counter or generator.count_request_tokens,
        project="course",
    )


def test_answer_maps_citations_to_stored_metadata(
    config: AppConfig, retrieval: Mock
) -> None:
    generator = FakeGenerator()
    result = ask(config, generator)
    assert result.status == "answered" and result.generation_called
    assert len(generator.calls) == 1
    assert generator.calls[0][2] == config.max_output_tokens
    assert result.index_id == "snapshot"
    assert result.sources[0].chunk_id == "chunk-1"
    assert result.sources[0].source_path == "notes-1.md"
    assert result.sources[0].line_start == 1
    assert retrieval.call_args.kwargs["project"] == "course"


def test_no_passages_skips_generation(config: AppConfig, retrieval: Mock) -> None:
    retrieval.return_value = ("snapshot", [])
    generator = FakeGenerator()
    result = ask(config, generator)
    assert result.status == "insufficient_evidence"
    assert not result.generation_called and not result.sources
    assert not generator.calls


def test_model_abstention_has_no_sources(config: AppConfig, retrieval: Mock) -> None:
    generator = FakeGenerator()
    generator.response = GenerationResult(
        "insufficient_evidence", "The notes do not identify the author.", ()
    )
    result = ask(config, generator)
    assert result.status == "insufficient_evidence" and result.generation_called
    assert not result.sources and len(generator.calls) == 1


@pytest.mark.parametrize(
    "response",
    [
        GenerationResult("answered", "Green [S999].", ("S999",)),
        GenerationResult("answered", "Green.", ("S1",)),
        GenerationResult("answered", "Green [S1].", ("S2",)),
        GenerationResult("answered", "Green [S1].\n\nUncited claim.", ("S1",)),
        GenerationResult("insufficient_evidence", "Unknown [S1].", ()),
    ],
)
def test_invalid_citations_never_return_an_answer(
    config: AppConfig, retrieval: Mock, response: GenerationResult
) -> None:
    generator = FakeGenerator()
    generator.response = response
    with pytest.raises(InvalidGenerationError):
        ask(config, generator)
    assert len(generator.calls) == 1  # No silent repair generation.


def test_generation_failure_is_not_abstention(
    config: AppConfig, retrieval: Mock
) -> None:
    generator = FakeGenerator()
    error = GenerationError("Simulated incomplete generation")
    generator.error = error
    with pytest.raises(GenerationError) as raised:
        ask(config, generator)
    assert raised.value is error and len(generator.calls) == 1


def test_retrieval_failure_stops_before_generation(
    config: AppConfig, retrieval: Mock
) -> None:
    retrieval.side_effect = ValueError("Embedding configuration changed")
    generator = FakeGenerator()
    with pytest.raises(ValueError, match="Embedding configuration changed"):
        ask(config, generator)
    assert not generator.calls


@pytest.mark.parametrize("question", [" ", "x" * 501])
def test_bad_question_stops_before_retrieval(
    config: AppConfig, retrieval: Mock, question: str
) -> None:
    generator = FakeGenerator()
    with pytest.raises(ValueError):
        ask(config, generator, question=question)
    retrieval.assert_not_called()
    assert not generator.calls


def test_insufficient_instruction_budget_is_an_error(
    config: AppConfig, retrieval: Mock
) -> None:
    generator = FakeGenerator()
    generator.context_window_tokens = config.max_output_tokens
    with pytest.raises(GenerationError, match="leave no evidence budget"):
        ask(config, generator)
    assert not generator.calls


def test_no_whole_passage_fits_skips_generation(
    config: AppConfig, retrieval: Mock
) -> None:
    generator = FakeGenerator()
    result = ask(replace(config, max_context_tokens=1), generator)
    assert result.status == "insufficient_evidence" and not result.generation_called
    assert not generator.calls


def test_full_request_recount_removes_whole_passages_without_retry(
    config: AppConfig, retrieval: Mock
) -> None:
    passages = [passage(), passage("The switch uses a mechanical lever.", 2)]
    retrieval.return_value = ("snapshot", passages)
    both = build_context("snapshot", passages, config, len, 20_000, 0)
    generator = FakeGenerator()
    generator.context_window_tokens = len(both.text) + 150
    counts: list[int] = []

    def count_complete_request(system: str, user: str) -> int:
        # Simulate non-additive formatting overhead absent from the empty bundle.
        payload = user.partition("Evidence (application-prepared JSON):\n")[2]
        evidence: object = json.JSONDecoder().raw_decode(payload)[0]
        encoded = json.dumps(evidence, ensure_ascii=False, indent=2)
        tokens = 100 if '"evidence": []' in encoded else len(encoded) + 300
        counts.append(tokens)
        return tokens

    result = ask(config, generator, counter=count_complete_request)
    assert result.status == "answered" and len(generator.calls) == 1
    assert len(counts) >= 3  # Empty request, overflowing request, reduced request.
    assert counts[1] + config.max_output_tokens > generator.context_window_tokens
    assert counts[-1] + config.max_output_tokens <= generator.context_window_tokens
    assert "mechanical lever" not in generator.calls[0][1]
    assert passages[0].chunk.text in generator.calls[0][1]
