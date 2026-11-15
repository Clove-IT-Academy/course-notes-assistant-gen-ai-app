"""Verified-copy behavior is independent of the model's free-form claims."""

import hashlib
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from course_qa.config import AppConfig
from course_qa.context import build_context
from course_qa.contracts import Chunk, ContextBundle, RetrievedPassage
from course_qa.evaluation import EvidenceRange, RetrievalCase, evaluate_answers
from course_qa.excerpts import verified_source_excerpts
from course_qa.generation import (
    GenerationError,
    GenerationResult,
    InvalidGenerationError,
)
from course_qa.qa import AnswerResult, answer_question


def passages() -> list[RetrievedPassage]:
    texts = [
        "OpenCV stores color images in BGR order.\n",
        "Precision: TP / (TP + FP).\n",
    ]
    return [
        RetrievedPassage(
            Chunk(
                f"chunk-{i}",
                text,
                {
                    "source_id": f"notes-{i}",
                    "title": "Notes",
                    "project": "course",
                    "document_type": "student_notes",
                    "source_path": f"notes-{i}.md",
                    "source_hash": "source-snapshot",
                    "content_hash": hashlib.sha256(text.encode()).hexdigest(),
                    "processing_version": "test",
                    "line_start": 1,
                    "line_end": 1,
                    "chunk_index": 0,
                },
            ),
            0.1,
            i,
        )
        for i, text in enumerate(texts, 1)
    ]


def context(config: AppConfig) -> ContextBundle:
    return build_context("snapshot", passages(), config, len, 20_000, 0)


class StubEmbedder:
    fingerprint = "test-embedding-space"
    dimension = 2

    def count_tokens(self, text: str) -> int:
        return len(text)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise AssertionError("Excerpt questions must not ingest documents.")

    def embed_query(self, text: str) -> list[float]:
        raise AssertionError("Retrieval is mocked at the service boundary.")


class StubGenerator:
    model_id = "test-local-model"
    context_window_tokens = 20_000

    def __init__(self) -> None:
        # Deliberately contradict the source; this prose must never be displayed
        # by verified-excerpt mode. Source selection/relevance is a separate gate.
        self.response = GenerationResult(
            "answered", "The channels are RGB [S1].", ("S1",)
        )
        self.calls = 0
        self.error: GenerationError | None = None

    def count_tokens(self, text: str) -> int:
        return len(text)

    def generate(
        self, *, system_prompt: str, user_prompt: str, max_output_tokens: int
    ) -> GenerationResult:
        self.calls += 1
        if self.error:
            raise self.error
        return self.response


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    return AppConfig(storage_dir=tmp_path, max_output_tokens=40)


def run_question(config: AppConfig, model: StubGenerator) -> AnswerResult:
    with patch("course_qa.qa.retrieve_passages", return_value=("snapshot", passages())):
        return answer_question(
            "How are frames stored?",
            config,
            StubEmbedder(),
            model,
            count_request_tokens=lambda system, user: len(system) + len(user) + 16,
            answer_mode="verified_excerpts",
        )


def test_excerpts_preserve_exact_passage_text_and_reference(config: AppConfig) -> None:
    bundle = context(config)
    excerpts = verified_source_excerpts(bundle)
    assert len(excerpts) == 2
    for excerpt in excerpts:
        source = bundle.sources[excerpt.reference.label]
        assert excerpt.text == source.text
        assert excerpt.reference.chunk_id == source.chunk_id
        assert excerpt.reference.index_id == bundle.index_id
        assert excerpt.reference.source_hash == source.metadata["source_hash"]


def test_modified_text_is_rejected_even_with_a_valid_label(config: AppConfig) -> None:
    bundle = context(config)
    forged = replace(bundle.sources["S1"], text="OpenCV stores RGB.")
    with pytest.raises(InvalidGenerationError, match="content hash"):
        verified_source_excerpts(
            replace(bundle, sources={**bundle.sources, "S1": forged})
        )


def test_empty_evidence_is_rejected() -> None:
    with pytest.raises(InvalidGenerationError):
        verified_source_excerpts(ContextBundle("snapshot", "", {}))


def test_changed_payload_cannot_be_verified_using_a_different_chunk(
    config: AppConfig,
) -> None:
    bundle = context(config)
    text = "Altered source text."
    original = bundle.sources["S1"]
    forged = replace(
        original,
        text=text,
        metadata={
            **original.metadata,
            "content_hash": hashlib.sha256(text.encode()).hexdigest(),
        },
    )
    with pytest.raises(InvalidGenerationError, match="differs from supplied evidence"):
        verified_source_excerpts(
            replace(bundle, sources={**bundle.sources, "S1": forged})
        )


def test_invalid_source_label_is_rejected(config: AppConfig) -> None:
    bundle = context(config)
    with pytest.raises(InvalidGenerationError, match="label"):
        verified_source_excerpts(replace(bundle, sources={"S0": bundle.sources["S1"]}))


def test_excerpt_mode_discards_wrong_generated_prose_and_keeps_full_context(
    config: AppConfig,
) -> None:
    model = StubGenerator()
    result = run_question(config, model)
    assert (
        result.status == "evidence_selected"
        and result.answer_mode == "verified_excerpts"
    )
    assert result.generation_called and model.calls == 1
    assert len(result.excerpts) == 2  # Not limited to the model's S1 citation.
    assert "RGB" not in result.answer
    assert [excerpt.text for excerpt in result.excerpts] == [
        p.chunk.text for p in passages()
    ]
    assert result.sources == tuple(excerpt.reference for excerpt in result.excerpts)


def test_model_abstention_still_returns_no_excerpts(config: AppConfig) -> None:
    model = StubGenerator()
    model.response = GenerationResult(
        "insufficient_evidence", "The notes do not establish this.", ()
    )
    result = run_question(config, model)
    assert result.status == "insufficient_evidence"
    assert result.answer != model.response.answer
    assert (
        result.answer_mode == "verified_excerpts"
        and not result.excerpts
        and not result.sources
    )


def test_generation_failure_is_not_repaired_with_excerpts(config: AppConfig) -> None:
    model = StubGenerator()
    model.error = GenerationError("Incomplete output")
    with pytest.raises(GenerationError, match="Incomplete"):
        run_question(config, model)
    assert model.calls == 1


def test_invalid_citations_still_reject_the_model_response(config: AppConfig) -> None:
    model = StubGenerator()
    model.response = GenerationResult("answered", "Claim [S999].", ("S999",))
    with pytest.raises(InvalidGenerationError):
        run_question(config, model)


def test_generated_mode_remains_separate(config: AppConfig) -> None:
    model = StubGenerator()
    with patch("course_qa.qa.retrieve_passages", return_value=("snapshot", passages())):
        result = answer_question(
            "Question",
            config,
            StubEmbedder(),
            model,
            count_request_tokens=lambda system, user: len(system) + len(user) + 16,
        )
    assert (
        result.status == "answered"
        and result.answer_mode == "generated"
        and not result.excerpts
    )


def test_evaluation_labels_evidence_selection_without_claiming_answer_accuracy(
    config: AppConfig,
) -> None:
    selected = run_question(config, StubGenerator())
    case = RetrievalCase(
        "test", "Question", "answered", (EvidenceRange("notes-1", 1, 1),)
    )
    with (
        patch("course_qa.evaluation.SnapshotStore") as store,
        patch("course_qa.evaluation.answer_question", return_value=selected),
    ):
        store.return_value.read_active_manifest.return_value = {"index_id": "snapshot"}
        report = evaluate_answers(
            [case],
            config,
            StubEmbedder(),
            StubGenerator(),
            count_request_tokens=lambda system, user: 1,
            answer_mode="verified_excerpts",
        )
    assert report["metric"] == "expected_evidence_selection_status_agreement"
    assert report["matching_status_cases"] == 1
    assert (
        report["human_review"] == "pending" and report["release_gate"] == "not_assessed"
    )
