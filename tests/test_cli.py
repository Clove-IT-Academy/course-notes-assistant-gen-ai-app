"""Terminal workflow regressions with temporary storage and mocked services.

No model weights, real evaluation cases or course index are used. These checks
cover CLI behavior; answer correctness requires separate model evaluation.
"""

import json
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

import app
from course_qa.citations import SourceReference
from course_qa.config import AppConfig
from course_qa.excerpts import VerifiedExcerpt
from course_qa.generation import GenerationError
from course_qa.qa import AnswerResult
from course_qa.settings import GenerationSettings, RuntimeSettings


@dataclass
class CliHarness:
    settings: RuntimeSettings
    load_settings: Mock
    embedder: Mock
    generator: Mock
    answer: Mock
    cases: Mock
    retrieval_evaluation: Mock
    answer_evaluation: Mock


@pytest.fixture
def cli(tmp_path: Path) -> Iterator[CliHarness]:
    storage = tmp_path / "storage"
    storage.mkdir()
    # Only the existence check is real; retrieval/storage services are mocked.
    (storage / "active.json").write_text("{}", encoding="utf-8")
    settings = RuntimeSettings(
        app=AppConfig(storage_dir=storage, max_output_tokens=40),
        model_dir=tmp_path / "embedding-model",
        source_manifest=tmp_path / "sources.yaml",
        generation=GenerationSettings(model_dir=tmp_path / "generation-model"),
    )
    result = AnswerResult(
        "answered",
        "The sample switch is green [S1].",
        (
            SourceReference(
                "S1", "chunk", "snapshot", "course", "Notes", "notes.md", 1, 2, "hash"
            ),
        ),
        "snapshot",
        "test-prompt",
        "test-model",
        True,
    )
    with ExitStack() as stack:
        loader = stack.enter_context(patch("app.load_settings", return_value=settings))
        embedder = stack.enter_context(patch("app.LocalSentenceEmbedder"))
        embedder.return_value.count_tokens.side_effect = len
        embedder.return_value.max_tokens = 256
        generator = stack.enter_context(
            patch("course_qa.local_generation.LocalQwenGenerator")
        )
        generator.return_value.count_tokens.side_effect = len
        answer = stack.enter_context(
            patch("course_qa.qa.answer_question", return_value=result)
        )
        cases = stack.enter_context(
            patch("course_qa.evaluation.load_cases", return_value=[])
        )
        retrieval_evaluation = stack.enter_context(
            patch(
                "course_qa.evaluation.evaluate_retrieval",
                return_value={"coverage_rate": 0.5},
            )
        )
        answer_evaluation = stack.enter_context(
            patch(
                "course_qa.evaluation.evaluate_answers",
                return_value={"human_review": "pending", "generation_errors": 1},
            )
        )
        # Queries and evaluation must never silently rebuild the index.
        stack.enter_context(
            patch("app.ingest", side_effect=AssertionError("Unexpected ingestion"))
        )
        yield CliHarness(
            settings,
            loader,
            embedder,
            generator,
            answer,
            cases,
            retrieval_evaluation,
            answer_evaluation,
        )


def test_ask_displays_sources_and_forwards_project(
    cli: CliHarness, capsys: pytest.CaptureFixture[str]
) -> None:
    assert app.main(["ask", "What color?", "--project", "course"]) == 0
    output = capsys.readouterr()
    assert "Status: answered" in output.out and "notes.md" in output.out
    assert "lines 1-2" in output.out
    assert cli.answer.call_args.args[0] == "What color?"
    assert cli.answer.call_args.kwargs["project"] == "course"
    assert (
        cli.answer.call_args.kwargs["count_request_tokens"]
        == cli.generator.return_value.count_request_tokens
    )
    cli.embedder.assert_called_once()
    cli.generator.assert_called_once()


def test_terminal_controls_are_escaped(
    cli: CliHarness, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.answer.return_value = replace(
        cli.answer.return_value, answer="Answer\x1b[31m [S1]."
    )
    assert app.main(["ask", "Question"]) == 0
    output = capsys.readouterr()
    assert "\x1b" not in output.out and r"\x1b" in output.out


def test_ask_generation_error_displays_no_answer(
    cli: CliHarness, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.answer.side_effect = GenerationError("Incomplete\x1b[31m")
    assert app.main(["ask", "Question"]) == 5
    output = capsys.readouterr()
    assert not output.out and "Generation error:" in output.err
    assert "\x1b" not in output.err
    cli.answer.assert_called_once()


@pytest.mark.parametrize(
    "argv", [["ask", " "], ["ask", "Q", "--project", " "], ["chat", "--project", " "]]
)
def test_invalid_input_loads_no_models(cli: CliHarness, argv: list[str]) -> None:
    assert app.main(argv) == 2
    cli.embedder.assert_not_called()
    cli.generator.assert_not_called()


@pytest.mark.parametrize(
    "argv", [["ask", "Q"], ["chat"], ["evaluate", "--mode", "answers"]]
)
def test_missing_generation_config_loads_no_models(
    cli: CliHarness, argv: list[str]
) -> None:
    cli.load_settings.return_value = replace(cli.settings, generation=None)
    assert app.main(argv) == 2
    cli.embedder.assert_not_called()
    cli.generator.assert_not_called()


@pytest.mark.parametrize(
    "argv", [["ask", "Q"], ["chat"], ["retrieve", "Q"], ["evaluate"]]
)
def test_missing_index_loads_no_models(cli: CliHarness, argv: list[str]) -> None:
    (cli.settings.app.storage_dir / "active.json").unlink()
    assert app.main(argv) == 3
    cli.embedder.assert_not_called()
    cli.generator.assert_not_called()


def test_chat_reuses_models_and_keeps_questions_independent(cli: CliHarness) -> None:
    with patch("builtins.input", side_effect=[" ", "  First?  ", "课程问题", "/QuIt"]):
        assert app.main(["chat", "--project", "course"]) == 0
    cli.embedder.assert_called_once()
    cli.generator.assert_called_once()
    assert [call.args[0] for call in cli.answer.call_args_list] == [
        "First?",
        "课程问题",
    ]
    assert all(
        len(call.args) == 4 and call.kwargs["project"] == "course"
        for call in cli.answer.call_args_list
    )


def test_chat_recovers_after_generation_error(
    cli: CliHarness, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.answer.side_effect = [
        GenerationError("Invalid response"),
        cli.answer.return_value,
    ]
    with patch("builtins.input", side_effect=["First?", "Second?", "/exit"]):
        assert app.main(["chat"]) == 0
    output = capsys.readouterr()
    assert (
        "Generation error:" in output.err and output.out.count("Status: answered") == 1
    )
    assert cli.answer.call_count == 2


@pytest.mark.parametrize(
    "exception,code", [(EOFError(), 0), (KeyboardInterrupt(), 130)]
)
def test_chat_terminal_exits(
    cli: CliHarness, exception: BaseException, code: int
) -> None:
    with patch("builtins.input", side_effect=exception):
        assert app.main(["chat"]) == code
    cli.answer.assert_not_called()


def test_chat_interruption_during_answer(cli: CliHarness) -> None:
    cli.answer.side_effect = KeyboardInterrupt()
    with patch("builtins.input", return_value="Question"):
        assert app.main(["chat"]) == 130
    cli.answer.assert_called_once()


def test_chat_index_failure_exits_instead_of_looping(cli: CliHarness) -> None:
    cli.answer.side_effect = ValueError("Incompatible index")
    with patch("builtins.input", return_value="Question"):
        assert app.main(["chat"]) == 2
    cli.answer.assert_called_once()


@pytest.mark.parametrize("limit", ["configured", "embedding", "generation"])
def test_chat_rejects_oversized_question_and_accepts_quit(
    cli: CliHarness, limit: str
) -> None:
    question = "x" * (
        501 if limit == "configured" else 300 if limit == "embedding" else 1
    )
    if limit == "generation":
        cli.generator.return_value.count_tokens.side_effect = None
        cli.generator.return_value.count_tokens.return_value = 501
    with patch("builtins.input", side_effect=[question, "/quit"]):
        assert app.main(["chat"]) == 0
    cli.answer.assert_not_called()


def test_evaluation_defaults_to_development_retrieval(
    cli: CliHarness, capsys: pytest.CaptureFixture[str]
) -> None:
    assert app.main(["evaluate"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["split"] == "development" and report["mode"] == "retrieval"
    assert cli.cases.call_args.args[1] == "development"
    cli.retrieval_evaluation.assert_called_once()
    cli.generator.assert_not_called()
    cli.answer_evaluation.assert_not_called()


def test_answer_evaluation_exports_report_without_claiming_release_pass(
    cli: CliHarness, tmp_path: Path
) -> None:
    output = tmp_path / "report.json"
    assert app.main(["evaluate", "--mode", "answers", "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["human_review"] == "pending" and report["generation_errors"] == 1
    assert report["split"] == "development" and report["mode"] == "answers"
    assert (
        cli.answer_evaluation.call_args.kwargs["count_request_tokens"]
        == cli.generator.return_value.count_request_tokens
    )
    cli.generator.assert_called_once()


def test_held_out_selection_is_explicit_and_only_mocked(cli: CliHarness) -> None:
    assert app.main(["evaluate", "--split", "held_out"]) == 0
    assert cli.cases.call_args.args[1] == "held_out"


def test_evaluation_rejects_invalid_cases_before_models(cli: CliHarness) -> None:
    cli.cases.side_effect = ValueError("Invalid case fixture")
    assert app.main(["evaluate"]) == 2
    cli.embedder.assert_not_called()
    cli.generator.assert_not_called()


def test_evaluation_rejects_existing_report_before_models(
    cli: CliHarness, tmp_path: Path
) -> None:
    output = tmp_path / "report.json"
    output.write_text("previous report", encoding="utf-8")
    assert app.main(["evaluate", "--output", str(output)]) == 2
    assert output.read_text() == "previous report"
    cli.embedder.assert_not_called()


def test_report_creation_race_cannot_overwrite_another_file(
    cli: CliHarness, tmp_path: Path
) -> None:
    output = tmp_path / "report.json"

    def evaluate(*args: object) -> dict[str, object]:
        output.write_text("created by another process", encoding="utf-8")
        return {"coverage_rate": 1.0}

    cli.retrieval_evaluation.side_effect = evaluate
    assert app.main(["evaluate", "--output", str(output)]) == 2
    assert output.read_text() == "created by another process"


def test_missing_report_directory_loads_no_models(
    cli: CliHarness, tmp_path: Path
) -> None:
    output = tmp_path / "missing" / "report.json"
    assert app.main(["evaluate", "--output", str(output)]) == 2
    cli.embedder.assert_not_called()


def test_invalid_split_is_rejected_by_argument_parser(cli: CliHarness) -> None:
    with pytest.raises(SystemExit) as raised:
        app.main(["evaluate", "--split", "everything"])
    assert raised.value.code == 2
    cli.load_settings.assert_not_called()
    cli.embedder.assert_not_called()


def test_verified_excerpts_are_displayed_as_source_text(
    cli: CliHarness, capsys: pytest.CaptureFixture[str]
) -> None:
    source = cli.answer.return_value.sources[0]
    cli.answer.return_value = replace(
        cli.answer.return_value,
        status="evidence_selected",
        answer="Selected source excerpts; no generated explanation is shown.",
        answer_mode="verified_excerpts",
        excerpts=(VerifiedExcerpt("Source text\x1b[31m", source),),
    )
    assert app.main(["ask", "Question", "--answer-mode", "verified_excerpts"]) == 0
    output = capsys.readouterr().out
    assert "Status: evidence_selected" in output and "Source text" in output
    assert "\x1b" not in output and "The sample switch is green" not in output
    assert cli.answer.call_args.kwargs["answer_mode"] == "verified_excerpts"


def test_retrieval_evaluation_does_not_silently_ignore_excerpt_mode(
    cli: CliHarness,
) -> None:
    assert app.main(["evaluate", "--answer-mode", "verified_excerpts"]) == 2
    cli.embedder.assert_not_called()
