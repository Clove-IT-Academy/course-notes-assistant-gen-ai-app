"""Installed-release command routing, error categories and classroom defaults."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from chromadb.errors import InternalError

import app
from course_qa.errors import IndexStateError, SourceReadError
from course_qa.generation import GenerationError
from course_qa.qa import AnswerResult


def settings_file(tmp_path: Path) -> Path:
    (tmp_path / "models").mkdir()
    (tmp_path / "generation").mkdir()
    (tmp_path / "storage").mkdir()
    (tmp_path / "storage" / "active.json").write_text("{}", encoding="utf-8")
    (tmp_path / "sources.yaml").write_text(
        "schema_version: 1\nsources: []", encoding="utf-8"
    )
    path = tmp_path / "app.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "model_dir": "models",
                "storage_dir": "storage",
                "source_manifest": "sources.yaml",
                "answer_mode": "verified_excerpts",
                "generation": {"model_dir": "generation"},
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("mode", [None, "generated"])
def test_configured_excerpt_default_and_explicit_generated_override(
    tmp_path: Path, mode: str | None
) -> None:
    path = settings_file(tmp_path)
    result = AnswerResult(
        "insufficient_evidence", "No evidence.", (), "snapshot", "prompt", "model", True
    )
    with (
        patch("app.LocalSentenceEmbedder"),
        patch("course_qa.local_generation.LocalQwenGenerator"),
        patch("course_qa.qa.answer_question", return_value=result) as question,
    ):
        argv = ["ask", "Question", "--settings", str(path)]
        if mode:
            argv += ["--answer-mode", mode]
        assert app.main(argv) == 0
        assert question.call_args.kwargs["answer_mode"] == (mode or "verified_excerpts")


@pytest.mark.parametrize(
    "error,code",
    [
        (IndexStateError("bad index"), 3),
        (SourceReadError("bad source"), 4),
        (GenerationError("bad generation"), 5),
        (RuntimeError("database locked"), 4),
        (KeyboardInterrupt(), 130),
        (InternalError("private third-party payload"), 4),
    ],
)
def test_typed_operational_exits(
    tmp_path: Path, error: BaseException, code: int, capsys: pytest.CaptureFixture[str]
) -> None:
    path = settings_file(tmp_path)
    with (
        patch("app.LocalSentenceEmbedder"),
        patch("course_qa.local_generation.LocalQwenGenerator"),
        patch("course_qa.qa.answer_question", side_effect=error),
    ):
        assert app.main(["ask", "Question", "--settings", str(path)]) == code
    output = capsys.readouterr()
    assert not output.out and "private third-party payload" not in output.err


def test_maintenance_commands_need_no_models_or_settings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with (
        patch("app.SnapshotStore") as store,
        patch(
            "app.load_settings", side_effect=AssertionError("must not load settings")
        ),
        patch(
            "app.LocalSentenceEmbedder",
            side_effect=AssertionError("must not load weights"),
        ),
    ):
        store.return_value.list_snapshots.return_value = []
        assert app.main(["snapshots", "--storage", str(tmp_path)]) == 0
        assert json.loads(capsys.readouterr().out) == []
        store.return_value.cleanup.return_value = ["candidate"]
        assert app.main(["cleanup", "--storage", str(tmp_path)]) == 0
        assert not json.loads(capsys.readouterr().out)["applied"]
        store.return_value.cleanup.assert_called_with(keep=2, apply=False)
        store.return_value.rollback.return_value = {"index_id": "a" * 32}
        assert app.main(["rollback", "a" * 32, "--storage", str(tmp_path)]) == 0
        store.return_value.rollback.assert_called_with("a" * 32)
