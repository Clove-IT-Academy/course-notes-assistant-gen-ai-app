"""Settings-to-ingestion strategy regressions using temporary notes and paths."""

from pathlib import Path
from typing import cast

import pytest
import yaml

from course_qa.config import SourceConfig
from course_qa.ingestion import prepare_chunks
from course_qa.settings import load_settings


@pytest.fixture
def settings_path(tmp_path: Path) -> Path:
    (tmp_path / "models").mkdir()
    (tmp_path / "sources.yaml").write_text(
        "schema_version: 1\nsources: []\n", encoding="utf-8"
    )
    path = tmp_path / "app.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "model_dir": "models",
                "storage_dir": "storage",
                "source_manifest": "sources.yaml",
            }
        ),
        encoding="utf-8",
    )
    return path


def set_strategy(path: Path, strategy: object) -> None:
    values = cast(dict[str, object], yaml.safe_load(path.read_text(encoding="utf-8")))
    values["chunking_strategy"] = strategy
    path.write_text(yaml.safe_dump(values), encoding="utf-8")


def test_existing_settings_without_strategy_keep_boundary_default(
    settings_path: Path,
) -> None:
    settings = load_settings(settings_path)
    assert settings.app.chunking_strategy == "boundary"
    assert not settings.app.storage_dir.exists()


@pytest.mark.parametrize("strategy", ["boundary", "markdown_blocks"])
def test_explicit_strategy_reaches_app_config(
    settings_path: Path, strategy: str
) -> None:
    set_strategy(settings_path, strategy)
    settings = load_settings(settings_path)
    assert settings.app.chunking_strategy == strategy
    assert not settings.app.storage_dir.exists()


@pytest.mark.parametrize(
    "strategy", ["", "markdown", "BOUNDARY", None, True, 42, [], {}]
)
def test_invalid_strategy_is_rejected(settings_path: Path, strategy: object) -> None:
    set_strategy(settings_path, strategy)
    with pytest.raises(ValueError, match="chunking_strategy"):
        load_settings(settings_path)
    assert not (settings_path.parent / "storage").exists()


def test_yaml_choice_controls_chunking_and_processing_identity(
    settings_path: Path,
) -> None:
    path = settings_path.parent / "notes.md"
    path.write_text("# First\n\nAlpha.\n\n## Second\n\nBeta.", encoding="utf-8")
    source = SourceConfig(
        "notes", "Notes", "course", path, path.parent, "student_notes"
    )
    baseline = load_settings(settings_path)
    boundary_chunks = prepare_chunks([source], baseline.app)
    assert len(boundary_chunks) == 1

    set_strategy(settings_path, "markdown_blocks")
    experiment = load_settings(settings_path)
    structural_chunks = prepare_chunks([source], experiment.app)
    assert len(structural_chunks) == 2
    assert {c.chunk_id for c in boundary_chunks}.isdisjoint(
        c.chunk_id for c in structural_chunks
    )
    assert (
        boundary_chunks[0].metadata["processing_version"]
        != structural_chunks[0].metadata["processing_version"]
    )

    set_strategy(settings_path, "boundary")
    restored = load_settings(settings_path)
    assert prepare_chunks([source], restored.app) == boundary_chunks
    assert not restored.app.storage_dir.exists()
