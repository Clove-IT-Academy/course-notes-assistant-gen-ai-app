"""Regression coverage for type/API repairs; never downloads a model."""

import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from course_qa.config import AppConfig, SourceConfig
from course_qa.context import build_context
from course_qa.contracts import parse_chunk_metadata
from course_qa.embeddings import LocalSentenceEmbedder, validate_vectors
from course_qa.ingestion import ingest, prepare_chunks
from course_qa.retrieval import retrieve_passages
from course_qa.store import SnapShotStore, SnapshotStore


class FakeEmbedder:
    fingerprint = "fixture-embedding-space-v1"
    dimension = 2

    def __init__(self) -> None:
        self.document_calls = 0
        self.fail = False

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("Simulated embedding failure")
        self.document_calls += 1
        return [
            [1.0, 0.0] if "motion" in text.lower() else [0.0, 1.0] for text in texts
        ]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0] if "motion" in text.lower() else [0.0, 1.0]


@pytest.fixture
def sources(tmp_path: Path) -> list[SourceConfig]:
    result = []
    for name, text in [
        ("motion", "Motion detection compares frames.\nBlur smooths image noise.\n"),
        (
            "forecast",
            "Forecasting predicts future values from historical observations.\n",
        ),
    ]:
        path = tmp_path / f"{name}.md"
        path.write_text(text, encoding="utf-8")
        result.append(
            SourceConfig(name, name.title(), name, path, tmp_path, "student_notes")
        )
    return result


def test_preview_is_traceable_and_creates_no_store(
    tmp_path: Path, sources: list[SourceConfig]
) -> None:
    config = AppConfig(storage_dir=tmp_path / "storage")
    before = sources[0].path.read_bytes()
    preview = ingest(sources, config, dry_run=True)
    assert preview["status"] == "preview"
    assert not config.storage_dir.exists()
    chunks = prepare_chunks(sources, config)
    assert chunks[0].metadata["line_start"] == 1
    assert chunks[0].metadata["line_end"] == 2
    assert sources[0].path.read_bytes() == before
    assert chunks == prepare_chunks(sources, config)


def test_real_chroma_round_trip_and_idempotence(
    tmp_path: Path, sources: list[SourceConfig]
) -> None:
    config = AppConfig(storage_dir=tmp_path / "storage")
    embedder = FakeEmbedder()
    created = ingest(sources, config, embedder)
    unchanged = ingest(sources, config, embedder)
    assert created["status"] == "published"
    assert unchanged["status"] == "unchanged"
    assert created["index_id"] == unchanged["index_id"]
    assert embedder.document_calls == 1

    index_id, passages = retrieve_passages("motion", config, embedder, project="motion")
    assert len(passages) == 1
    assert passages[0].chunk.metadata["source_id"] == "motion"
    bundle = build_context(index_id, passages, config, len, 10000, 100)
    assert bundle.sources["S1"] == passages[0].chunk
    assert json.loads(bundle.text)["evidence"][0]["source_label"] == "S1"
    assert not (config.storage_dir / ".lock").exists()

    # Persistence must work from a separate interpreter, not just a cached client.
    code = """
from pathlib import Path
from collections.abc import Sequence
from course_qa.store import SnapshotStore
import sys
store = SnapshotStore(Path(sys.argv[1]))
with store.locked():
    manifest, collection = store.open_active('fixture-embedding-space-v1', 2)
    assert collection.count() == 2
    print(manifest['index_id'])
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    completed = subprocess.run(
        [sys.executable, "-c", code, str(config.storage_dir)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert index_id in completed.stdout


def test_failed_update_retains_active_snapshot(
    tmp_path: Path, sources: list[SourceConfig]
) -> None:
    config = AppConfig(storage_dir=tmp_path / "storage")
    embedder = FakeEmbedder()
    ingest(sources, config, embedder)
    pointer = (config.storage_dir / "active.json").read_bytes()
    sources[0].path.write_text("Motion content has changed.", encoding="utf-8")
    embedder.fail = True
    with pytest.raises(RuntimeError, match="Simulated"):
        ingest(sources, config, embedder)
    assert (config.storage_dir / "active.json").read_bytes() == pointer
    assert not (config.storage_dir / ".lock").exists()


def test_removed_source_is_absent_from_new_snapshot(
    tmp_path: Path, sources: list[SourceConfig]
) -> None:
    config = AppConfig(storage_dir=tmp_path / "storage")
    embedder = FakeEmbedder()
    old = ingest(sources, config, embedder)
    new = ingest(sources[:1], config, embedder)
    assert new["index_id"] != old["index_id"]
    _, results = retrieve_passages("forecast", config, embedder, project="forecast")
    assert results == []


def test_embedding_mismatch_is_rejected(
    tmp_path: Path, sources: list[SourceConfig]
) -> None:
    config = AppConfig(storage_dir=tmp_path / "storage")
    ingest(sources, config, FakeEmbedder())
    store = SnapshotStore(config.storage_dir)
    with store.locked(), pytest.raises(ValueError, match="configuration changed"):
        store.open_active("different-model-same-dimension", 2)


def test_missing_index_is_not_created(tmp_path: Path) -> None:
    config = AppConfig(storage_dir=tmp_path / "storage")
    with pytest.raises(FileNotFoundError, match="Ingest notes first"):
        retrieve_passages("motion", config, FakeEmbedder())
    assert not (config.storage_dir / "indexes").exists()


def test_malformed_manifest_is_rejected(
    tmp_path: Path, sources: list[SourceConfig]
) -> None:
    config = AppConfig(storage_dir=tmp_path / "storage")
    result = ingest(sources, config, FakeEmbedder())
    path = config.storage_dir / "indexes" / str(result["index_id"]) / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["dimension"] = True  # bool must not be accepted as an integer dimension
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="dimension"):
        SnapshotStore(config.storage_dir).read_active_manifest()


def test_metadata_is_checked_at_database_boundary() -> None:
    with pytest.raises(ValueError, match="source_id"):
        parse_chunk_metadata({"line_start": 1})


def test_historical_class_spelling_is_compatible() -> None:
    assert SnapShotStore is SnapshotStore


@pytest.mark.parametrize("dimension, limit", [(None, 8), (0, 8), (2, None), (2, 0)])
def test_optional_model_values_are_rejected(
    tmp_path: Path, dimension: int | None, limit: int | None
) -> None:
    model = Mock()
    model.get_embedding_dimension.return_value = dimension
    model.max_seq_length = limit
    with (
        patch("sentence_transformers.SentenceTransformer", return_value=model),
        pytest.raises(ValueError, match="positive"),
    ):
        LocalSentenceEmbedder(tmp_path)


def test_text_encoder_uses_correct_arguments_and_special_tokens(tmp_path: Path) -> None:
    (tmp_path / "model.txt").write_text("synthetic model fixture")
    model = Mock()
    model.get_embedding_dimension.return_value = 2
    model.max_seq_length = 8
    model.tokenizer.return_value = {"input_ids": [1, 2, 3]}
    model.encode.return_value = np.array([[1.0, 0.0]], dtype=np.float32)
    with patch(
        "sentence_transformers.SentenceTransformer", return_value=model
    ) as load_model:
        embedder = LocalSentenceEmbedder(tmp_path)
        assert embedder.embed_query("motion") == [1.0, 0.0]
        assert embedder.embed_documents([]) == []
        assert model.tokenizer.call_args.kwargs["add_special_tokens"] is True
        assert load_model.call_args.kwargs["local_files_only"] is True
        assert load_model.call_args.kwargs["trust_remote_code"] is False
        model.tokenizer.return_value = {"input_ids": list(range(9))}
        with pytest.raises(ValueError, match="maximum token limit"):
            embedder.embed_query("too many tokens")


@pytest.mark.parametrize("vectors", [[[0.0, 0.0]], [[float("nan"), 1.0]], [[1.0]]])
def test_invalid_vectors_are_rejected(vectors: list[list[float]]) -> None:
    with pytest.raises(ValueError):
        validate_vectors(vectors, expected_count=1, dimension=2)
