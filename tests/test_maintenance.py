"""Atomic rollback and cleanup exercise real isolated Chroma snapshots."""

import json
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

import pytest

from course_qa.config import AppConfig, SourceConfig
from course_qa.errors import IndexStateError
from course_qa.ingestion import ingest
from course_qa.store import SnapshotStore


class TinyEmbedder:
    fingerprint = "maintenance-test"
    dimension = 2

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def snapshots(tmp_path: Path) -> tuple[SnapshotStore, list[str]]:
    path = tmp_path / "notes.txt"
    source = SourceConfig("notes", "Notes", "test", path, tmp_path, "student_notes")
    config = AppConfig(storage_dir=tmp_path / "store")
    ids: list[str] = []
    for number in range(3):
        path.write_text(f"The approved note version is {number}.", encoding="utf-8")
        result = ingest([source], config, TinyEmbedder())
        assert result["status"] != "preview"
        ids.append(result["index_id"])
    return SnapshotStore(config.storage_dir), ids


def test_rollback_reopens_target_and_cleanup_never_removes_active(
    tmp_path: Path,
) -> None:
    store, ids = snapshots(tmp_path)
    with store.locked():
        assert len(store.list_snapshots()) == 3
        store.rollback(ids[0])
        current, collection = store.open_active("maintenance-test", 2)
        assert current["index_id"] == ids[0] and collection.count() == 1
        planned = store.cleanup(keep=0)
        assert set(planned) == set(ids[1:])
        assert all((store.root / "indexes" / id_).exists() for id_ in ids)
        assert set(store.cleanup(keep=0, apply=True)) == set(planned)
        current, collection = store.open_active("maintenance-test", 2)
        assert current["index_id"] == ids[0] and collection.count() == 1
    assert (tmp_path / "notes.txt").read_text().endswith("2.")
    assert not (store.root / ".lock").exists()


def test_bad_target_and_failed_publication_leave_active_pointer_unchanged(
    tmp_path: Path,
) -> None:
    store, ids = snapshots(tmp_path)
    pointer = store.root / "active.json"
    before = pointer.read_bytes()
    with store.locked():
        with pytest.raises(ValueError):
            store.rollback("../notes")
        with pytest.raises(IndexStateError):
            store.rollback("f" * 32)
        with (
            patch(
                "course_qa.store.os.replace",
                side_effect=OSError("Interrupted publication"),
            ),
            pytest.raises(OSError),
        ):
            store.rollback(ids[0])
        assert pointer.read_bytes() == before
    assert not list(store.root.glob("active-*.tmp"))


def test_corrupt_pointer_can_be_explicitly_restored(tmp_path: Path) -> None:
    store, ids = snapshots(tmp_path)
    (store.root / "active.json").write_text("bad json", encoding="utf-8")
    with store.locked():
        with pytest.raises(IndexStateError):
            store.cleanup(keep=0, apply=True)
        store.rollback(ids[0])
        assert (
            json.loads((store.root / "active.json").read_text())["index_id"] == ids[0]
        )


def test_maintenance_requires_lock_and_rejects_symlinks(tmp_path: Path) -> None:
    store, _ = snapshots(tmp_path)
    with pytest.raises(RuntimeError, match="lock"):
        store.list_snapshots()
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "keep.txt"
    marker.write_text("Do not delete", encoding="utf-8")
    (store.root / "indexes" / ("e" * 32)).symlink_to(outside, target_is_directory=True)
    with store.locked(), pytest.raises(ValueError, match="symlinks"):
        store.cleanup(keep=0, apply=True)
    assert marker.read_text() == "Do not delete"


def test_rollback_rejects_incompatible_embedding_space(tmp_path: Path) -> None:
    store, ids = snapshots(tmp_path)
    manifest_path = store.root / "indexes" / ids[0] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["embedding_fingerprint"] = "different-space"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    before = (store.root / "active.json").read_bytes()
    with store.locked(), pytest.raises(IndexStateError, match="different embedding"):
        store.rollback(ids[0])
    assert (store.root / "active.json").read_bytes() == before


def test_cleanup_leaves_incomplete_and_unknown_directories(tmp_path: Path) -> None:
    store, ids = snapshots(tmp_path)
    incomplete = store.root / "indexes" / ("d" * 32)
    incomplete.mkdir()
    marker = incomplete / "keep.txt"
    marker.write_text("unfinished build", encoding="utf-8")
    with store.locked():
        deleted = store.cleanup(keep=0, apply=True)
        assert set(deleted) == set(ids[:-1])
    assert marker.read_text() == "unfinished build"
