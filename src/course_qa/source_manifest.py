"""Read an explicit source allowlist without embedding or changing the index."""

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import yaml

from course_qa.config import SourceConfig


def _object(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{location} must be a mapping.")
    mapping = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in mapping):
        raise ValueError(f"{location} keys must be strings.")
    return cast(Mapping[str, object], mapping)


def _text(data: Mapping[str, object], key: str, location: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}.{key} must be nonempty text.")
    return value.strip()


def _excluded_lines(value: object, location: str) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list):
        raise TypeError(f"{location} must be a list of [start, end] pairs.")

    ranges: list[tuple[int, int]] = []
    for pair in cast(list[object], value):
        if not isinstance(pair, list):
            raise TypeError(f"{location} entries must be [start, end] pairs.")
        bounds = cast(list[object], pair)
        if len(bounds) != 2:
            raise ValueError(f"{location} entries must contain two integers.")
        start, end = bounds
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
        ):
            raise TypeError(f"{location} entries must contain two integers.")
        ranges.append((start, end))

    return tuple(ranges)


def load_source_manifest(manifest_path: Path) -> list[SourceConfig]:
    """Load schema version 1; resolve paths relative to the manifest file.

    The manifest is trusted application configuration. Its approved roots
    explicitly select where notes may be read; notes cannot change those roots.
    SourceConfig checks containment, file types and exclusion range ordering.
    Ingestion separately enforces corpus and file-size limits.
    """
    manifest_path = manifest_path.resolve(strict=True)
    try:
        raw: object = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError("Source manifest contains invalid YAML.") from error

    document = _object(raw, "manifest")
    if set(document) != {"schema_version", "sources"}:
        raise ValueError("Manifest must contain only schema_version and sources.")

    schema_version = document["schema_version"]
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError("Only source manifest schema_version 1 is supported.")

    entries = document["sources"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest sources must be a nonempty list.")

    required = {
        "source_id",
        "title",
        "project",
        "document_type",
        "path",
        "approved_root",
    }
    sources: list[SourceConfig] = []
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()

    for index, entry in enumerate(cast(list[object], entries), start=1):
        location = f"sources[{index}]"
        data = _object(entry, location)
        if not required.issubset(data) or set(data) - required - {
            "excluded_lines",
            "excluded_pages",
        }:
            raise ValueError(f"{location} has missing or unrecognized fields.")
        pages = data.get("excluded_pages", [])
        if not isinstance(pages, list):
            raise TypeError(f"{location}.excluded_pages must be a list.")
        page_numbers: list[int] = []
        for page in cast(list[object], pages):
            if type(page) is not int:
                raise ValueError("Excluded page numbers must be integers.")
            page_numbers.append(page)

        source = SourceConfig(
            source_id=_text(data, "source_id", location),
            title=_text(data, "title", location),
            project=_text(data, "project", location),
            document_type=_text(data, "document_type", location),
            path=manifest_path.parent / _text(data, "path", location),
            approved_root=manifest_path.parent / _text(data, "approved_root", location),
            excluded_lines=_excluded_lines(
                data.get("excluded_lines", []), f"{location}.excluded_lines"
            ),
            excluded_pages=tuple(page_numbers),
        )
        resolved_path = source.resolve_path()
        if source.source_id in seen_ids:
            raise ValueError(f"Duplicate source_id: {source.source_id}")
        if resolved_path in seen_paths:
            raise ValueError("The manifest selects the same source file twice.")

        seen_ids.add(source.source_id)
        seen_paths.add(resolved_path)
        sources.append(source)

    return sources
