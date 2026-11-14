"""Load explicit local settings for ingestion and retrieval."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from course_qa.config import AppConfig


@dataclass(frozen=True)
class RuntimeSettings:
    app: AppConfig
    model_dir: Path
    source_manifest: Path
    document_prefix: str = ""
    query_prefix: str = ""


def load_settings(settings_path: Path) -> RuntimeSettings:
    """Resolve all configured paths relative to the settings file.

    Loading settings never creates storage, loads model weights or downloads a
    model. The model directory must already have been provisioned explicitly.
    """
    settings_path = settings_path.resolve(strict=True)
    try:
        raw: object = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError("Settings file contains invalid YAML.") from error

    if not isinstance(raw, dict):
        raise TypeError("Settings must be a mapping.")
    values = cast(dict[object, object], raw)
    if any(not isinstance(key, str) for key in values):
        raise ValueError("Settings keys must be strings.")
    data = cast(Mapping[str, object], values)

    required = {"schema_version", "model_dir", "storage_dir", "source_manifest"}
    optional = {"document_prefix", "query_prefix"}
    if not required.issubset(data) or set(data) - required - optional:
        raise ValueError("Settings have missing or unrecognized fields.")
    schema_version = data["schema_version"]
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError("Only settings schema_version 1 is supported.")

    def configured_path(key: str) -> Path:
        value = data[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a nonempty path.")
        return (settings_path.parent / Path(value).expanduser()).resolve()

    def prefix(key: str) -> str:
        value = data.get(key, "")
        if not isinstance(value, str):
            raise TypeError(f"{key} must be text.")
        # Preserve spaces: some models require prefixes such as 'query: '.
        return value

    model_dir = configured_path("model_dir")
    source_manifest = configured_path("source_manifest")
    storage_dir = configured_path("storage_dir")
    if not model_dir.is_dir():
        raise ValueError("model_dir must point to an existing local model directory.")
    if not source_manifest.is_file():
        raise ValueError("source_manifest must point to an existing file.")
    if storage_dir.exists() and not storage_dir.is_dir():
        raise ValueError("storage_dir must be a directory path.")
    if storage_dir.is_relative_to(model_dir) or model_dir.is_relative_to(storage_dir):
        raise ValueError("Storage and model directories must not contain each other.")

    return RuntimeSettings(
        app=AppConfig(storage_dir=storage_dir),
        model_dir=model_dir,
        source_manifest=source_manifest,
        document_prefix=prefix("document_prefix"),
        query_prefix=prefix("query_prefix"),
    )
