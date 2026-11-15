"""Provision and smoke-test the local language model in one explicit step."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol, cast

import huggingface_hub

from course_qa.citations import validate_citations
from course_qa.contracts import Chunk, ChunkMetadata, ContextBundle
from course_qa.local_generation import MODEL_ID, LocalQwenGenerator
from course_qa.paths import project_root
from course_qa.prompts import build_prompt

MODEL_REVISION = "fb163acb387a011a0cd205b259aa1b253299a05d"
PROJECT_ROOT = project_root()


class _SnapshotHub(Protocol):
    def snapshot_download(
        self,
        *,
        repo_id: str,
        revision: str,
        local_dir: Path,
        allow_patterns: list[str],
        max_workers: int,
    ) -> str: ...


def smoke_test(model_dir: Path) -> None:
    """Use synthetic evidence to verify local inference, parsing and citations.

    This checks wiring and one supported claim, not general answer quality.
    It does not read the course notes or execute any evaluation cases.
    """
    generator = LocalQwenGenerator(model_dir, max_generation_seconds=120.0)
    metadata: ChunkMetadata = {
        "source_id": "setup-fixture",
        "title": "Synthetic setup fixture",
        "project": "setup",
        "document_type": "fixture",
        "source_path": "synthetic.txt",
        "source_hash": "synthetic-not-a-file",
        "content_hash": "synthetic-not-a-file",
        "line_start": 1,
        "line_end": 1,
        "chunk_index": 0,
        "processing_version": "setup-fixture-v1",
    }
    chunk = Chunk("setup-chunk", "The setup marker is BLUE.", metadata)
    context = ContextBundle(
        "setup-only",
        json.dumps(
            {
                "index_id": "setup-only",
                "evidence": [{"source_label": "S1", "text": chunk.text}],
            }
        ),
        {"S1": chunk},
    )
    prompt = build_prompt("What is the setup marker?", context)
    result = generator.generate(
        system_prompt=prompt.system_prompt,
        user_prompt=prompt.user_prompt,
        max_output_tokens=192,
    )
    # This diagnostic contains only the fixed synthetic fixture, never notes
    # or user questions. Keep citation validation strict; do not repair output.
    print("Synthetic smoke-test response (not yet validated):")
    print(
        json.dumps(
            {
                "status": result.status,
                "answer": result.answer,
                "cited_labels": result.cited_labels,
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    validate_citations(result, context)
    if result.status != "answered" or "blue" not in result.answer.casefold():
        raise RuntimeError("Generation smoke test did not return the supported marker.")
    print("Local inference, JSON parsing and citation smoke check passed.")


def setup_generation_model(destination: Path) -> None:
    """Save a completed download before testing model behavior.

    A saved directory means weights are available, not that answers are verified.
    Smoke-test failures retain the model so --check-only can diagnose it offline.
    Incomplete download stages are still cleaned up; existing models are never
    overwritten.
    """
    destination = destination.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"Destination already exists: {destination}. "
            "Use --check-only to validate it without downloading or replacing it."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_ID} at revision {MODEL_REVISION}")
    with TemporaryDirectory(
        prefix=".generation-setup-", dir=destination.parent
    ) as temporary:
        staged = Path(temporary) / "model"
        cast(_SnapshotHub, huggingface_hub).snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=staged,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.jinja",
                "merges.txt",
                "vocab.json",
                "tokenizer.model",
                "LICENSE",
                "README.md",
            ],
            max_workers=2,
        )
        (staged / "provisioning.json").write_text(
            json.dumps(
                {
                    "model_id": MODEL_ID,
                    "revision": MODEL_REVISION,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            raise FileExistsError(
                "Destination appeared during setup; refusing to replace it."
            )
        staged.rename(destination)
    print(f"Saved generation model: {destination}")
    check_saved_model(destination)


def check_saved_model(model_dir: Path) -> None:
    """Record the current check outcome without deleting or replacing weights."""
    record = {
        "model_id": MODEL_ID,
        "tested_with_adapter": "LocalQwenGenerator",
        "smoke_test_status": "not_passed",
    }
    report_path = model_dir / "setup_check.json"
    # Clear any old success before loading; interruption must not leave a stale
    # successful check for this attempt. Do not rewrite model provenance.
    report_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    try:
        smoke_test(model_dir)
    except (OSError, ValueError, TypeError, RuntimeError):
        print(f"Weights retained at: {model_dir}")
        print("Smoke test has NOT passed. Retry with --check-only after diagnosis.")
        raise
    record["smoke_test_status"] = "passed"
    report_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "models/generation"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Test existing local weights without downloading.",
    )
    args = parser.parse_args()
    try:
        if args.check_only:
            check_saved_model(args.output.expanduser().resolve(strict=True))
        else:
            setup_generation_model(args.output)
    except KeyboardInterrupt:
        print("\nGeneration model setup interrupted.")
        return 130
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        print(f"Generation model setup failed: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
