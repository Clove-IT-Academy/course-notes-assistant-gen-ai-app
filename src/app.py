"""Terminal entry point for the ingestion and retrieval teaching increment."""

import argparse
import json
import sys
from pathlib import Path

from course_qa.embeddings import LocalSentenceEmbedder
from course_qa.ingestion import ingest
from course_qa.retrieval import retrieve_passages
from course_qa.settings import load_settings
from course_qa.source_manifest import load_source_manifest
from course_qa.store import SnapshotStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def terminal_text(value: str) -> str:
    """Escape terminal controls while preserving readable multiline notes."""
    return "".join(
        character
        if character.isprintable() or character in "\n\t"
        else repr(character)[1:-1]
        for character in value
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect and index approved course notes."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    preview = commands.add_parser(
        "preview", help="Inspect chunks without loading a model."
    )
    preview.add_argument(
        "--manifest", type=Path, default=PROJECT_ROOT / "configs/sources.yaml"
    )

    for command in ("ingest", "retrieve"):
        subparser = commands.add_parser(command)
        subparser.add_argument(
            "--settings", type=Path, default=PROJECT_ROOT / "configs/app.yaml"
        )
        if command == "retrieve":
            subparser.add_argument("question")
            subparser.add_argument("--project")

    status = commands.add_parser(
        "status", help="Read active snapshot metadata without a model."
    )
    status.add_argument("--storage", type=Path, default=PROJECT_ROOT / "storage")
    args = parser.parse_args(argv)

    try:
        if args.command == "preview":
            # Preview uses preparation only: no weights, embeddings or database.
            from course_qa.config import AppConfig

            result = ingest(
                load_source_manifest(args.manifest),
                AppConfig(storage_dir=PROJECT_ROOT / "storage"),
                dry_run=True,
            )
            if result["status"] != "preview":
                raise RuntimeError("Expected an ingestion preview.")
            print(
                f"Sources: {result['source_count']} | Chunks: {result['chunk_count']}"
            )
            for chunk in result["chunks"][:3]:
                print(terminal_text(json.dumps(chunk.metadata, ensure_ascii=False)))
                print(terminal_text(chunk.text))
                print()
            return 0

        if args.command == "status":
            store = SnapshotStore(args.storage)
            if not (args.storage / "active.json").exists():
                print("No active index. Run ingest after configuring the local model.")
                return 3
            with store.locked():
                manifest = store.read_active_manifest()
            print(terminal_text(json.dumps(manifest, indent=2)))
            return 0

        settings = load_settings(args.settings)
        embedder = LocalSentenceEmbedder(
            settings.model_dir,
            batch_size=settings.app.embedding_batch_size,
            document_prefix=settings.document_prefix,
            query_prefix=settings.query_prefix,
        )
        if args.command == "ingest":
            result = ingest(
                load_source_manifest(settings.source_manifest), settings.app, embedder
            )
            print(terminal_text(json.dumps(result, indent=2)))
            return 0

        index_id, passages = retrieve_passages(
            args.question, settings.app, embedder, project=args.project
        )
        print(terminal_text(f"Snapshot: {index_id}"))
        if not passages:
            print("No passages matched the retrieval settings.")
        for passage in passages:
            metadata = passage.chunk.metadata
            print(
                terminal_text(
                    f"\nRank {passage.rank} | distance {passage.distance:.4f} | "
                    f"{metadata['project']} / {metadata['source_path']} / "
                    f"lines {metadata['line_start']}-{metadata['line_end']}"
                )
            )
            print(terminal_text(passage.chunk.text))
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        print(terminal_text(f"Error: {error}"), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
