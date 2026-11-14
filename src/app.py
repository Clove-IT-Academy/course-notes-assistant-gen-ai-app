"""Terminal entry point for indexing, cited answers and evaluation reports."""

import argparse
import json
import sys
from pathlib import Path

from course_qa.embeddings import LocalSentenceEmbedder
from course_qa.generation import GenerationError
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
        description="Index approved course notes and ask questions with citations."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    preview = commands.add_parser(
        "preview", help="Inspect chunks without loading a model."
    )
    preview.add_argument(
        "--manifest", type=Path, default=PROJECT_ROOT / "configs/sources.yaml"
    )

    for command in ("ingest", "retrieve", "ask", "chat", "evaluate"):
        subparser = commands.add_parser(command)
        subparser.add_argument(
            "--settings", type=Path, default=PROJECT_ROOT / "configs/app.yaml"
        )
        if command in ("retrieve", "ask"):
            subparser.add_argument("question")
        if command in ("retrieve", "ask", "chat"):
            subparser.add_argument("--project")
        if command == "evaluate":
            subparser.add_argument(
                "--cases", type=Path, default=PROJECT_ROOT / "eval/cases.jsonl"
            )
            subparser.add_argument(
                "--split", choices=("development", "held_out"), default="development"
            )
            subparser.add_argument(
                "--mode", choices=("retrieval", "answers"), default="retrieval"
            )
            subparser.add_argument(
                "--output", type=Path, help="Save a new JSON report; never overwrite."
            )

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
        if args.command == "evaluate":
            from course_qa.evaluation import (
                evaluate_answers,
                evaluate_retrieval,
                load_cases,
            )

            # Validate fixtures/output before loading either model. Held-out runs
            # require explicit --split selection; development remains the default.
            cases = load_cases(args.cases, args.split)
            if args.output is not None:
                if args.output.exists() or args.output.is_symlink():
                    raise ValueError("Output already exists; choose a new report path.")
                if not args.output.parent.is_dir():
                    raise ValueError("The report output directory must already exist.")
            if args.mode == "answers" and settings.generation is None:
                raise ValueError("Answer evaluation requires generation settings.")
            if not (settings.app.storage_dir / "active.json").is_file():
                print("No active index. Run ingest first.", file=sys.stderr)
                return 3
            evaluation_embedder = LocalSentenceEmbedder(
                settings.model_dir,
                batch_size=settings.app.embedding_batch_size,
                document_prefix=settings.document_prefix,
                query_prefix=settings.query_prefix,
            )
            if args.mode == "answers":
                from course_qa.local_generation import LocalQwenGenerator

                generation = settings.generation
                if generation is None:
                    raise ValueError("Answer evaluation requires generation settings.")
                print("Loading local Qwen generation model (CPU)...", file=sys.stderr)
                evaluation_generator = LocalQwenGenerator(
                    generation.model_dir,
                    context_window_tokens=generation.context_window_tokens,
                    max_generation_seconds=generation.max_generation_seconds,
                )
                report = evaluate_answers(
                    cases,
                    settings.app,
                    evaluation_embedder,
                    evaluation_generator,
                    count_request_tokens=evaluation_generator.count_request_tokens,
                )
            else:
                report = evaluate_retrieval(cases, settings.app, evaluation_embedder)
            report["split"] = args.split
            report["mode"] = args.mode
            rendered = json.dumps(report, indent=2, ensure_ascii=True)
            if args.output is None:
                print(rendered)
            else:
                # Exclusive creation also prevents overwriting after a path race.
                with args.output.open("x", encoding="utf-8") as output:
                    output.write(rendered + "\n")
                print(terminal_text(f"Report saved: {args.output.resolve()}"))
            # Completion is not a release-gate pass; answer review stays pending.
            return 0

        if args.command in ("retrieve", "ask", "chat"):
            if args.command != "chat" and not args.question.strip():
                raise ValueError("Enter a nonempty question.")
            if args.project is not None and not args.project.strip():
                raise ValueError("The project filter cannot be empty.")
            if args.command in ("ask", "chat") and settings.generation is None:
                raise ValueError(
                    f"The {args.command} command requires a generation section "
                    "in the settings file."
                )
            if not (settings.app.storage_dir / "active.json").is_file():
                print("No active index. Run ingest first.", file=sys.stderr)
                return 3

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

        if args.command in ("ask", "chat"):
            # Load once per command, then reuse both models for independent questions.
            from course_qa.local_generation import LocalQwenGenerator
            from course_qa.qa import answer_question

            generation = settings.generation
            if generation is None:
                raise ValueError("Generation settings are required for ask/chat.")
            print("Loading local Qwen generation model (CPU)...", file=sys.stderr)
            generator = LocalQwenGenerator(
                generation.model_dir,
                context_window_tokens=generation.context_window_tokens,
                max_generation_seconds=generation.max_generation_seconds,
            )

            def answer_and_display(question: str) -> None:
                result = answer_question(
                    question,
                    settings.app,
                    embedder,
                    generator,
                    count_request_tokens=generator.count_request_tokens,
                    project=args.project,
                )
                # The service returns only after response and citation validation.
                print(terminal_text(f"Snapshot: {result.index_id}"))
                print(f"Status: {result.status}")
                print(terminal_text(result.answer))
                if result.sources:
                    print("\nSources (indexed version):")
                    for source in result.sources:
                        print(terminal_text(source.render()))

            if args.command == "ask":
                answer_and_display(args.question)
                return 0

            print("Independent questions; no conversation history is used or saved.")
            print("Type /quit or /exit to leave. EOF also exits; Ctrl+C interrupts.")
            while True:
                try:
                    question = input("\nQuestion> ").strip()
                except EOFError:
                    print("\nGoodbye.")
                    return 0
                if question.casefold() in ("/quit", "/exit"):
                    print("Goodbye.")
                    return 0
                if not question:
                    continue
                # Reject question-level input errors here. Retrieval/configuration
                # failures still propagate and end the loop instead of being hidden.
                if (
                    embedder.count_tokens(question) > settings.app.max_question_tokens
                    or generator.count_tokens(question)
                    > settings.app.max_question_tokens
                    or embedder.count_tokens(settings.query_prefix + question)
                    > embedder.max_tokens
                ):
                    print(
                        "Question exceeds the configured or embedding-model token limit.",
                        file=sys.stderr,
                    )
                    continue
                try:
                    answer_and_display(question)
                except GenerationError as error:
                    print(terminal_text(f"Generation error: {error}"), file=sys.stderr)

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
    except GenerationError as error:
        print(terminal_text(f"Generation error: {error}"), file=sys.stderr)
        return 5
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        print(terminal_text(f"Error: {error}"), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
