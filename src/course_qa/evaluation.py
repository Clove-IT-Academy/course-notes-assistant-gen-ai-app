"""Measure retrieval and answer outcomes; factual correctness needs human review."""

import argparse
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

from course_qa.config import AppConfig
from course_qa.contracts import RetrievedPassage, require_integer, require_string
from course_qa.embeddings import Embedder, LocalSentenceEmbedder
from course_qa.generation import GenerationError, Generator
from course_qa.qa import answer_question
from course_qa.retrieval import retrieve_passages
from course_qa.settings import load_settings
from course_qa.store import SnapshotStore


@dataclass(frozen=True)
class EvidenceRange:
    source_id: str
    line_start: int
    line_end: int


@dataclass(frozen=True)
class RetrievalCase:
    case_id: str
    question: str
    expected_status: str
    evidence: tuple[EvidenceRange, ...]
    alternatives: tuple[tuple[EvidenceRange, ...], ...] = ()
    expected_claims: tuple[str, ...] = ()


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise TypeError("Evaluation records must be JSON objects.")
    # JSON object keys are always strings.
    return cast(Mapping[str, object], value)


def _evidence_ranges(value: object) -> tuple[EvidenceRange, ...]:
    if not isinstance(value, list):
        raise TypeError("Evidence groups must be lists of source ranges.")
    ranges: list[EvidenceRange] = []
    for entry in cast(list[object], value):
        span = _mapping(entry)
        first = require_integer(span, "line_start", 1)
        last = require_integer(span, "line_end", first)
        ranges.append(EvidenceRange(require_string(span, "source_id"), first, last))
    return tuple(ranges)


def evidence_group_coverage(
    case: RetrievalCase, passages: Sequence[RetrievedPassage]
) -> list[list[bool]]:
    """Each group is an AND of ranges; alternative groups are combined with OR."""
    return [
        [evidence_covered(span, passages) for span in group]
        for group in (case.evidence, *case.alternatives)
    ]


def load_cases(path: Path, split: str = "development") -> list[RetrievalCase]:
    """Load the chosen split; held-out execution requires explicit selection."""
    if split not in {"development", "held_out"}:
        raise ValueError("Unknown evaluation split.")
    cases: list[RetrievalCase] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw: object = json.loads(line)
            data = _mapping(raw)
            if require_integer(data, "schema_version", 1) != 1:
                raise ValueError("Unsupported evaluation schema.")
            case_id = require_string(data, "case_id")
            if case_id in seen:
                raise ValueError("Duplicate case ID.")
            seen.add(case_id)
            case_split = require_string(data, "split")
            if case_split not in {"development", "held_out"}:
                raise ValueError("Unknown case split.")
            if case_split != split:
                continue
            status = require_string(data, "expected_status")
            if status not in {"answered", "insufficient_evidence"}:
                raise ValueError("Unknown expected status.")
            evidence = _evidence_ranges(data.get("expected_evidence"))
            raw_alternatives = data.get("expected_evidence_alternatives", [])
            if not isinstance(raw_alternatives, list):
                raise TypeError(
                    "expected_evidence_alternatives must be a list of groups."
                )
            alternatives = tuple(
                _evidence_ranges(group)
                for group in cast(list[object], raw_alternatives)
            )
            if any(not group for group in alternatives):
                raise ValueError("Alternative evidence groups must not be empty.")
            if status == "answered" and not evidence:
                raise ValueError("Answerable cases require expected evidence.")
            raw_claims = data.get("expected_claims", [])
            if not isinstance(raw_claims, list):
                raise TypeError("expected_claims must be a list of text claims.")
            claims: list[str] = []
            for claim in cast(list[object], raw_claims):
                if not isinstance(claim, str) or not claim.strip():
                    raise ValueError("Expected claims must be nonempty text.")
                claims.append(claim)
            cases.append(
                RetrievalCase(
                    case_id,
                    require_string(data, "question"),
                    status,
                    evidence,
                    alternatives,
                    tuple(claims),
                )
            )
        except (ValueError, TypeError) as error:
            raise ValueError(
                f"Invalid evaluation case on line {number}: {error}"
            ) from error
    if not cases:
        raise ValueError("No evaluation cases found for the selected split.")
    return cases


def evidence_covered(
    expected: EvidenceRange, passages: Sequence[RetrievedPassage]
) -> bool:
    """Require complete line-range coverage, possibly across multiple chunks."""
    intervals = sorted(
        (passage.chunk.metadata["line_start"], passage.chunk.metadata["line_end"])
        for passage in passages
        if passage.chunk.metadata["source_id"] == expected.source_id
    )
    next_line = expected.line_start
    for first, last in intervals:
        if first > next_line:
            break
        next_line = max(next_line, last + 1)
        if next_line > expected.line_end:
            return True
    return False


def evaluate_retrieval(
    cases: Sequence[RetrievalCase], config: AppConfig, embedder: Embedder
) -> dict[str, object]:
    """Inspect top-k candidates, keeping insufficient-evidence cases unscored."""
    records: list[dict[str, object]] = []
    answerable = 0
    covered_cases = 0
    index_ids: set[str] = set()
    for case in cases:
        started = time.perf_counter()
        index_id, passages = retrieve_passages(case.question, config, embedder)
        elapsed = time.perf_counter() - started
        index_ids.add(index_id)
        groups = evidence_group_coverage(case, passages)
        coverage = groups[0]
        fully_covered: bool | None = None
        if case.expected_status == "answered":
            answerable += 1
            fully_covered = any(bool(group) and all(group) for group in groups)
            covered_cases += int(fully_covered)
        records.append(
            {
                "case_id": case.case_id,
                "index_id": index_id,
                "expected_status": case.expected_status,
                "expected_range_coverage": coverage,
                "evidence_group_coverage": groups,
                "matched_evidence_groups": [
                    index for index, group in enumerate(groups) if group and all(group)
                ],
                "all_expected_ranges_covered": fully_covered,
                "latency_seconds": round(elapsed, 4),
                "retrieved": [
                    {
                        "chunk_id": passage.chunk.chunk_id,
                        "rank": passage.rank,
                        "distance": passage.distance,
                        "source_id": passage.chunk.metadata["source_id"],
                        "line_start": passage.chunk.metadata["line_start"],
                        "line_end": passage.chunk.metadata["line_end"],
                    }
                    for passage in passages
                ],
            }
        )
    if len(index_ids) > 1:
        raise RuntimeError(
            "The active snapshot changed during evaluation; rerun on one snapshot."
        )
    return {
        "metric": "any_complete_evidence_group_coverage_at_k",
        "top_k": config.top_k,
        "embedding_fingerprint": embedder.fingerprint,
        "answerable_cases": answerable,
        "fully_covered_cases": covered_cases,
        "coverage_rate": covered_cases / answerable if answerable else None,
        "limitations": "Line coverage is a retrieval proxy, not claim support or answer accuracy. "
        "Abstention and adversarial behavior require generation evaluation. "
        "Line ranges apply only to the reviewed source version.",
        "cases": records,
    }


def evaluate_answers(
    cases: Sequence[RetrievalCase],
    config: AppConfig,
    embedder: Embedder,
    generator: Generator,
    *,
    count_request_tokens: Callable[[str, str], int],
    answer_mode: Literal["generated", "verified_excerpts"] = "generated",
) -> dict[str, object]:
    """Run independent questions and retain validated answers for human review.

    Status agreement is not answer accuracy. Generation failures count as
    failures, never abstentions. Input/storage failures abort the evaluation.
    Results contain course-derived text; save only to an explicit report path.
    """
    if not cases:
        raise ValueError("Answer evaluation requires at least one case.")
    if answer_mode not in ("generated", "verified_excerpts"):
        raise ValueError("Unknown answer mode.")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Evaluation case IDs must be unique.")
    store = SnapshotStore(config.storage_dir)

    def active_index() -> str:
        with store.locked():
            manifest = store.read_active_manifest()
        if manifest is None:
            raise FileNotFoundError("No active index. Ingest notes first.")
        return manifest["index_id"]

    index_id = active_index()
    records: list[dict[str, object]] = []
    matches = errors = abstention_cases = correct_abstentions = 0
    for case in cases:
        if active_index() != index_id:
            raise RuntimeError("The active snapshot changed during answer evaluation.")
        started = time.perf_counter()
        record: dict[str, object] = {
            "case_id": case.case_id,
            "question": case.question,
            "expected_status": case.expected_status,
            "expected_claims": list(case.expected_claims),
            "human_review": "pending",
            "answer_mode": answer_mode,
        }
        expected_output_status = (
            "evidence_selected"
            if answer_mode == "verified_excerpts" and case.expected_status == "answered"
            else case.expected_status
        )
        record["expected_output_status"] = expected_output_status
        abstention_cases += int(case.expected_status == "insufficient_evidence")
        try:
            result = answer_question(
                case.question,
                config,
                embedder,
                generator,
                count_request_tokens=count_request_tokens,
                answer_mode=answer_mode,
            )
            if result.index_id != index_id:
                raise RuntimeError(
                    "The active snapshot changed during answer evaluation."
                )
            matched = result.status == expected_output_status
            matches += int(matched)
            correct_abstentions += int(
                matched and case.expected_status == "insufficient_evidence"
            )
            record.update(
                actual_status=result.status,
                status_matches=matched,
                answer=result.answer,
                citations=[asdict(source) for source in result.sources],
                citation_validation="passed",
                prompt_version=result.prompt_version,
                generation_model_id=result.generation_model_id,
                generation_called=result.generation_called,
                excerpts=[asdict(excerpt) for excerpt in result.excerpts],
                excerpt_integrity="passed" if result.excerpts else "not_applicable",
            )
        except GenerationError as error:
            errors += 1
            record.update(
                actual_status="generation_error",
                status_matches=False,
                error_type=type(error).__name__,
                citation_validation="no_answer_accepted",
            )
        record["latency_seconds"] = round(time.perf_counter() - started, 4)
        records.append(record)
    if active_index() != index_id:
        raise RuntimeError("The active snapshot changed during answer evaluation.")
    return {
        "metric": (
            "expected_evidence_selection_status_agreement"
            if answer_mode == "verified_excerpts"
            else "expected_response_status_agreement"
        ),
        "answer_mode": answer_mode,
        "index_id": index_id,
        "embedding_fingerprint": embedder.fingerprint,
        "generation_model_id": generator.model_id,
        "case_count": len(cases),
        "matching_status_cases": matches,
        "status_agreement_rate": matches / len(cases),
        "generation_errors": errors,
        "expected_abstention_cases": abstention_cases,
        "correct_abstentions": correct_abstentions,
        "abstention_rate": (
            correct_abstentions / abstention_cases if abstention_cases else None
        ),
        "human_review": "pending",
        "release_gate": "not_assessed",
        "limitations": "Status agreement and valid citations do not establish factual "
        "support, expected-claim coverage or injection resistance. Human review is "
        "required; no answer-accuracy or release-pass claim is made. Model/provider "
        "token usage is not measured. Source ranges require the reviewed version. "
        "Verified excerpts establish copying integrity, not relevance or completeness.",
        "cases": records,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=root / "eval/cases.jsonl")
    parser.add_argument("--settings", type=Path, default=root / "configs/app.yaml")
    parser.add_argument("--mode", choices=("retrieval", "answers"), default="retrieval")
    parser.add_argument(
        "--answer-mode", choices=("generated", "verified_excerpts"), default="generated"
    )
    parser.add_argument(
        "--output", type=Path, help="Write a new report; never overwrite."
    )
    parser.add_argument(
        "--split", choices=("development", "held_out"), default="development"
    )
    args = parser.parse_args()
    if args.mode == "retrieval" and args.answer_mode != "generated":
        parser.error("--answer-mode applies only to --mode answers.")
    if args.output is not None and args.output.exists():
        parser.error("Output already exists; choose a new report path.")
    cases = load_cases(args.cases, args.split)
    settings = load_settings(args.settings)
    if args.mode == "answers" and settings.generation is None:
        parser.error("Answer evaluation requires generation settings.")
    embedder = LocalSentenceEmbedder(
        settings.model_dir,
        batch_size=settings.app.embedding_batch_size,
        document_prefix=settings.document_prefix,
        query_prefix=settings.query_prefix,
    )
    if args.mode == "answers":
        from course_qa.local_generation import LocalQwenGenerator

        generation = settings.generation
        if generation is None:
            parser.error("Answer evaluation requires generation settings.")
        generator = LocalQwenGenerator(
            generation.model_dir,
            context_window_tokens=generation.context_window_tokens,
            max_generation_seconds=generation.max_generation_seconds,
        )
        report = evaluate_answers(
            cases,
            settings.app,
            embedder,
            generator,
            count_request_tokens=generator.count_request_tokens,
            answer_mode=args.answer_mode,
        )
    else:
        report = evaluate_retrieval(cases, settings.app, embedder)
    report["split"] = args.split
    rendered = json.dumps(report, indent=2, ensure_ascii=True)
    if args.output is None:
        print(rendered)
    else:
        with args.output.open("x", encoding="utf-8") as output:
            output.write(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
