"""Measure evidence-line coverage; this does not score answer correctness."""

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from course_qa.config import AppConfig
from course_qa.contracts import RetrievedPassage, require_integer, require_string
from course_qa.embeddings import Embedder, LocalSentenceEmbedder
from course_qa.retrieval import retrieve_passages
from course_qa.settings import load_settings


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


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise TypeError("Evaluation records must be JSON objects.")
    # JSON object keys are always strings.
    return cast(Mapping[str, object], value)


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
            entries = data.get("expected_evidence")
            if not isinstance(entries, list):
                raise TypeError("expected_evidence must be a list.")
            evidence: list[EvidenceRange] = []
            for entry in cast(list[object], entries):
                span = _mapping(entry)
                first = require_integer(span, "line_start", 1)
                last = require_integer(span, "line_end", first)
                evidence.append(
                    EvidenceRange(require_string(span, "source_id"), first, last)
                )
            if status == "answered" and not evidence:
                raise ValueError("Answerable cases require expected evidence.")
            cases.append(
                RetrievalCase(
                    case_id, require_string(data, "question"), status, tuple(evidence)
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
        coverage = [evidence_covered(span, passages) for span in case.evidence]
        fully_covered: bool | None = None
        if case.expected_status == "answered":
            answerable += 1
            fully_covered = all(coverage)
            covered_cases += int(fully_covered)
        records.append(
            {
                "case_id": case.case_id,
                "index_id": index_id,
                "expected_status": case.expected_status,
                "expected_range_coverage": coverage,
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
        "metric": "full_expected_line_range_coverage_at_k",
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


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=root / "eval/cases.jsonl")
    parser.add_argument("--settings", type=Path, default=root / "configs/app.yaml")
    parser.add_argument(
        "--split", choices=("development", "held_out"), default="development"
    )
    args = parser.parse_args()
    cases = load_cases(args.cases, args.split)
    settings = load_settings(args.settings)
    embedder = LocalSentenceEmbedder(
        settings.model_dir,
        batch_size=settings.app.embedding_batch_size,
        document_prefix=settings.document_prefix,
        query_prefix=settings.query_prefix,
    )
    report = evaluate_retrieval(cases, settings.app, embedder)
    report["split"] = args.split
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
