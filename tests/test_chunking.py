"""Structural chunking and locator regressions; no models or index are needed."""

import hashlib
import random
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import pytest

from course_qa.chunking import chunk_spans, markdown_chunk_spans
from course_qa.config import AppConfig, SourceConfig
from course_qa.ingestion import PROCESSING_VERSION, prepare_chunks


def assert_exact_coverage(text: str, chunks: list[tuple[int, str]], limit: int) -> None:
    covered: set[int] = set()
    for start, piece in chunks:
        assert piece.strip() and len(piece) <= limit
        assert text[start : start + len(piece)] == piece
        covered.update(range(start, start + len(piece)))
    assert all(
        position in covered for position, char in enumerate(text) if not char.isspace()
    )
    assert all(left[0] < right[0] for left, right in pairwise(chunks))


def test_sections_never_share_chunks_or_overlap() -> None:
    text = "# First\n\nFirst paragraph.\n\n## Second\n\nSecond paragraph.\n"
    boundary = text.index("## Second")
    chunks = markdown_chunk_spans(text, 1000, 200)
    assert len(chunks) == 2
    assert chunks[0][0] + len(chunks[0][1]) == boundary
    assert chunks[1][0] == boundary
    assert_exact_coverage(text, chunks, 1000)


@pytest.mark.parametrize("fence", ["```", "~~~", "````"])
def test_code_block_stays_whole_and_internal_headings_are_not_sections(
    fence: str,
) -> None:
    code = f"{fence}python\n# inside code\n\nprint(1)\n{fence}\n"
    text = "# Section\n\nIntro.\n\n" + code + "\nClosing paragraph.\n"
    chunks = markdown_chunk_spans(text, 1000, 0)
    assert len(chunks) == 1 and code in chunks[0][1]
    assert_exact_coverage(text, chunks, 1000)


def test_shorter_or_different_fence_does_not_close_code() -> None:
    text = "````python\n```\n~~~\n# still inside code\n````\n# Real heading\nDone."
    chunks = markdown_chunk_spans(text, 1000, 0)
    assert len(chunks) == 2
    assert "# still inside code" in chunks[0][1]
    assert chunks[1][1].startswith("# Real heading")


def test_unclosed_code_is_preserved_without_synthetic_closing_fence() -> None:
    text = "```python\n# This remains code\n\nprint(1)"
    assert markdown_chunk_spans(text, 1000, 0) == [(0, text)]


def test_small_table_is_not_split_to_fill_previous_chunk() -> None:
    table = "| Name | Value |\n| --- | --- |\n| One | Two |\n"
    text = "A preliminary paragraph with some detail.\n\n" + table
    chunks = markdown_chunk_spans(text, 60, 0)
    assert len(chunks) == 2 and table in chunks[-1][1]
    assert_exact_coverage(text, chunks, 60)


def test_overlap_uses_complete_paragraphs_and_is_bounded() -> None:
    paragraphs = [letter * 20 + ".\n\n" for letter in "abcd"]
    text = "# Topic\n\n" + "".join(paragraphs)
    chunks = markdown_chunk_spans(text, 60, 24)
    starts = {0, *(text.index(paragraph) for paragraph in paragraphs)}
    assert all(start in starts for start, _ in chunks)
    overlaps = [left[0] + len(left[1]) - right[0] for left, right in pairwise(chunks)]
    assert any(overlap > 0 for overlap in overlaps)
    assert all(overlap <= 24 for overlap in overlaps)
    assert_exact_coverage(text, chunks, 60)


@pytest.mark.parametrize("text", ["word " * 100, "x" * 200, "课程内容🙂" * 60])
def test_oversized_blocks_preserve_offsets_and_all_non_whitespace(text: str) -> None:
    chunks = markdown_chunk_spans(text, 31, 10)
    assert len(chunks) > 1
    assert_exact_coverage(text, chunks, 31)
    assert chunks == markdown_chunk_spans(text, 31, 10)


def test_token_limit_counts_prefix_and_special_tokens() -> None:
    text = "# Notes\n\n" + "A long paragraph. " * 30
    prefix = "document: "

    def count_tokens(value: str) -> int:
        return len(value) + 2  # Stand-in tokenizer including two special tokens.

    chunks = markdown_chunk_spans(
        text, 1000, 20, count_tokens=count_tokens, max_tokens=70, prefix=prefix
    )
    assert all(count_tokens(prefix + piece) <= 70 for _, piece in chunks)
    assert_exact_coverage(text, chunks, 1000)


def test_prefix_leaving_no_room_for_content_fails_clearly() -> None:
    with pytest.raises(ValueError, match="exceed the token limit"):
        markdown_chunk_spans(
            "text", 100, 0, count_tokens=len, max_tokens=3, prefix="long prefix"
        )


@pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (5, -1), (5, 5)])
def test_invalid_character_limits_are_rejected(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        markdown_chunk_spans("text", size, overlap)


def test_token_counter_and_limit_must_be_supplied_together() -> None:
    with pytest.raises(ValueError):
        markdown_chunk_spans("text", max_tokens=20)
    with pytest.raises(ValueError):
        markdown_chunk_spans("text", count_tokens=len)
    with pytest.raises(ValueError):
        markdown_chunk_spans("text", count_tokens=len, max_tokens=0)


@pytest.mark.parametrize("text", ["", " \t\n", "### Title\r\n\r\nCafé 与课程.\r\n"])
def test_empty_and_crlf_unicode_documents(text: str) -> None:
    chunks = markdown_chunk_spans(text, 100, 0)
    assert_exact_coverage(text, chunks, 100)
    if not text.strip():
        assert chunks == []


def test_seeded_mixed_documents_keep_coverage_and_forward_progress() -> None:
    rng = random.Random(42)
    blocks = [
        "# Header\n",
        "\n",
        "words here.\n",
        "```\n",
        "~~~\n",
        "| A | B |\n",
        "课程内容\n",
    ]
    for _ in range(100):
        text = "".join(rng.choice(blocks) for _ in range(rng.randint(1, 30)))
        size = rng.randint(1, 120)
        overlap = rng.randrange(size)
        chunks = markdown_chunk_spans(
            text, size, overlap, count_tokens=len, max_tokens=size
        )
        assert_exact_coverage(text, chunks, size)


def test_ingestion_routes_markdown_and_preserves_exclusion_line_mapping(
    tmp_path: Path,
) -> None:
    original = b"# First\r\n\r\nAlpha.\r\nSECRET\r\n## Second\r\n\r\nBeta.\r\n"
    markdown = tmp_path / "notes.MD"
    markdown.write_bytes(original)
    source = SourceConfig(
        "notes",
        "Notes",
        "course",
        markdown,
        tmp_path,
        "student_notes",
        excluded_lines=((4, 4),),
    )
    config = AppConfig(storage_dir=tmp_path / "storage")
    chunks = prepare_chunks([source], config)
    assert [(c.metadata["line_start"], c.metadata["line_end"]) for c in chunks] == [
        (1, 3),
        (5, 7),
    ]
    assert all("SECRET" not in c.text for c in chunks)
    assert all(c.metadata["processing_version"] == PROCESSING_VERSION for c in chunks)
    assert all(
        c.metadata["source_hash"] == hashlib.sha256(original).hexdigest()
        for c in chunks
    )
    assert prepare_chunks([source], config) == chunks
    assert markdown.read_bytes() == original
    assert not config.storage_dir.exists()


def test_plain_text_does_not_treat_hash_lines_as_markdown_sections(
    tmp_path: Path,
) -> None:
    text = "# First\n\nAlpha.\n\n## Second\n\nBeta."
    path = tmp_path / "notes.txt"
    path.write_text(text, encoding="utf-8")
    source = SourceConfig("notes", "Notes", "course", path, tmp_path, "student_notes")
    config = AppConfig(storage_dir=tmp_path / "storage")
    chunks = prepare_chunks([source], config)
    assert len(chunks) == 1 and chunks[0].text == text
    markdown = tmp_path / "notes.md"
    markdown.write_text(text, encoding="utf-8")
    assert len(prepare_chunks([replace(source, path=markdown)], config)) == 2
    assert chunk_spans(text, 1000, 0) == [(0, text)]
