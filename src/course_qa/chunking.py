"""Split text into bounded passages while keeping original character offsets."""

import re
from collections.abc import Callable

CHUNKING_VERSION = "markdown-block-token-chunks-v4"


def _validate_limits(
    chunk_size: int,
    chunk_overlap: int,
    count_tokens: Callable[[str], int] | None,
    max_tokens: int | None,
) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be between 0 and chunk_size - 1.")
    if (count_tokens is None) != (max_tokens is None):
        raise ValueError("Supply count_tokens and max_tokens together.")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")


def _preferred_end(text: str, start: int, limit: int) -> int:
    """Prefer a paragraph or line break near the end, then any word boundary."""
    if limit == len(text) or text[limit].isspace() or text[limit - 1].isspace():
        return limit

    midpoint = start + (limit - start) // 2
    for separator in ("\n\n", "\n"):
        position = text.rfind(separator, midpoint, limit)
        if position >= midpoint:
            return position + len(separator)

    for position in range(limit - 1, start, -1):
        if text[position].isspace():
            return position + 1

    # An unbroken string longer than the limit must be split explicitly.
    return limit


def chunk_spans(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    *,
    count_tokens: Callable[[str], int] | None = None,
    max_tokens: int | None = None,
    prefix: str = "",
) -> list[tuple[int, str]]:
    """Return (original character offset, unchanged text) for each chunk.

    Sizes and overlap are in characters. When a tokenizer and token limit are
    supplied together, count the prefix and special tokens through that callback.
    The callback must count the full input without truncation. Overlap is an
    upper bound: it is reduced to keep word boundaries and guarantee progress.
    """
    _validate_limits(chunk_size, chunk_overlap, count_tokens, max_tokens)

    chunks: list[tuple[int, str]] = []
    start = 0
    while start < len(text):
        limit = min(start + chunk_size, len(text))
        end = _preferred_end(text, start, limit)

        if count_tokens is not None and max_tokens is not None:
            while count_tokens(prefix + text[start:end]) > max_tokens:
                if end - start <= 1:
                    raise ValueError("Prefix and one character exceed the token limit.")
                limit = start + max(1, (end - start) // 2)
                end = _preferred_end(text, start, limit)

        piece = text[start:end]
        if piece.strip():
            chunks.append((start, piece))
        if end == len(text):
            break

        overlap = min(chunk_overlap, (end - start) // 2)
        next_start = end - overlap
        # Move a proposed overlap start out of the middle of a word. If the
        # boundary cannot fit within the overlap, use the previous chunk's end.
        while (
            next_start < end
            and next_start > 0
            and not text[next_start - 1].isspace()
            and not text[next_start].isspace()
        ):
            next_start += 1
        while next_start < len(text) and text[next_start].isspace():
            next_start += 1
        start = next_start

    return chunks


def _markdown_blocks(text: str) -> list[list[tuple[int, int]]]:
    """Group contiguous block offsets by ATX section, ignoring fenced headings.

    This small parser handles paragraphs, blank-separated lists/tables, and
    backtick/tilde fenced code. It is not a complete CommonMark parser; Setext
    headings and nested container syntax are treated as ordinary paragraph text.
    """
    sections: list[list[tuple[int, int]]] = []
    blocks: list[tuple[int, int]] = []
    start = offset = 0
    fence_character = ""
    fence_length = 0

    def finish(end: int) -> None:
        nonlocal start
        if end > start:
            blocks.append((start, end))
        start = end

    for line in text.splitlines(keepends=True):
        end = offset + len(line)
        fence = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line.rstrip("\r\n"))
        if fence_character:
            if (
                fence is not None
                and fence.group(1)[0] == fence_character
                and len(fence.group(1)) >= fence_length
                and not fence.group(2).strip()
            ):
                fence_character = ""
                finish(end)
        elif fence is not None and not (
            fence.group(1)[0] == "`" and "`" in fence.group(2)
        ):
            finish(offset)
            fence_character = fence.group(1)[0]
            fence_length = len(fence.group(1))
        elif re.match(r"^ {0,3}#{1,6}(?:[ \t]+|$)", line.rstrip("\r\n")):
            finish(offset)
            if blocks:
                sections.append(blocks)
            blocks = []
        elif not line.strip():
            finish(end)
        offset = end

    finish(len(text))
    if blocks:
        sections.append(blocks)
    return sections


def markdown_chunk_spans(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    *,
    count_tokens: Callable[[str], int] | None = None,
    max_tokens: int | None = None,
    prefix: str = "",
) -> list[tuple[int, str]]:
    """Pack whole Markdown blocks within sections, preserving original offsets.

    A paragraph, table or code fence stays whole when it fits both budgets.
    Oversized blocks use chunk_spans explicitly; fragments are never rewritten
    with synthetic headings or fences. Overlap includes whole trailing blocks
    only, never crosses a heading, and may be zero. Heading ancestry is not added
    to metadata here. Callers must version/rebuild their index when adopting this
    function; the original chunk_spans interface remains available unchanged.
    """
    _validate_limits(chunk_size, chunk_overlap, count_tokens, max_tokens)

    def fits(start: int, end: int) -> bool:
        return end - start <= chunk_size and (
            count_tokens is None
            or max_tokens is None
            or count_tokens(prefix + text[start:end]) <= max_tokens
        )

    chunks: list[tuple[int, str]] = []

    def emit(blocks: list[tuple[int, int]]) -> None:
        if blocks:
            start, end = blocks[0][0], blocks[-1][1]
            if text[start:end].strip():
                chunks.append((start, text[start:end]))

    for section in _markdown_blocks(text):
        pending: list[tuple[int, int]] = []
        for start, end in section:
            if not fits(start, end):
                emit(pending)
                pending = []
                # Keep the fallback's offsets relative to the original document.
                chunks.extend(
                    (start + relative, piece)
                    for relative, piece in chunk_spans(
                        text[start:end],
                        chunk_size,
                        chunk_overlap,
                        count_tokens=count_tokens,
                        max_tokens=max_tokens,
                        prefix=prefix,
                    )
                )
                continue

            if pending and not fits(pending[0][0], end):
                emit(pending)
                retained: list[tuple[int, int]] = []
                for previous in reversed(pending):
                    if pending[-1][1] - previous[0] > chunk_overlap:
                        break
                    retained.insert(0, previous)
                while retained and not fits(retained[0][0], end):
                    retained.pop(0)
                pending = retained
            pending.append((start, end))
        emit(pending)
    return chunks


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    """Keep the original teaching interface for callers needing only text."""
    return [piece for _, piece in chunk_spans(text, chunk_size, chunk_overlap)]
