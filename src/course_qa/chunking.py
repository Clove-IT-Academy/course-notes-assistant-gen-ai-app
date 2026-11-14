"""Split text into bounded passages while keeping original character offsets."""

from collections.abc import Callable


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
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be between 0 and chunk_size - 1.")
    if (count_tokens is None) != (max_tokens is None):
        raise ValueError("Supply count_tokens and max_tokens together.")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")

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


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    """Keep the original teaching interface for callers needing only text."""
    return [piece for _, piece in chunk_spans(text, chunk_size, chunk_overlap)]
