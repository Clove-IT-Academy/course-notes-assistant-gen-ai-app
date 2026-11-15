from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from course_qa.errors import SourceReadError


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int
    text: str


def load_pdf_pages(
    file_path: Path,
    *,
    max_pages: int = 200,
    max_characters: int = 2_000_000,
    excluded_pages: tuple[int, ...] = (),
    data: bytes | None = None,
) -> list[ExtractedPage]:
    """Extract text per physical page; no OCR or password attempts.

    Empty/image-only included pages fail rather than silently losing evidence.
    Exclusions are explicit one-based physical pages, even across skipped pages.
    """
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    if max_pages <= 0 or max_characters <= 0:
        raise ValueError("PDF limits must be positive.")
    try:
        reader = PdfReader(
            BytesIO(file_path.read_bytes() if data is None else data), strict=True
        )
        if reader.is_encrypted:
            raise SourceReadError(
                "Encrypted PDFs are not supported; provide an approved unencrypted copy."
            )
        count = len(reader.pages)
        if count == 0 or count > max_pages:
            raise SourceReadError("PDF has no pages or exceeds the page limit.")
        if any(
            type(page) is not int or not 1 <= page <= count for page in excluded_pages
        ):
            raise SourceReadError("A PDF exclusion is outside its physical page range.")
        result: list[ExtractedPage] = []
        total = 0
        for number, page in enumerate(reader.pages, 1):
            if number in excluded_pages:
                continue
            contents = page.get_contents()
            if contents is not None and len(contents.get_data()) > 10_000_000:
                raise SourceReadError(
                    f"PDF page {number} exceeds the decoded-content limit."
                )
            text = page.extract_text()
            if not text.strip():
                raise SourceReadError(
                    f"PDF page {number} has no extractable text (blank/scanned); OCR is outside this app."
                )
            total += len(text)
            if total > max_characters:
                raise SourceReadError("PDF extraction exceeds the character limit.")
            result.append(ExtractedPage(number, text))
        if not result:
            raise SourceReadError("No included PDF pages remain.")
        return result
    except SourceReadError:
        raise
    except (OSError, PyPdfError, ValueError, TypeError, RecursionError) as error:
        raise SourceReadError(f"Cannot extract PDF: {file_path.name}") from error


def load_text_file(file_path: str) -> str:
    """
    Load the content of a text / md file.

    Args:
        file_path (str): The path to the text / md file.

    Returns:
        str: The content of the text / md file.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    if path.suffix.lower() not in {".md", ".txt"}:
        raise ValueError(
            f"Unsupported file type: {path.suffix}. Only .md and .txt files are supported."
        )

    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("The course-notes file is empty.")

    return text
