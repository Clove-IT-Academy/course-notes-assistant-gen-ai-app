"""Text-PDF extraction, original page locators, persistence and failures."""

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    IndirectObject,
    NameObject,
    PdfObject,
)

from course_qa.config import AppConfig, SourceConfig
from course_qa.context import build_context
from course_qa.errors import SourceReadError
from course_qa.evaluation import EvidenceRange, evidence_covered
from course_qa.excerpts import verified_source_excerpts
from course_qa.ingestion import ingest, prepare_chunks
from course_qa.loaders import load_pdf_pages
from course_qa.retrieval import retrieve_passages
from course_qa.source_manifest import load_source_manifest


class FixtureWriter(PdfWriter):
    def fixture_object(self, value: PdfObject) -> IndirectObject:
        return self._add_object(value)


def write_pdf(
    path: Path, pages: Sequence[str | None], *, encrypted: bool = False
) -> None:
    writer = FixtureWriter()
    font = writer.fixture_object(
        DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
    )
    for text in pages:
        page = writer.add_blank_page(width=400, height=400)
        if text is not None:
            page[NameObject("/Resources")] = DictionaryObject(
                {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
            )
            stream = DecodedStreamObject()
            escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream.set_data(f"BT /F1 12 Tf 30 350 Td ({escaped}) Tj ET".encode("ascii"))
            page[NameObject("/Contents")] = writer.fixture_object(stream)
    if encrypted:
        writer.encrypt("secret")
    with path.open("wb") as target:
        writer.write(target)


class TinyEmbedder:
    fingerprint = "pdf-test-v1"
    dimension = 2

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def test_pdf_pages_keep_physical_numbers_after_explicit_exclusions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "notes.pdf"
    write_pdf(path, ["First physical page.", None, "Third physical page."])
    pages = load_pdf_pages(path, excluded_pages=(2,))
    assert [p.page_number for p in pages] == [1, 3]
    assert "Third physical page" in pages[1].text
    with pytest.raises(SourceReadError, match="page 2"):
        load_pdf_pages(path)


def test_encrypted_malformed_and_oversized_pdfs_fail_clearly(tmp_path: Path) -> None:
    path = tmp_path / "notes.pdf"
    write_pdf(path, ["Secret"], encrypted=True)
    with pytest.raises(SourceReadError, match="Encrypted"):
        load_pdf_pages(path)
    path.write_bytes(b"not a PDF")
    with pytest.raises(SourceReadError, match="Cannot extract"):
        load_pdf_pages(path)
    write_pdf(path, ["Page one", "Page two"])
    with pytest.raises(SourceReadError, match="page limit"):
        load_pdf_pages(path, max_pages=1)
    with pytest.raises(SourceReadError, match="character limit"):
        load_pdf_pages(path, max_characters=2)
    with pytest.raises(SourceReadError, match="physical page"):
        load_pdf_pages(path, excluded_pages=(3,))
    with pytest.raises(SourceReadError, match="No included"):
        load_pdf_pages(path, excluded_pages=(1, 2))


def test_pdf_chunks_persist_page_citations_and_verify_excerpts(tmp_path: Path) -> None:
    path = tmp_path / "notes.pdf"
    write_pdf(
        path,
        ["OpenCV uses BGR channels.", "Grayscale uses weighted channel intensities."],
    )
    original = path.read_bytes()
    source = SourceConfig(
        "pdf-notes", "PDF Notes", "course", path, tmp_path, "student_notes"
    )
    config = AppConfig(storage_dir=tmp_path / "store", max_output_tokens=40)
    chunks = prepare_chunks([source], config)
    assert [c.metadata.get("page_number") for c in chunks] == [1, 2]
    assert all(c.metadata["line_start"] == 1 for c in chunks)
    assert chunks == prepare_chunks([source], config)
    result = ingest([source], config, TinyEmbedder())
    assert result["status"] != "preview"
    index_id, passages = retrieve_passages(
        "What are the channel conventions?", config, TinyEmbedder()
    )
    assert result["index_id"] == index_id
    assert evidence_covered(EvidenceRange("pdf-notes", 1, 1, 1), passages)
    assert not evidence_covered(EvidenceRange("pdf-notes", 1, 1, 3), passages)
    assert not evidence_covered(EvidenceRange("pdf-notes", 1, 1), passages)
    bundle = build_context(index_id, passages, config, len, 20_000, 0)
    excerpts = verified_source_excerpts(bundle)
    assert {e.reference.page_number for e in excerpts} == {1, 2}
    assert all("page " in e.reference.render() for e in excerpts)
    assert path.read_bytes() == original


def test_manifest_pdf_exclusions_and_format_specific_validation(tmp_path: Path) -> None:
    path = tmp_path / "notes.pdf"
    write_pdf(path, ["First", None, "Last"])
    manifest = tmp_path / "sources.yaml"
    manifest.write_text(
        "schema_version: 1\nsources:\n  - source_id: pdf\n    title: PDF\n    project: course\n    document_type: student_notes\n    path: notes.pdf\n    approved_root: .\n    excluded_pages: [2]\n",
        encoding="utf-8",
    )
    source = load_source_manifest(manifest)[0]
    assert source.excluded_pages == (2,)
    assert len(prepare_chunks([source], AppConfig(storage_dir=tmp_path / "store"))) == 2
    with pytest.raises(ValueError, match="excluded_pages"):
        replace(source, excluded_lines=((1, 1),)).resolve_path()
    with pytest.raises(ValueError, match="sorted"):
        replace(source, excluded_pages=(2, 1)).resolve_path()
    text = tmp_path / "notes.txt"
    text.write_text("Text", encoding="utf-8")
    with pytest.raises(ValueError, match="only to PDFs"):
        replace(source, path=text).resolve_path()
