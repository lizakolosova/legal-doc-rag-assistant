"""Document parser for PDF and DOCX files.

Converts raw files into ParsedSection lists for downstream chunking.
"""

import logging
import time
from pathlib import Path
from uuid import UUID

import fitz  # PyMuPDF
from docx import Document

from app.config import settings
from app.exceptions import (
    DocumentParseError,
    DocumentTooLargeError,
    UnsupportedFormatError,
)
from app.models.schemas import ParsedSection

logger = logging.getLogger(__name__)

_HEADING_STYLES: frozenset[str] = frozenset(f"Heading {i}" for i in range(1, 10))


def parse_pdf(file_path: Path, document_id: UUID) -> list[ParsedSection]:
    """Extract text per page from a PDF file.

    Args:
        file_path: Path to the PDF file on disk.
        document_id: UUID of the parent document (propagated to all sections).

    Returns:
        List of ParsedSection, one per page that yields non-empty text.

    Raises:
        DocumentParseError: If a page contains only images (scanned), or if
            the entire document yields no extractable text.
    """
    start = time.monotonic()
    doc = fitz.open(str(file_path))
    sections: list[ParsedSection] = []

    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            text = page.get_text().strip()

            if not text:
                if page.get_images():
                    raise DocumentParseError(
                        filename=file_path.name,
                        reason=(
                            f"Page {page_index + 1} appears to be a scanned image. "
                            "This system requires text-based PDFs. "
                            "Please upload an OCR'd version."
                        ),
                    )
                continue

            sections.append(
                ParsedSection(
                    document_id=document_id,
                    page_number=page_index + 1,
                    text=text,
                    section_index=len(sections),
                )
            )
    finally:
        doc.close()

    if not sections:
        raise DocumentParseError(
            filename=file_path.name,
            reason="Document yields no extractable text.",
        )

    elapsed_ms = int((time.monotonic() - start) * 1000)
    total_chars = sum(len(s.text) for s in sections)
    logger.info(
        "Parsed PDF '%s': %d pages, %d chars extracted in %dms",
        file_path.name,
        len(sections),
        total_chars,
        elapsed_ms,
    )
    return sections


def parse_docx(file_path: Path, document_id: UUID) -> list[ParsedSection]:
    """Extract text grouped by headings from a DOCX file.

    Paragraphs are grouped under their nearest preceding heading. A new section
    is emitted whenever a heading-style paragraph is encountered. Paragraphs
    that appear before the first heading form their own leading section.

    If the document contains no heading-style paragraphs, all text is placed
    in a single ParsedSection.

    Args:
        file_path: Path to the DOCX file on disk.
        document_id: UUID of the parent document.

    Returns:
        List of ParsedSection, one per heading group (or one total if no
        headings exist).

    Raises:
        DocumentParseError: If no text can be extracted from the document.
    """
    start = time.monotonic()
    doc = Document(str(file_path))
    sections: list[ParsedSection] = []
    current_parts: list[str] = []

    for para in doc.paragraphs:
        stripped = para.text.strip()
        if para.style.name in _HEADING_STYLES and stripped:
            if current_parts:
                sections.append(
                    ParsedSection(
                        document_id=document_id,
                        page_number=None,
                        text="\n".join(current_parts),
                        section_index=len(sections),
                    )
                )
            current_parts = [stripped]
        elif stripped:
            current_parts.append(stripped)

    if current_parts:
        sections.append(
            ParsedSection(
                document_id=document_id,
                page_number=None,
                text="\n".join(current_parts),
                section_index=len(sections),
            )
        )

    if not sections:
        raise DocumentParseError(
            filename=file_path.name,
            reason="Document yields no extractable text.",
        )

    elapsed_ms = int((time.monotonic() - start) * 1000)
    total_chars = sum(len(s.text) for s in sections)
    logger.info(
        "Parsed DOCX '%s': %d sections, %d chars extracted in %dms",
        file_path.name,
        len(sections),
        total_chars,
        elapsed_ms,
    )
    return sections


def parse_document(file_path: Path, document_id: UUID) -> list[ParsedSection]:
    """Route a document to the appropriate parser after size validation.

    Checks file size against settings.max_file_size_mb. For PDFs, also checks
    page count against settings.max_pages. Dispatches to parse_pdf or
    parse_docx based on file extension.

    Args:
        file_path: Path to the document file on disk.
        document_id: UUID of the parent document.

    Returns:
        List of ParsedSection from the appropriate sub-parser.

    Raises:
        DocumentTooLargeError: If the file exceeds the size or page limit.
        UnsupportedFormatError: If the extension is not .pdf or .docx.
        DocumentParseError: If the file cannot be parsed into text.
    """
    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise DocumentTooLargeError(
            filename=file_path.name,
            limit_type="size",
            actual=round(size_mb, 2),
            limit=settings.max_file_size_mb,
        )

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        doc = fitz.open(str(file_path))
        page_count = len(doc)
        doc.close()
        if page_count > settings.max_pages:
            raise DocumentTooLargeError(
                filename=file_path.name,
                limit_type="pages",
                actual=page_count,
                limit=settings.max_pages,
            )
        return parse_pdf(file_path, document_id)

    if suffix == ".docx":
        return parse_docx(file_path, document_id)

    raise UnsupportedFormatError(
        filename=file_path.name,
        detected_type=suffix,
    )
