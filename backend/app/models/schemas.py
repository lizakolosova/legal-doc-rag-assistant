"""Pydantic request/response schemas shared across the application.

These models are the *only* way data crosses module boundaries.
No raw dicts are passed between ingestion, retrieval, generation, or API layers.
"""

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------


class DocumentUpload(BaseModel):
    """Metadata captured at upload time, before parsing begins.

    Args:
        document_id: Auto-generated UUID for this document.
        filename: Original filename as supplied by the client.
        content_type: MIME type detected or declared by the client.
        size_bytes: Raw file size in bytes.
        uploaded_at: UTC timestamp of the upload request.
    """

    document_id: UUID = Field(default_factory=uuid4)
    filename: str
    content_type: str
    size_bytes: int
    uploaded_at: datetime


class ParsedSection(BaseModel):
    """A contiguous block of text extracted from a document.

    Sections map to pages (PDF) or paragraphs/headings (DOCX).

    Args:
        document_id: UUID of the parent document.
        page_number: 1-based page number; None for DOCX sections without a
            clear page boundary.
        text: Extracted plain text for this section.
        section_index: 0-based position of this section within the document.
    """

    document_id: UUID
    page_number: int | None
    text: str
    section_index: int


class TextChunk(BaseModel):
    """A single chunk produced by the text splitter.

    Args:
        chunk_id: Auto-generated UUID.
        document_id: UUID of the parent document.
        text: Chunk content.
        chunk_index: 0-based position within the document.
        page_number: Source page (carried forward from ParsedSection).
        token_count: Approximate token count (set by chunker).
    """

    chunk_id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    text: str
    chunk_index: int
    page_number: int | None
    token_count: int


class EmbeddedChunk(BaseModel):
    """A TextChunk augmented with its embedding vector.

    Args:
        chunk_id: Matches the originating TextChunk.
        document_id: UUID of the parent document.
        text: Chunk content (duplicated for convenience so callers need only
            this model after embedding).
        chunk_index: 0-based position within the document.
        page_number: Source page.
        embedding: Dense vector produced by the embedding model.
    """

    chunk_id: UUID
    document_id: UUID
    text: str
    chunk_index: int
    page_number: int | None
    embedding: list[float]


# ---------------------------------------------------------------------------
# Query / retrieval pipeline
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Incoming query from the user.

    Args:
        query: The natural-language question.
        document_id: If provided, retrieval is scoped to this document only.
            If omitted, the full corpus is searched.
        top_k: Override the default number of chunks returned to the LLM.
            Falls back to ``settings.rerank_top_k`` when None.
    """

    query: str = Field(..., min_length=1, max_length=2000)
    document_id: UUID | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)


class Citation(BaseModel):
    """A single source citation attached to an answer.

    Args:
        chunk_id: UUID of the source chunk.
        document_id: UUID of the source document.
        filename: Human-readable filename shown in the footnote.
        page_number: Page number for the citation marker.
        chunk_text: The verbatim chunk text shown in the "Show source" toggle.
        relevance_score: Final reranker score (higher = more relevant).
    """

    chunk_id: UUID
    document_id: UUID
    filename: str
    page_number: int | None
    chunk_text: str
    relevance_score: float


class QueryResponse(BaseModel):
    """Full response returned to the client after generation.

    Args:
        query_id: Auto-generated UUID for this query (used for logging).
        query: The original question echoed back.
        answer: LLM-generated answer with inline ``[Doc: …, Page: …]`` markers.
        citations: Ordered list of sources referenced in the answer.
        latency_ms: Per-stage timing summary for observability.
    """

    query_id: UUID = Field(default_factory=uuid4)
    query: str
    answer: str
    citations: list[Citation]
    latency_ms: dict[str, int]
