import enum
from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel
from pydantic.fields import Field


class DocumentStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    ready = "ready"
    failed = "failed"


class DocumentUpload(BaseModel):

    document_id: UUID = Field(default_factory=uuid4)
    filename: str
    content_type: str
    size_bytes: int
    uploaded_at: datetime


class ParsedSection(BaseModel):
    document_id: UUID
    source_file: str | None = None
    page_number: int | None
    section_header: str | None = None
    text: str
    section_index: int


class TextChunk(BaseModel):

    chunk_id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    source_file: str | None = None
    section_header: str | None = None
    text: str
    chunk_index: int
    total_chunks: int
    page_number: int | None
    token_count: int


class EmbeddedChunk(BaseModel):

    chunk_id: UUID
    document_id: UUID
    text: str
    chunk_index: int
    page_number: int | None
    source_file: str | None = None
    section_header: str | None = None
    embedding: list[float]

class RetrievedChunk(BaseModel):

    chunk_id: str
    document_id: str
    text: str
    score: float
    source_file: str
    page_number: int
    section_header: str | None


class QueryRequest(BaseModel):

    query: str = Field(..., min_length=1, max_length=2000)
    document_id: UUID | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)


class Citation(BaseModel):

    chunk_id: UUID
    document_id: UUID
    filename: str
    page_number: int | None
    chunk_text: str
    relevance_score: float


class QueryResponse(BaseModel):

    query_id: UUID = Field(default_factory=uuid4)
    query: str
    answer: str
    citations: list[Citation]
    latency_ms: dict[str, int]


class DocumentResponse(BaseModel):

    document_id: UUID
    filename: str
    status: DocumentStatus
    num_chunks: int | None = None
    upload_time: datetime
    file_size_bytes: int
    error_message: str | None = None


class IngestResponse(BaseModel):

    document_id: UUID
    status: DocumentStatus
