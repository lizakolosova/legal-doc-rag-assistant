"""FastAPI application entry point.

Creates the app, registers exception handlers, mounts routers, and exposes
a /health endpoint for liveness checks.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.exceptions import (
    AppError,
    DocumentParseError,
    DocumentTooLargeError,
    EmbeddingError,
    RetrievalError,
    UnsupportedFormatError,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Legal Doc RAG Assistant",
    description=(
        "Upload legal PDFs or DOCX files and ask questions. "
        "Answers include inline citations back to the source document."
    ),
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(DocumentTooLargeError)
async def document_too_large_handler(
    request: Request, exc: DocumentTooLargeError
) -> JSONResponse:
    """Return 400 when a document exceeds the configured size or page limit."""
    logger.warning("Document too large: %s", exc)
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(UnsupportedFormatError)
async def unsupported_format_handler(
    request: Request, exc: UnsupportedFormatError
) -> JSONResponse:
    """Return 415 for unsupported file types."""
    logger.warning("Unsupported format: %s", exc)
    return JSONResponse(status_code=415, content={"detail": str(exc)})


@app.exception_handler(DocumentParseError)
async def document_parse_handler(
    request: Request, exc: DocumentParseError
) -> JSONResponse:
    """Return 422 when a document cannot be parsed (e.g. scanned PDF)."""
    logger.error("Parse error: %s", exc)
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(EmbeddingError)
async def embedding_error_handler(
    request: Request, exc: EmbeddingError
) -> JSONResponse:
    """Return 502 when the embedding API call fails."""
    logger.error("Embedding error: %s", exc)
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(RetrievalError)
async def retrieval_error_handler(
    request: Request, exc: RetrievalError
) -> JSONResponse:
    """Return 500 when the retrieval pipeline fails."""
    logger.error("Retrieval error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.exception_handler(AppError)
async def generic_app_error_handler(
    request: Request, exc: AppError
) -> JSONResponse:
    """Catch-all for any AppError subclass not handled above."""
    logger.error("Unhandled application error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    """Liveness probe for Docker / load-balancer health checks.

    Returns:
        A JSON object with ``status: ok``.
    """
    return {"status": "ok"}
