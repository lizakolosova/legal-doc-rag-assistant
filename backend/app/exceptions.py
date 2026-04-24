"""Custom exception hierarchy for the Legal Doc RAG Assistant.

All application-specific errors subclass AppError so callers can catch
the entire family with a single ``except AppError`` clause. Each subclass
carries the context needed to produce a useful HTTP response.

FastAPI exception handlers (registered in main.py) map these to the
appropriate HTTP status codes and response bodies.
"""


class AppError(Exception):
    """Base class for all application-defined exceptions."""


# ---------------------------------------------------------------------------
# Ingestion errors
# ---------------------------------------------------------------------------


class DocumentParseError(AppError):
    """Raised when a document cannot be parsed into text.

    Args:
        filename: Name of the file that failed to parse.
        reason: Human-readable description of the parse failure.
    """

    def __init__(self, filename: str, reason: str) -> None:
        self.filename = filename
        self.reason = reason
        super().__init__(f"Failed to parse '{filename}': {reason}")


class DocumentTooLargeError(AppError):
    """Raised when a document exceeds the configured size or page limit.

    Args:
        filename: Name of the file that exceeded the limit.
        limit_type: Either 'size' or 'pages'.
        actual: The actual value (MB or page count).
        limit: The configured limit.
    """

    def __init__(
        self,
        filename: str,
        limit_type: str,
        actual: int | float,
        limit: int | float,
    ) -> None:
        self.filename = filename
        self.limit_type = limit_type
        self.actual = actual
        self.limit = limit
        super().__init__(
            f"'{filename}' exceeds {limit_type} limit: {actual} > {limit}"
        )


class UnsupportedFormatError(AppError):
    """Raised when a file has a MIME type or extension the system cannot handle.

    Args:
        filename: Name of the uploaded file.
        detected_type: The MIME type or extension that was detected.
    """

    def __init__(self, filename: str, detected_type: str) -> None:
        self.filename = filename
        self.detected_type = detected_type
        super().__init__(
            f"Unsupported format for '{filename}': {detected_type}. "
            "Only PDF and DOCX files are accepted."
        )


# ---------------------------------------------------------------------------
# Embedding errors
# ---------------------------------------------------------------------------


class EmbeddingError(AppError):
    """Raised when the embedding API call fails or returns unexpected output.

    Args:
        reason: Human-readable description of the failure.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Embedding failed: {reason}")


# ---------------------------------------------------------------------------
# Retrieval errors
# ---------------------------------------------------------------------------


class RetrievalError(AppError):
    """Raised when the retrieval pipeline cannot complete a query.

    Args:
        reason: Human-readable description of the failure.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Retrieval failed: {reason}")
