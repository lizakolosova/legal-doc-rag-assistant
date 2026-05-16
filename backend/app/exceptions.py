class AppError(Exception):
    """Base class for all application-level errors."""


class DocumentParseError(AppError):
    """Raised when a document cannot be parsed into text."""

    def __init__(self, filename: str, reason: str) -> None:
        self.filename = filename
        self.reason = reason
        super().__init__(f"Failed to parse '{filename}': {reason}")


class DocumentTooLargeError(AppError):
    """Raised when a document exceeds the configured size or page-count limit."""

    def __init__(self, filename: str, limit_type: str, actual: int | float, limit: int | float) -> None:
        self.filename = filename
        self.limit_type = limit_type
        self.actual = actual
        self.limit = limit
        super().__init__(
            f"'{filename}' exceeds {limit_type} limit: {actual} > {limit}"
        )


class UnsupportedFormatError(AppError):
    """Raised when an uploaded file is not a supported format (PDF or DOCX)."""

    def __init__(self, filename: str, detected_type: str) -> None:
        self.filename = filename
        self.detected_type = detected_type
        super().__init__(
            f"Unsupported format for '{filename}': {detected_type}. "
            "Only PDF and DOCX files are accepted."
        )


class EmbeddingError(AppError):
    """Raised when the embedding model fails to produce vectors."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Embedding failed: {reason}")


class RetrievalError(AppError):
    """Raised when a retrieval operation (vector search or BM25) fails."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Retrieval failed: {reason}")


class GenerationError(AppError):
    """Raised when the LLM generation call fails after all retries."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Generation failed: {reason}")
