class AppError(Exception):


class DocumentParseError(AppError):
    def __init__(self, filename: str, reason: str) -> None:
        self.filename = filename
        self.reason = reason
        super().__init__(f"Failed to parse '{filename}': {reason}")


class DocumentTooLargeError(AppError):
    def __init__(self, filename: str,limit_type: str, actual: int | float,limit: int | float) -> None:
        self.filename = filename
        self.limit_type = limit_type
        self.actual = actual
        self.limit = limit
        super().__init__(
            f"'{filename}' exceeds {limit_type} limit: {actual} > {limit}"
        )


class UnsupportedFormatError(AppError):
    def __init__(self, filename: str, detected_type: str) -> None:
        self.filename = filename
        self.detected_type = detected_type
        super().__init__(
            f"Unsupported format for '{filename}': {detected_type}. "
            "Only PDF and DOCX files are accepted."
        )

class EmbeddingError(AppError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Embedding failed: {reason}")

class RetrievalError(AppError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Retrieval failed: {reason}")
