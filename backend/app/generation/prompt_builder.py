import logging

from app.models.schemas import Citation, RetrievedChunk

logger = logging.getLogger(__name__)

_CHUNK_MAX_CHARS = 1500

_SYSTEM_PROMPT = (
    "You are a legal document assistant. Answer questions based ONLY on the provided context. "
    "Cite every claim using inline footnote markers like [1], [2]. "
    "If the context does not contain enough information to answer, say exactly: "
    "'The provided documents do not contain sufficient information to answer this question.' "
    "Never fabricate information."
)

_NO_CONTEXT_USER = (
    "No context documents were provided. "
    "State that you have no context to answer this question."
)


def build_messages(question: str, chunks: list[RetrievedChunk],) -> list[dict]:
    if not chunks:
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _NO_CONTEXT_USER},
        ]

    context_parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        text = chunk.text
        if len(text) > _CHUNK_MAX_CHARS:
            text = text[:_CHUNK_MAX_CHARS] + "..."
        context_parts.append(
            f"[{i}] Source: {chunk.source_file}, Page {chunk.page_number}\n{text}\n---"
        )

    context = "\n".join(context_parts)
    user_content = f"{context}\n\nQuestion: {question}"

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def extract_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    return [
        Citation(
            index=i,
            source_file=chunk.source_file,
            page_number=chunk.page_number,
            chunk_text=chunk.text,
            section_header=chunk.section_header,
        )
        for i, chunk in enumerate(chunks, start=1)
    ]
