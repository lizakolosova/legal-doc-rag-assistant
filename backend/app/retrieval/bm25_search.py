from __future__ import annotations

import logging
import string
import time
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.models.database import Chunk
from app.models.schemas import RetrievedChunk

if TYPE_CHECKING:
    from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    lowered = text.lower()
    stripped = lowered.translate(str.maketrans("", "", string.punctuation))
    return stripped.split()


async def build_bm25_index(session: AsyncSession) -> tuple[BM25Okapi, list[Chunk]]:
    from rank_bm25 import BM25Okapi  # deferred: not available outside Docker

    result = await session.execute(
        select(Chunk).options(selectinload(Chunk.document))
    )
    chunks: list[Chunk] = list(result.scalars().all())
    tokenized = [_tokenize(chunk.text) for chunk in chunks]
    return BM25Okapi(tokenized), chunks


def bm25_search(
    query: str,
    chunks: list[Chunk],
    bm25_index: BM25Okapi,
    top_k: int | None = None,
    document_ids: list[str] | None = None,
) -> list[RetrievedChunk]:
    if not chunks:
        return []

    k = top_k if top_k is not None else settings.retrieval_top_k
    start = time.monotonic()

    tokenized_query = _tokenize(query)
    scores: list[float] = bm25_index.get_scores(tokenized_query).tolist()

    if document_ids:
        doc_id_set = {str(did) for did in document_ids}
        scores = [
            score if str(chunk.document_id) in doc_id_set else 0.0
            for score, chunk in zip(scores, chunks)
        ]

    max_score = max(scores) if scores else 0.0
    if max_score > 0:
        normalized = [s / max_score for s in scores]
    else:
        normalized = list(scores)

    scored = sorted(zip(normalized, chunks), key=lambda x: x[0], reverse=True)

    results = [
        RetrievedChunk(
            chunk_id=str(chunk.chroma_id),
            document_id=str(chunk.document_id),
            text=chunk.text,
            score=score,
            source_file=chunk.document.filename if chunk.document else "",
            page_number=chunk.page_number if chunk.page_number is not None else -1,
            section_header=chunk.section_header,
        )
        for score, chunk in scored[:k]
        if score > 0
    ]

    elapsed_ms = int((time.monotonic() - start) * 1000)
    filter_desc = f"document_ids={document_ids}" if document_ids else "no filter"
    logger.info(
        "bm25_search query='%.50s' top_k=%d filter=%s results=%d elapsed=%dms",
        query,
        k,
        filter_desc,
        len(results),
        elapsed_ms,
    )

    return results