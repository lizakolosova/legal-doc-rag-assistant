from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.database import Chunk
from app.models.schemas import RetrievedChunk
from app.retrieval.bm25_search import bm25_search, build_bm25_index
from app.retrieval.vector_search import vector_search

if TYPE_CHECKING:
    from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    vector_results: list[RetrievedChunk],
    bm25_results: list[RetrievedChunk],
    k: int = 60,
) -> list[RetrievedChunk]:
    """Merge two ranked lists using Reciprocal Rank Fusion.

    Each chunk's RRF score is the sum of 1/(k + rank) across both lists.
    Chunks that appear in both lists receive a higher combined score.

    Args:
        vector_results: Chunks ranked by vector similarity.
        bm25_results: Chunks ranked by BM25 score.
        k: RRF constant (default 60 per the original paper).

    Returns:
        Deduplicated chunks sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}

    for rank, chunk in enumerate(vector_results, start=1):
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
        chunk_map[chunk.chunk_id] = chunk

    for rank, chunk in enumerate(bm25_results, start=1):
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
        chunk_map[chunk.chunk_id] = chunk

    return [
        chunk_map[chunk_id].model_copy(update={"score": rrf_score})
        for chunk_id, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    ]


async def hybrid_search(
    query: str,
    session: AsyncSession,
    top_k: int | None = None,
    document_ids: list[str] | None = None,
    bm25_index: tuple[BM25Okapi, list[Chunk]] | None = None,
) -> list[RetrievedChunk]:
    """Run vector search and BM25 in parallel, then merge with RRF.

    Args:
        query: Natural-language question.
        session: Active async SQLAlchemy session (used to build BM25 index if not supplied).
        top_k: Number of merged results to return; defaults to settings.retrieval_top_k.
        document_ids: If set, restricts both searches to these document IDs.
        bm25_index: Pre-built (index, chunks) tuple; rebuilt from Postgres if None.

    Returns:
        Merged and deduplicated chunks sorted by RRF score descending.
    """
    k = top_k if top_k is not None else settings.retrieval_top_k
    start = time.monotonic()

    if bm25_index is None:
        bm25_index = await build_bm25_index(session)

    bm25_idx, chunks = bm25_index

    async def _run_bm25() -> list[RetrievedChunk]:
        return bm25_search(query, chunks, bm25_idx, top_k=k, document_ids=document_ids)

    vector_results, bm25_results = await asyncio.gather(
        vector_search(query, top_k=k, document_ids=document_ids),
        _run_bm25(),
    )

    merged = reciprocal_rank_fusion(vector_results, bm25_results)[:k]

    elapsed_ms = int((time.monotonic() - start) * 1000)
    logger.info(
        "hybrid_search query='%.50s' vector=%d bm25=%d merged=%d elapsed=%dms",
        query,
        len(vector_results),
        len(bm25_results),
        len(merged),
        elapsed_ms,
    )

    return merged