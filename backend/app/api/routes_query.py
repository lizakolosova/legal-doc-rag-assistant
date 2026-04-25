import logging
import time

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_async_session
from app.models.schemas import QueryRequest, QueryResponse
from app.retrieval.hybrid import hybrid_search
from app.retrieval.reranker import rerank

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/query", tags=["query"])


async def _get_session() -> AsyncSession:
    session_factory = get_async_session()
    async with session_factory() as session:
        yield session


@router.post("", response_model=QueryResponse)
async def query_documents( body: QueryRequest, session: AsyncSession = Depends(_get_session)) -> QueryResponse:
    start = time.monotonic()
    k = body.top_k

    results = await hybrid_search(
        body.question,
        session,
        top_k=k * 2,
        document_ids=body.document_ids,
    )

    if body.use_reranker:
        chunks = rerank(body.question, results, top_k=k)
        retrieval_method = "hybrid+reranker"
    else:
        chunks = results[:k]
        retrieval_method = "hybrid"

    latency_ms = (time.monotonic() - start) * 1000

    logger.info(
        "query question='%.50s' method=%s chunks=%d elapsed=%.1fms",
        body.question,
        retrieval_method,
        len(chunks),
        latency_ms,
    )

    return QueryResponse(
        chunks=chunks,
        retrieval_method=retrieval_method,
        latency_ms=latency_ms,
    )
