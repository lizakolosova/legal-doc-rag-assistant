import json
import logging
import time
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.generation.llm_client import generate_answer
from app.generation.prompt_builder import build_messages, extract_citations
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
async def query_documents(
    body: QueryRequest,
    session: AsyncSession = Depends(_get_session),
) -> QueryResponse:
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

    messages = build_messages(body.question, chunks)
    answer = await generate_answer(messages, stream=False)
    citations = extract_citations(chunks)

    latency_ms = (time.monotonic() - start) * 1000

    logger.info(
        "query question='%.50s' method=%s chunks=%d elapsed=%.1fms",
        body.question,
        retrieval_method,
        len(chunks),
        latency_ms,
    )

    return QueryResponse(
        answer=answer,
        citations=citations,
        chunks=chunks,
        retrieval_method=retrieval_method,
        latency_ms=latency_ms,
    )


@router.post("/stream")
async def stream_query_documents(
    body: QueryRequest,
    session: AsyncSession = Depends(_get_session),
) -> StreamingResponse:
    k = body.top_k

    results = await hybrid_search(
        body.question,
        session,
        top_k=k * 2,
        document_ids=body.document_ids,
    )

    if body.use_reranker:
        chunks = rerank(body.question, results, top_k=k)
    else:
        chunks = results[:k]

    messages = build_messages(body.question, chunks)
    citations = extract_citations(chunks)
    answer_stream = await generate_answer(messages, stream=True)

    async def event_generator() -> AsyncGenerator[str, None]:
        async for delta in answer_stream:
            yield json.dumps({"delta": delta}) + "\n"
        yield json.dumps({"done": True, "citations": [c.model_dump() for c in citations]}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")