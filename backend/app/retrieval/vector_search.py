import logging
import time

import openai

from app.config import settings
from app.exceptions import RetrievalError
from app.models.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "documents"


async def vector_search(
    query: str,
    top_k: int | None = None,
    document_ids: list[str] | None = None,
) -> list[RetrievedChunk]:
    k = top_k if top_k is not None else settings.retrieval_top_k
    start = time.monotonic()

    client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
    try:
        response = await client.embeddings.create(
            model=settings.embedding_model,
            input=[query],
        )
    except openai.OpenAIError as exc:
        raise RetrievalError(str(exc)) from exc

    query_embedding: list[float] = response.data[0].embedding

    where_filter: dict | None = None
    if document_ids:
        if len(document_ids) == 1:
            where_filter = {"document_id": {"$eq": document_ids[0]}}
        else:
            where_filter = {"document_id": {"$in": document_ids}}

    import chromadb

    try:
        chroma_client = await chromadb.AsyncHttpClient(
            host=settings.chroma_host, port=settings.chroma_port
        )
        collection = await chroma_client.get_or_create_collection(_COLLECTION_NAME)

        query_kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter is not None:
            query_kwargs["where"] = where_filter

        results = await collection.query(**query_kwargs)
    except Exception as exc:
        raise RetrievalError(str(exc)) from exc

    ids: list[str] = results.get("ids", [[]])[0]
    docs: list[str] = results.get("documents", [[]])[0]
    metas: list[dict] = results.get("metadatas", [[]])[0]
    dists: list[float] = results.get("distances", [[]])[0]

    chunks = [
        RetrievedChunk(
            chunk_id=chunk_id,
            document_id=meta.get("document_id", ""),
            text=text,
            score=1.0 - dist,
            source_file=meta.get("source_file", ""),
            page_number=int(meta.get("page_number", -1)),
            section_header=meta.get("section_header") or None,
        )
        for chunk_id, text, meta, dist in zip(ids, docs, metas, dists)
    ]

    chunks.sort(key=lambda c: c.score, reverse=True)

    elapsed_ms = int((time.monotonic() - start) * 1000)
    filter_desc = f"document_ids={document_ids}" if document_ids else "no filter"
    logger.info(
        "vector_search query='%.50s' top_k=%d filter=%s results=%d elapsed=%dms",
        query,
        k,
        filter_desc,
        len(chunks),
        elapsed_ms,
    )

    return chunks