# Technical Architecture

## Module Responsibilities

### `app/config.py`
Single source of truth for all runtime configuration. Uses Pydantic `BaseSettings`, which reads from environment variables (with `.env` file fallback). Every tunable value — model names, chunk sizes, top-k counts, file limits — lives here. No module outside `config.py` calls `os.getenv` directly. This means the entire system can be reconfigured for a different deployment environment (staging, cloud) purely through environment variables with zero code changes.

### `app/exceptions.py`
Custom exception hierarchy rooted at `AppError`. Each subclass carries structured data (filename, limit type, reason) rather than just a message string, which allows FastAPI exception handlers in `main.py` to return precise HTTP responses. All `except` clauses catch specific exception types — bare `except:` and generic `except Exception` are used only at pipeline entry points where the intent is to mark a document as failed and re-raise.

### `app/models/schemas.py`
Pydantic models for every cross-module data contract. The pipeline stages communicate exclusively through these typed models: `ParsedSection` (parser output), `TextChunk` (chunker output), `EmbeddedChunk` (embedder output), `RetrievedChunk` (retrieval output), `QueryRequest`/`QueryResponse` (API boundary). No raw dicts are passed between modules.

### `app/models/database.py`
SQLAlchemy ORM definitions and async session factory. Three tables: `documents` (document metadata and ingestion status), `chunks` (text and ChromaDB IDs for BM25 retrieval), `evaluation_runs` (RAGAS metric history). All database access goes through session objects obtained from `get_async_session()` — there is no scattered SQL in route handlers.

### `app/ingestion/parser.py`
Extracts text from PDF and DOCX files into `ParsedSection` objects. For PDFs, each page becomes one section (using PyMuPDF). For DOCX files, paragraphs are grouped under their nearest heading using Word style names. Scanned image pages are detected by checking for images with no accompanying text and are rejected with a user-facing `DocumentParseError`. Size and page count limits are enforced before any parsing begins.

### `app/ingestion/chunker.py`
Splits `ParsedSection` objects into fixed-size `TextChunk` objects using LangChain's `RecursiveCharacterTextSplitter`. Chunk size and overlap come from `settings`. Each chunk inherits metadata (document ID, source file, page number, section header) from its parent section, preserving traceability through the pipeline. Empty sections are skipped.

### `app/ingestion/embedder.py`
Orchestrates the full ingestion pipeline: parse → chunk → embed → store. Embedding uses the local `all-MiniLM-L6-v2` sentence-transformers model (384-dimensional vectors, no API cost). The model is loaded once as a module-level singleton. ChromaDB storage uses `upsert` with deterministic IDs (`{document_id}_{chunk_index}`) so re-ingestion is idempotent. On any failure, the Postgres document record is updated to `failed` status with the error reason, and all partial writes are rolled back.

### `app/retrieval/vector_search.py`
Embeds the query using the same local `all-MiniLM-L6-v2` model used at ingestion time, then queries ChromaDB for the nearest neighbours. Optionally filters by document ID(s) using ChromaDB's metadata filter syntax. Returns `RetrievedChunk` objects with cosine similarity scores (converted from distances). The model singleton is shared with `embedder.py` through module-level lazy initialisation.

### `app/retrieval/bm25_search.py`
Builds a BM25Okapi index over all chunks loaded from Postgres, then scores the tokenised query against it. Scores are normalised to [0, 1] by dividing by the maximum score. Document-level filtering is applied post-scoring by zeroing out scores for excluded documents. The index is rebuilt on each request (acceptable at this scale; a cache would be the first production optimisation).

### `app/retrieval/hybrid.py`
Runs vector search and BM25 in parallel using `asyncio.gather`, then merges the two ranked lists with Reciprocal Rank Fusion. RRF scores each chunk as `sum(1 / (k + rank))` across both lists, where k=60 follows the original paper. Chunks appearing in both lists accumulate score from both contributions. The merged list is sorted by RRF score and truncated to `top_k`.

### `app/retrieval/reranker.py`
Applies a cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to re-score the hybrid search results. The cross-encoder sees each (query, chunk) pair jointly, enabling precise relevance judgements that bi-encoder embeddings cannot capture. The model is loaded once and cached as a module-level singleton. The model name is read from `settings.reranker_model`.

### `app/generation/prompt_builder.py`
Builds the message list for the LLM. Each chunk is formatted as a numbered context block `[N] Source: file, Page P\ntext\n---`. The system prompt instructs the model to cite every claim using `[N]` markers and to refuse to answer if the context is insufficient. Long chunks are truncated to 1,500 characters before inclusion. `extract_citations` maps the same chunks to `Citation` objects whose indices align with the markers in the generated answer.

### `app/generation/llm_client.py`
The only module that imports the Google Gemini SDK (`google-generativeai`). `generate_answer` dispatches to a blocking (full-response) path or a streaming (token-delta) path. The blocking path includes retry logic for `ResourceExhausted` (quota exceeded) with exponential delays (1s, 2s, 4s). The streaming path yields token deltas as an `AsyncGenerator`. Swapping the LLM provider requires changes only in this file.

### `app/evaluation/eval_dataset.py`
Loads the golden Q&A dataset from a JSON file. Each record has a question, expected answer, and a list of relevant source files. The default dataset path is `eval_data/legal_qa_golden.json` relative to the project root. A custom path can be passed for testing.

### `app/evaluation/metrics.py`
Two metric families: heuristic retrieval metrics (context precision and recall, computed by comparing retrieved source files against the golden relevant-sources list) and RAGAS generation metrics (faithfulness and answer relevancy, computed by calling the RAGAS library which uses an LLM as judge). RAGAS failures are caught and returned as `None` rather than aborting the evaluation run.

### `app/evaluation/run_eval.py`
Orchestrates a full evaluation run: loads the golden dataset, runs the retrieval and generation pipeline for each question, computes both metric families, aggregates a summary, and persists an `EvaluationRun` row to Postgres. Individual question failures are logged and skipped rather than aborting the run.

### `app/api/routes_documents.py`, `routes_query.py`, `routes_eval.py`
FastAPI routers that define the HTTP API. Route handlers are thin: they validate input via Pydantic, delegate to pipeline functions, and return Pydantic response models. No business logic lives in route handlers. Each router has its own `_get_session` dependency that yields a database session scoped to the request.

---

## Data Flow: Ingestion Pipeline

```
POST /api/documents  (multipart file upload)
  │
  ├─ Save upload to temp file
  │
  ├─ parse_document(file_path, document_id)
  │     ├─ Enforce size/page limits (config: max_file_size_mb, max_pages)
  │     ├─ PDF → parse_pdf()  →  [ParsedSection per page]
  │     └─ DOCX → parse_docx()  →  [ParsedSection per heading group]
  │
  ├─ chunk_sections(sections)
  │     └─ RecursiveCharacterTextSplitter → [TextChunk] (config: chunk_size, chunk_overlap)
  │
  ├─ embed_chunks(chunks)
  │     └─ sentence-transformers all-MiniLM-L6-v2 (local) → [EmbeddedChunk]
  │
  ├─ store_in_chroma(embedded)
  │     └─ ChromaDB upsert with deterministic IDs
  │
  ├─ Persist Chunk rows to Postgres
  ├─ Update Document status → "ready"
  └─ Return {document_id, status: "ready"}

On any failure:
  ├─ Rollback Postgres session
  ├─ Update Document status → "failed" with error_message
  └─ Re-raise exception (FastAPI exception handler returns appropriate HTTP error)
```

---

## Data Flow: Query Pipeline

```
POST /api/query  {question, document_ids?, top_k, use_reranker}
  │
  ├─ hybrid_search(question, session, top_k=top_k*2, document_ids)
  │     ├─ [parallel] vector_search()
  │     │     ├─ Embed question (sentence-transformers local model)
  │     │     └─ ChromaDB query with optional document_id filter
  │     ├─ [parallel] bm25_search()
  │     │     ├─ Load all chunks from Postgres + build BM25 index
  │     │     └─ Score tokenised query; zero-out filtered documents
  │     └─ reciprocal_rank_fusion(vector_results, bm25_results) → merged[:top_k*2]
  │
  ├─ (if use_reranker) rerank(question, merged, top_k=top_k)
  │     └─ CrossEncoder.predict([(question, chunk.text), ...]) → top_k chunks
  │
  ├─ build_messages(question, chunks) → [system, user] messages
  ├─ generate_answer(messages) → answer string
  ├─ extract_citations(chunks) → [Citation]
  │
  └─ Return {answer, citations, chunks, retrieval_method, latency_ms}

POST /api/query/stream  →  same pipeline, answer yielded as NDJSON token deltas,
                           citations emitted in final {done: true} frame
```

---

## Key Design Constraints

**No LangChain except the text splitter.** The retrieval pipeline (BM25, vector search, RRF, reranking) is implemented from scratch. This makes the system comprehensible as a portfolio piece — a reviewer can trace exactly what happens to a query without needing to understand a framework's internals.

**Pydantic models at every module boundary.** `ParsedSection → TextChunk → EmbeddedChunk → RetrievedChunk` forms an explicit type contract across pipeline stages. The API request and response bodies are also Pydantic models. This prevents accidental dict-key errors and makes the data flow self-documenting.

**Async everywhere for I/O.** All database queries, Gemini API calls, and ChromaDB operations are async. The reranker (which uses a local PyTorch model) is sync but runs in milliseconds and is not an I/O operation. Mixing sync and async would require `asyncio.to_thread` wrappers; the architecture avoids this by keeping the cross-encoder as a fast in-process call.

**Repository pattern via session objects.** All SQL lives in `models/database.py` ORM definitions or in explicit `session.execute(select(...))` calls inside route handlers. There are no string-interpolated SQL queries and no scattered `get_async_engine()` calls scattered through the codebase.

**Cloud-ready configuration.** All hostnames, ports, credentials, and tunable parameters come from environment variables via Pydantic Settings. No `localhost` references appear in service-to-service communication. `POSTGRES_URL` can be pointed at RDS; `CHROMA_HOST` can be pointed at a managed ChromaDB instance; `GEMINI_API_KEY` is never hardcoded. The docker-compose file is the only deployment artifact needed for local development.
