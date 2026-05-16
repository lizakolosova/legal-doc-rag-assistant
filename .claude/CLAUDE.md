# Legal Doc RAG Assistant

## What this is
A production-grade RAG system: upload legal PDFs/DOCX → ask questions → get answers with source citations. This is a **portfolio project** — code quality, documentation, and architecture matter as much as functionality.

## Tech stack
- **Backend**: FastAPI (Python 3.11+), async everywhere
- **LLM**: Google Gemini 2.5 Flash Lite (free tier)
- **Embeddings**: sentence-transformers all-MiniLM-L6-v2 (local, 384 dims)
- **Vector DB**: ChromaDB
- **Hybrid search**: BM25 (rank_bm25) + vector search, merged with Reciprocal Rank Fusion
- **Reranker**: cross-encoder/ms-marco-MiniLM-L-6-v2 (sentence-transformers)
- **Document parsing**: PyMuPDF (PDF), python-docx (DOCX)
- **Chunking**: LangChain RecursiveCharacterTextSplitter (512 tokens, 50 overlap)
- **Evaluation**: RAGAS framework
- **Frontend**: React + TypeScript (Vite)
- **Database**: PostgreSQL (metadata, chat history, eval results)
- **Containerization**: Docker + docker-compose

## Project structure
```
backend/app/
├── main.py              # FastAPI entry point
├── config.py            # Pydantic Settings (env vars)
├── models/              # Pydantic schemas + SQLAlchemy models
├── ingestion/           # parser.py, chunker.py, embedder.py
├── retrieval/           # vector_search.py, bm25_search.py, hybrid.py, reranker.py
├── generation/          # prompt_builder.py, llm_client.py
├── evaluation/          # eval_dataset.py, metrics.py, run_eval.py
└── api/                 # routes_documents.py, routes_query.py, routes_eval.py
frontend/src/
├── components/          # DocumentUpload, ChatInterface, AnswerCard, EvalDashboard
├── hooks/               # useApi.ts
└── App.tsx
```

## Code standards — IMPORTANT

### Python
- Type hints on ALL function signatures (params + return)
- Pydantic models for ALL request/response schemas — never pass raw dicts across module boundaries
- Docstrings on every public function: one-line summary, then Args/Returns/Raises
- Use `logging` module, never `print()`. Logger per module: `logger = logging.getLogger(__name__)`
- Async functions for all I/O operations (database, API calls, file reads)
- Environment variables via Pydantic Settings (`config.py`), never hardcoded strings or `os.getenv` scattered in code
- Imports: stdlib → third-party → local, separated by blank lines
- Error handling: custom exception classes in `exceptions.py`, FastAPI exception handlers, never bare `except:`

### TypeScript / React
- Functional components only, with explicit prop types (interfaces, not `any`)
- Custom hooks for API calls (`useApi.ts`)
- No inline styles — use CSS modules or Tailwind utility classes

### Testing
- Run individual tests with: `pytest tests/test_<module>.py::test_<name> -v`
- Run all tests: `pytest tests/ -v`
- Tests use fixtures from `conftest.py` — check there before creating test data manually

### Git
- Conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`
- One logical change per commit — never bundle unrelated changes
- Never commit `.env`, API keys, or secrets

## Architecture principles
- Each pipeline (ingestion / retrieval / generation / evaluation) is a separate module with its own responsibility
- Modules communicate through Pydantic models, not raw dicts
- Database operations go through repository functions in `models/database.py`, not scattered SQL
- The retrieval pipeline is composable: vector → hybrid → reranked, each step independent and testable
- LLM calls are isolated in `llm_client.py` so the provider can be swapped without touching business logic

## How to verify changes
- After editing Python: `cd backend && python -m py_compile app/<file>.py`
- After editing tests: `cd backend && pytest tests/test_<relevant>.py -v`
- After editing frontend: `cd frontend && npx tsc --noEmit`
- Before committing: run the relevant test file, not the entire suite

## Common mistakes to avoid
- Do NOT use LangChain for anything except the text splitter — we're building the RAG pipeline from scratch to demonstrate understanding
- Do NOT use `datetime.now()` without timezone — always `datetime.now(UTC)`
- Do NOT return raw ChromaDB results to the API — always map to Pydantic response models
- Do NOT hardcode chunk_size or top_k — these come from `config.py`
- Do NOT mix sync and async — if a function calls any async operation, it must be async itself
- When compacting, preserve: current task, list of modified files, any failing tests, and the architecture decisions made so far
