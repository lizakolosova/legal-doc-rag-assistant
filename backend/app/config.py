"""Application configuration via Pydantic Settings.

All tunable values are loaded from environment variables (or a .env file).
Import the singleton `settings` object rather than instantiating Settings directly.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the Legal Doc RAG Assistant.

    Values are read from environment variables first, then from a .env file
    in the backend working directory, then from the defaults below.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- OpenAI ---
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # --- ChromaDB ---
    chroma_host: str = "chromadb"
    chroma_port: int = 8000

    # --- PostgreSQL ---
    postgres_url: str  # e.g. postgresql+asyncpg://user:pass@host:5432/dbname

    # --- Chunking ---
    chunk_size: int = 512
    chunk_overlap: int = 50

    # --- Retrieval ---
    top_k: int = 10          # candidates returned by each retriever before fusion
    retrieval_top_k: int = 20  # total candidates passed into reranker
    rerank_top_k: int = 5    # chunks passed to the LLM after reranking

    # --- Document limits ---
    max_file_size_mb: int = 50
    max_pages: int = 200


settings = Settings()  # type: ignore[call-arg]  # populated from env at import time
