from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    chroma_host: str = "chromadb"
    chroma_port: int = 8000

    postgres_url: str

    chunk_size: int = 512
    chunk_overlap: int = 50

    top_k: int = 10
    retrieval_top_k: int = 20
    rerank_top_k: int = 5

    max_file_size_mb: int = 50
    max_pages: int = 200


settings = Settings()