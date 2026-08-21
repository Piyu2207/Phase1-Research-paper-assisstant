from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # Gemini Developer API key. Prefer GEMINI_API_KEY; keep GOOGLE_API_KEY
    # as a backwards-compatible alias for existing local .env files.
    gemini_api_key: str | None = None
    google_api_key: str | None = None

    # Environment variables take precedence over these defaults.
    generation_model: str = "gemini-3.6-flash"
    router_model: str = "gemini-3.6-flash"

    chunk_size: int = 3000
    chunk_overlap: int = 400
    retrieval_k: int = 10
    rerank_top_k: int = 5

    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    vectorstore_dir: Path = ROOT_DIR / "data" / "vectorstore"
    evaluation_dir: Path = ROOT_DIR / "data" / "evaluation"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    settings.evaluation_dir.mkdir(parents=True, exist_ok=True)
    return settings
