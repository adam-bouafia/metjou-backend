"""Settings, read from the environment (prefix METJOU_) or a .env file."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="METJOU_", env_file=".env")

    database_url: str = "postgresql://metjou:metjou@localhost:5432/metjou"

    # Local ONNX embedding model, no external API. The e5 family needs the
    # "query: " and "passage: " prefixes, see embeddings.py.
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_source: str = "Xenova/multilingual-e5-small"
    embedding_file: str = "onnx/model_quantized.onnx"
    embedding_dim: int = 384
    model_cache_dir: str = "models"

    # How many passages /v1/ask returns.
    answer_count: int = 3

    # Below this cosine similarity the best passage is flagged as a weak
    # match, and the app points to a helpline instead. Tune with eval/.
    min_similarity: float = 0.80


@lru_cache
def get_settings() -> Settings:
    return Settings()
