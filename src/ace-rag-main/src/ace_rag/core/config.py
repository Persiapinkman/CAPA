from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Runtime settings for the ACE sidecar.

    Environment variables use ACE_RAG_ to avoid clashing with gbrain-rag.
    """

    model_config = SettingsConfigDict(
        env_prefix="ACE_RAG_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "ACE RAG"
    PROJECT_DESCRIPTION: str = "ACE Playbook sidecar for gbrain-rag"
    PROJECT_VERSION: str = "0.1.0"

    HOST: str = "0.0.0.0"
    PORT: int = 6062
    RELOAD: bool = False

    V2_BASE_URL: str = "http://127.0.0.1:6061/api/v1/rag"
    V2_TIMEOUT_SECONDS: float = 60.0

    PLAYBOOK_DB_PATH: Path = Field(default=PROJECT_ROOT / "data" / "playbook.sqlite3")
    PLAYBOOK_SEED_PATH: Path = Field(default=PROJECT_ROOT / "data_source" / "playbook_seed.jsonl")
    PLAYBOOK_TOP_K: int = 8
    AUTO_IMPORT_SEED: bool = True
    ENABLE_PLAYBOOK_AUTO_ORGANIZE: bool = True
    PLAYBOOK_ORGANIZE_DELTA_THRESHOLD: int = 5
    PLAYBOOK_ORGANIZE_MAX_ITEMS: int = 120

    DEFAULT_RETRIEVAL_METHOD: str = "hybrid"
    DEFAULT_TOP_K: int = 12
    DEFAULT_CANDIDATE_LIMIT: int = 80
    ENABLE_V2_LLM_QUERY_EXPANSION: bool = True

    LLM_MODEL: str = "Qwen3.5-4B"
    LLM_BASE_URL: str | None = "http://10.111.32.253:8000/v1"
    LLM_API_KEY: str | None = None
    LLM_MAX_TOKENS: int = 2048
    LLM_TEMPERATURE: float | None = 0.1
    LLM_TOP_P: float | None = 0.5
    LLM_SEED: int | None = None
    DISABLE_LLM: bool = False


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    if not settings.PLAYBOOK_DB_PATH.is_absolute():
        settings.PLAYBOOK_DB_PATH = (PROJECT_ROOT / settings.PLAYBOOK_DB_PATH).resolve()
    if not settings.PLAYBOOK_SEED_PATH.is_absolute():
        settings.PLAYBOOK_SEED_PATH = (PROJECT_ROOT / settings.PLAYBOOK_SEED_PATH).resolve()
    settings.PLAYBOOK_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return settings


def pydantic_dump(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return dict(model)
