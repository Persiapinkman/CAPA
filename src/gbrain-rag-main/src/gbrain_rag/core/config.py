from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Runtime settings.

    Environment variables use the GBRAIN_RAG_ prefix so this service can live
    next to swift-rag without stealing its configuration.
    """

    model_config = SettingsConfigDict(
        env_prefix="GBRAIN_RAG_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "GBrain RAG"
    PROJECT_DESCRIPTION: str = "gbrain-inspired hybrid RAG service for swift-rag corpus"
    PROJECT_VERSION: str = "0.1.0"

    HOST: str = "0.0.0.0"
    PORT: int = 6061
    RELOAD: bool = False

    DATA_SOURCE_DIR: Path = Field(default=PROJECT_ROOT / "data_source")
    INDEX_DB_PATH: Path = Field(default=PROJECT_ROOT / "data" / "index" / "gbrain.sqlite3")
    ARTIFACTS_DIR: Path = Field(default=PROJECT_ROOT / "data" / "artifacts")

    EMBEDDING_BACKEND: str = "sentence-transformers"
    EMBEDDING_MODEL: str = "bge_m3"
    EMBEDDING_MODELS: list[str] = ["bge_m3", "EvoQwen2.5-VL-Retriever-3B-v1"]
    EMBEDDING_DEVICE: str = "cuda"
    HASHING_DIM: int = 384
    ALLOW_HASHING_FALLBACK: bool = True

    BGE_M3_MODEL_PATH: Path = Field(default=PROJECT_ROOT / "bge-m3")
    EVOQWEN_3B_MODEL_PATH: Path = Field(
        default=PROJECT_ROOT / "EvoQwen2.5-VL-Retriever-3B-v1"
    )

    DEFAULT_RETRIEVAL_METHOD: str = "hybrid"
    DEFAULT_TOP_K: int = 8
    DEFAULT_CANDIDATE_LIMIT: int = 80
    MIN_HEALTHY_SOURCE_COUNTS: dict[str, int] = {
        "document": 100,
        "table": 100,
        "adela": 100,
    }
    RRF_K: int = 60
    VECTOR_WEIGHT: float = 1.0
    KEYWORD_WEIGHT: float = 1.0
    GRAPH_WEIGHT: float = 0.28
    STRUCTURED_WEIGHT: float = 0.45

    LLM_MODEL: str = "Qwen3.5-4B"
    LLM_BASE_URL: str = "http://10.111.32.253:8000/v1"
    LLM_API_KEY: str | None = "token.sdc@2026"
    LLM_MAX_TOKENS: int = 2048
    LLM_TEMPERATURE: float | None = 0.1
    LLM_TOP_P: float | None = 0.5
    LLM_SEED: int | None = None
    DISABLE_LLM: bool = False
    ENABLE_LLM_QUERY_EXPANSION: bool = True
    LLM_QUERY_EXPANSION_MAX_TERMS: int = 8

    PDF_TABLE_MIN_ROWS: int = 2
    PDF_TABLE_MIN_COLS: int = 2
    CHUNK_SIZE: int = 900
    CHUNK_OVERLAP: int = 120

    TABLE_SEARCHABLE_FIELDS: list[str] = [
        "target_name",
        "algorithm_type",
        "algorithm_name",
        "application_scene",
        "owner",
        "model_name",
        "supported_device",
        "recommended_config",
        "last_updated",
        "last_updated_month",
        "oid",
    ]
    TABLE_RETURN_FIELDS: list[str] = [
        "target_name",
        "algorithm_type",
        "algorithm_name",
        "application_scene",
        "owner",
        "model_name",
        "supported_device",
        "recommended_config",
        "last_updated",
        "ones_release_link",
        "oid",
    ]
    ADELA_SEARCHABLE_FIELDS: list[str] = [
        "model_name",
        "name",
        "label_list",
        "labels",
        "type",
        "platform",
        "status",
        "did",
        "rid",
        "version",
        "version_train_date",
        "source_file",
    ]
    ADELA_RETURN_FIELDS: list[str] = [
        "model_name",
        "name",
        "label_list",
        "labels",
        "type",
        "platform",
        "status",
        "did",
        "rid",
        "version",
        "version_train_date",
        "source_file",
        "model_info",
        "benchmark_info",
    ]
    ADELA_DEPLOYMENT_URL_TEMPLATE: str = (
        "http://adela.sensetime.com/mainpage/project/3/models?deployment_id={did}"
    )

    def model_path_for(self, model_name: str) -> Path:
        aliases: dict[str, Path] = {
            "bge_m3": self.BGE_M3_MODEL_PATH,
            "bge-m3": self.BGE_M3_MODEL_PATH,
            "EvoQwen2.5-VL-Retriever-3B-v1": self.EVOQWEN_3B_MODEL_PATH,
            "evoqwen_3b": self.EVOQWEN_3B_MODEL_PATH,
        }
        return aliases.get(model_name, Path(model_name))


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    if not settings.DATA_SOURCE_DIR.is_absolute():
        settings.DATA_SOURCE_DIR = (PROJECT_ROOT / settings.DATA_SOURCE_DIR).resolve()
    if not settings.INDEX_DB_PATH.is_absolute():
        settings.INDEX_DB_PATH = (PROJECT_ROOT / settings.INDEX_DB_PATH).resolve()
    if not settings.ARTIFACTS_DIR.is_absolute():
        settings.ARTIFACTS_DIR = (PROJECT_ROOT / settings.ARTIFACTS_DIR).resolve()
    if not settings.BGE_M3_MODEL_PATH.is_absolute():
        settings.BGE_M3_MODEL_PATH = (PROJECT_ROOT / settings.BGE_M3_MODEL_PATH).resolve()
    if not settings.EVOQWEN_3B_MODEL_PATH.is_absolute():
        settings.EVOQWEN_3B_MODEL_PATH = (PROJECT_ROOT / settings.EVOQWEN_3B_MODEL_PATH).resolve()
    settings.INDEX_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    settings.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return settings


def pydantic_dump(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return dict(model)
