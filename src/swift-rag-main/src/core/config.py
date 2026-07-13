from pydantic_settings import BaseSettings
from typing import List, Dict, Any, Optional
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # API 配置
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "RAG API Service"
    PROJECT_DESCRIPTION: str = "RAG服务API接口"
    PROJECT_VERSION: str = "1.0.0"

    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 6060
    RELOAD: bool = False

    EMBEDDING_ARTIFACTS_DIR: str = str(
        PROJECT_ROOT / "data_source" / "embedding_artifacts"
    )
    DOCUMENT_EMBEDDING_ARTIFACTS_DIR: str = str(
        PROJECT_ROOT / "data_source" / "embedding_artifacts" / "documents"
    )

    # 向量数据库配置（测试接口用）
    TEST_VECTOR_DB_URI: str = str(
        PROJECT_ROOT / "data_source" / "embedding_artifacts" / "documents" / "milvus_test.db"
    )
    TEST_COLLECTION_NAME: str = "llamacollection"

    # data_source 离线入库后的默认检索库（主默认库）
    DATA_SOURCE_VECTOR_DB_URI: str = str(
        PROJECT_ROOT / "data_source" / "embedding_artifacts" / "documents" / "milvus_data_source_evoqwen_3b.db"
    )
    DATA_SOURCE_COLLECTION_NAME: str = "llamacollection"
    DATA_SOURCE_BGE_VECTOR_DB_URI: str = str(
        PROJECT_ROOT / "data_source" / "embedding_artifacts" / "documents" / "milvus_data_source_bge.db"
    )
    DATA_SOURCE_EVOQWEN_VECTOR_DB_URI: str = str(
        PROJECT_ROOT / "data_source" / "embedding_artifacts" / "documents" / "milvus_data_source_evoqwen_3b.db"
    )
    DATA_SOURCE_EVOQWEN_7B_VECTOR_DB_URI: str = str(
        PROJECT_ROOT / "data_source" / "embedding_artifacts" / "documents" / "milvus_data_source_evoqwen_7b.db"
    )
    DATA_SOURCE_VECTOR_STORE_CONFIGS: Dict[str, Dict[str, str]] = {
        "bge_m3": {
            "uri": str(
                PROJECT_ROOT / "data_source" / "embedding_artifacts" / "documents" / "milvus_data_source_bge.db"
            ),
            "collection_name": "llamacollection",
        },
        "EvoQwen2.5-VL-Retriever-3B-v1": {
            "uri": str(
                PROJECT_ROOT / "data_source" / "embedding_artifacts" / "documents" / "milvus_data_source_evoqwen_3b.db"
            ),
            "collection_name": "llamacollection",
        },
        "EvoQwen2.5-VL-Retriever-7B-v1": {
            "uri": str(
                PROJECT_ROOT / "data_source" / "embedding_artifacts" / "documents" / "milvus_data_source_evoqwen_7b.db"
            ),
            "collection_name": "llamacollection",
        },
    }
    PDF_REFERENCE_MAPPING_PATH: str = str(PROJECT_ROOT / "data_source" / "pdf_reference_links.csv")
    TABLE_DATA_JSONL_PATH: str = str(
        PROJECT_ROOT / "data_source" / "tables" / "model_release_records.jsonl"
    )
    ADELA_DATA_DIR: str = str(PROJECT_ROOT / "data_source" / "adela" / "data")
    ADELA_DATA_JSONL_PATH: str = str(
        PROJECT_ROOT / "data_source" / "adela" / "adela_release_records.jsonl"
    )
    ADELA_RELEASE_RECORDS_CSV_PATH: str = str(
        PROJECT_ROOT / "data_source" / "adela" / "adela_release_records.csv"
    )
    ADELA_DEPLOYMENT_URL_TEMPLATE: str = (
        "http://adela.sensetime.com/mainpage/project/3/models?deployment_id={did}"
    )
    PUBLIC_CLOUD_MODELS_API_URL: str = "http://10.111.32.253:8000/v1/models"
    PUBLIC_CLOUD_MODELS_API_TOKEN: str = "token.sdc@2026"
    DEFAULT_PUBLIC_CLOUD_RETRIEVAL_METHOD: str = "keyword"
    PUBLIC_CLOUD_TOP_K: int = 20
    TABLE_SEARCHABLE_FIELDS: List[str] = [
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
    ]
    TABLE_RETURN_FIELDS: List[str] = [
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
    ADELA_SEARCHABLE_FIELDS: List[str] = [
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
    ADELA_RETURN_FIELDS: List[str] = [
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
        "version_major",
        "version_minor",
        "version_patch",
        "version_train_date",
        "returncode",
        "source_file",
        "command",
        "stderr",
        "model_info",
        "benchmark_info",
    ]

    # 本地向量数据库默认配置
    LOCAL_VECTOR_DB_URI: str = str(
        PROJECT_ROOT / "data_source" / "embedding_artifacts" / "documents" / "milvus_test_temp.db"
    )
    LOCAL_COLLECTION_NAME: str = "llamacollection"
    OVERWRITE_VECTOR_STORE: bool = True

    # 初始化时加载的Embedding模型配置
    INIT_EMBEDDING_MODEL_CONFIGS: List[Dict[str, Any]] = [
        {
            "model_path": str(PROJECT_ROOT / "bge-m3"),
            "model_name": "bge_m3",
            "batchsize": 16,
            "device": "cuda:5",
        },
        {
            "model_path": str(PROJECT_ROOT / "EvoQwen2.5-VL-Retriever-3B-v1"),
            "model_name": "EvoQwen2.5-VL-Retriever-3B-v1",
            "batchsize": 8,
            "device": "cuda:6",
        },
    ]
    # 默认使用的Embedding模型
    DEFAULT_EMBEDDING_MODELS: List[str] = ["bge_m3", "EvoQwen2.5-VL-Retriever-3B-v1"]
    DEFAULT_SINGLE_EMBEDDING_MODEL: List[str] = ["EvoQwen2.5-VL-Retriever-3B-v1"]
    DEFAULT_STRUCTURED_EMBEDDING_MODEL: List[str] = ["bge_m3"]

    # 默认 LLM 配置
    DEFAULT_LLM_MODEL: str = "Qwen3.5-4B"
    DEFAULT_LLM_BASE_URL: str = "http://10.111.32.253:8000/v1"
    DEFAULT_LLM_API_KEY: str = "token.sdc@2026"
    DEFAULT_LLM_MAX_TOKENS: int = 1024 * 2
    DEFAULT_LLM_TEMPERATURE: Optional[float] = 0.1
    DEFAULT_LLM_TOP_P: Optional[float] = 0.5
    DEFAULT_LLM_SEED: Optional[int] = None

    # 检索配置
    DEFAULT_RETRIEVAL_METHOD: str = "hybrid"
    DEFAULT_TABLE_RETRIEVAL_METHOD: str = "hybrid"
    DEFAULT_ADELA_RETRIEVAL_METHOD: str = "hybrid"
    BM25_K1: float = 1.5
    BM25_B: float = 0.75
    BM25_CANDIDATE_LIMIT: int = 16384

    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE_PATH: Optional[str] = str(PROJECT_ROOT / "logs" / "rag_service.log")
    LOG_TO_FILE: bool = True
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    RAG_CHAT_TIMING_LOG_PATH: str = str(PROJECT_ROOT / "logs" / "rag_chat_timing.jsonl")
    ENABLE_RAG_CHAT_TIMING_LOG: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """获取应用配置，使用lru_cache避免重复加载"""
    return Settings()


def get_default_llm_config() -> Dict[str, Any]:
    """返回服务统一维护的默认 LLM 参数。"""
    settings = get_settings()
    return {
        "model": settings.DEFAULT_LLM_MODEL,
        "base_url": settings.DEFAULT_LLM_BASE_URL,
        "api_key": settings.DEFAULT_LLM_API_KEY,
        "max_tokens": settings.DEFAULT_LLM_MAX_TOKENS,
        "temperature": settings.DEFAULT_LLM_TEMPERATURE,
        "top_p": settings.DEFAULT_LLM_TOP_P,
        "seed": settings.DEFAULT_LLM_SEED,
    }
