from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from gbrain_rag.core.config import get_settings

settings = get_settings()


class LLMConfig(BaseModel):
    model: str | None = Field(default=settings.LLM_MODEL)
    base_url: str | None = Field(default=settings.LLM_BASE_URL)
    api_key: str | None = Field(default=settings.LLM_API_KEY)
    max_tokens: int | None = Field(default=settings.LLM_MAX_TOKENS)
    temperature: float | None = Field(default=settings.LLM_TEMPERATURE)
    top_p: float | None = Field(default=settings.LLM_TOP_P)
    seed: int | None = Field(default=settings.LLM_SEED)


class SourceConfig(BaseModel):
    enabled: bool = True
    top_k: int | None = None
    retrieval_method: Literal["vector", "keyword", "bm25", "hybrid"] | None = None
    similarity_threshold: float | None = None


class RetrieveRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    query: str
    retrieval_method: Literal["vector", "keyword", "bm25", "hybrid"] = Field(
        default=settings.DEFAULT_RETRIEVAL_METHOD
    )
    top_k: int = Field(default=settings.DEFAULT_TOP_K, ge=1, le=100)
    candidate_limit: int = Field(default=settings.DEFAULT_CANDIDATE_LIMIT, ge=1, le=1000)
    similarity_threshold: float | None = None
    sources: list[Literal["document", "table", "adela"]] | None = None
    route_with_llm: bool = False
    expand_query_with_llm: bool = Field(default=settings.ENABLE_LLM_QUERY_EXPANSION)
    query_expansion_terms: list[str] = Field(default_factory=list)
    include_full_documents: bool = Field(
        default=False,
        description="是否返回命中 chunk 所属文档的完整索引内容；多个命中归属同一文档时只返回一份。",
    )
    embedding_model: str | None = Field(default=settings.EMBEDDING_MODEL)
    embedding_models: list[str] | None = Field(default=None)
    embedding_backend: str | None = Field(default=None)
    document: SourceConfig = Field(default_factory=SourceConfig)
    table: SourceConfig = Field(default_factory=SourceConfig)
    adela: SourceConfig = Field(default_factory=SourceConfig)


class QueryRequest(RetrieveRequest):
    llm_config: LLMConfig | None = None
    stream: bool = Field(default=False, description="是否开启流式回答（SSE）")


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str | None = Field(default=settings.EMBEDDING_MODEL)
    embedding_backend: str | None = None


class EvidenceItem(BaseModel):
    evidence_id: str
    legacy_evidence_id: str | None = None
    source_type: str
    score: float
    source_rank: int
    source_score: float
    title: str
    snippet: str
    doc_id: str
    doc_name: str
    page_label: int | str | None = None
    block_type: str
    source_path: str | None = None
    reference_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    matched_entities: list[str] = Field(default_factory=list)
    retrieval_signals: dict[str, float] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)


class FullDocumentItem(BaseModel):
    doc_id: str
    doc_name: str
    source_type: str
    source_path: str | None = None
    content: str
    chunk_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class RoutePlanResponse(BaseModel):
    document: bool
    table: bool
    adela: bool
    reason: str
    sources: list[str]


class RetrieveResponse(BaseModel):
    query: str
    route_plan: RoutePlanResponse
    evidences: list[EvidenceItem]
    full_documents: list[FullDocumentItem] = Field(default_factory=list)
    timings: dict[str, Any]
    retrieved_count: int


class QueryResponse(RetrieveResponse):
    answer: str
    knowledge_base_fully_answered: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="LLM 给出的知识库证据充分回答置信度，范围 0.0-1.0；0 表示未命中、证据不足或仅返回降级片段。",
    )
    llm_config: dict[str, Any]


class EmbeddingObject(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingObject]
    model: str
    usage: dict[str, int] = Field(default_factory=dict)
