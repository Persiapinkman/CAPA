from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ace_rag.core.config import get_settings


settings = get_settings()

SourceType = Literal["document", "table", "adela"]


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
    sources: list[SourceType] | None = None
    route_with_llm: bool = False
    expand_query_with_llm: bool = Field(default=settings.ENABLE_V2_LLM_QUERY_EXPANSION)
    query_expansion_terms: list[str] = Field(default_factory=list)
    embedding_model: str | None = None
    embedding_models: list[str] | None = None
    embedding_backend: str | None = None
    document: SourceConfig = Field(default_factory=SourceConfig)
    table: SourceConfig = Field(default_factory=SourceConfig)
    adela: SourceConfig = Field(default_factory=SourceConfig)

    use_playbook: bool = True
    playbook_top_k: int = Field(default=settings.PLAYBOOK_TOP_K, ge=0, le=50)
    playbook_only: bool = False


class QueryRequest(RetrieveRequest):
    llm_config: LLMConfig | None = None
    stream: bool = False


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="allow")

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
    metadata: dict[str, Any] = Field(default_factory=dict)
    matched_entities: list[str] = Field(default_factory=list)
    retrieval_signals: dict[str, float] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)


class RoutePlanResponse(BaseModel):
    document: bool
    table: bool
    adela: bool
    reason: str
    sources: list[str]


class PlaybookItem(BaseModel):
    item_id: str
    section: str
    content: str
    status: str = "active"
    tags: list[str] = Field(default_factory=list)
    source_hints: list[str] = Field(default_factory=list)
    query_intents: list[str] = Field(default_factory=list)
    expansion_terms: list[str] = Field(default_factory=list)
    helpful_count: int = 0
    harmful_count: int = 0
    confidence: float = 0.5
    provenance: dict[str, Any] = Field(default_factory=dict)
    created_at: float | None = None
    updated_at: float | None = None


class PlaybookHit(PlaybookItem):
    score: float
    score_details: dict[str, float] = Field(default_factory=dict)


class PlaybookDebug(BaseModel):
    used: bool
    items: list[PlaybookHit] = Field(default_factory=list)
    query_expansion_terms: list[str] = Field(default_factory=list)
    source_hints: list[str] = Field(default_factory=list)


class RetrieveResponse(BaseModel):
    query: str
    route_plan: RoutePlanResponse
    evidences: list[EvidenceItem]
    timings: dict[str, Any]
    retrieved_count: int
    playbook: PlaybookDebug
    v2_request: dict[str, Any]


class QueryResponse(RetrieveResponse):
    answer: str
    llm_config: dict[str, Any]
    run_id: str


class FeedbackRequest(BaseModel):
    run_id: str
    feedback_type: Literal["helpful", "harmful", "correction", "missing_evidence", "other"] = "other"
    rating: int | None = Field(default=None, ge=1, le=5)
    corrected_answer: str | None = None
    expected_evidence_ids: list[str] = Field(default_factory=list)
    comment: str | None = None


class FeedbackResponse(BaseModel):
    feedback_id: str
    operation_id: str | None = None
    status: str


class PlaybookOrganizeRequest(BaseModel):
    include_sections: list[str] | None = None
    include_inactive: bool = False
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    max_items: int = Field(default=200, ge=1, le=1000)


class PlaybookOrganizeCandidate(BaseModel):
    candidate_id: str
    title: str
    summary: str
    strategy: str
    item_ids: list[str]
    sections: list[str]
    tags: list[str] = Field(default_factory=list)
    source_hints: list[str] = Field(default_factory=list)
    query_intents: list[str] = Field(default_factory=list)
    expansion_terms: list[str] = Field(default_factory=list)
    confidence: float
    rationale: str
    apply_mode: Literal["preview_only", "manual_review", "auto_upsert"] = "preview_only"


class PlaybookOrganizeResponse(BaseModel):
    item_count: int
    strategies: list[dict[str, str]]
    candidates: list[PlaybookOrganizeCandidate]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service: str
    v2: dict[str, Any]
    playbook: dict[str, Any]
