from copy import deepcopy
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union, Literal
from src.core.config import get_settings

settings = get_settings()


def _default_data_source_vector_store_configs() -> Dict[str, "VectorStoreConfig"]:
    return {
        model_name: VectorStoreConfig(**cfg)
        for model_name, cfg in deepcopy(settings.DATA_SOURCE_VECTOR_STORE_CONFIGS).items()
    }

class ChunkNode(BaseModel):
    id: str = Field(..., description="节点（小文本块）唯一标识符")
    text: str = Field(..., description="小文本块内容")
    index_id: str = Field(..., description="对应的大文本块的唯一标识符")
    index_text: str = Field(..., description="对应的大文本块的完整内容")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="包含文档标题、页码等元数据信息")
    doc_id: str = Field(..., description="文档的唯一标识符")
    doc_name: str = Field(..., description="文档的原始文件名")

class EmbeddingInfo(BaseModel):
    model: str = Field(..., description="用于生成向量的模型名称,如bge-m3")
    embedding: List[float] = Field(..., description="文本块的向量表示")

class ChunkNodeWithEmbedding(ChunkNode):
    embeddings: List[EmbeddingInfo] = Field(..., description="文本块在不同模型下的向量表示列表")

class ChunkNodeWithScore(ChunkNode):
    score: float = Field(..., description="文本块与查询的相关性得分")


class VectorStoreConfig(BaseModel):
    uri: str = Field(..., description="向量数据库连接地址")
    collection_name: str = Field(..., description="向量数据库中的 collection 名称")

TEXT_EXAMPLE = "示例文本：这是一个用于接口示例展示的文档内容。"

class ChunkingEmbeddingRequest(BaseModel):
    text: str = Field(..., description="待进行处理的原始文本内容", example=TEXT_EXAMPLE)
    input_type: str = Field(
        default="autopdf",
        description="输入文本的格式类型，当前支持 autopdf、raw、json_list、markdown、mineru、pdf_blocks",
        example="autopdf"
    )
    doc_id: str = Field(..., description="唯一标识文档", example="6613d59e8028bd3edc8c3f03d3ef5977")
    doc_name: str = Field(..., description="原始文档的文件名", example="甲状腺球蛋白核实试剂说明书")
    embedding_models: List[str] = Field(
        default=settings.DEFAULT_EMBEDDING_MODELS,
        description="用于生成文本向量的模型列表,可同时使用多个模型",
        example=settings.DEFAULT_EMBEDDING_MODELS
    )

class ChunkingEmbeddingResponse(BaseModel):
    doc_id: str = Field(..., description="唯一标识文档")
    doc_name: str = Field(..., description="原始文档的文件名")
    index_nodes: List[ChunkNodeWithEmbedding] = Field(..., description="文档分块和向量化后的结果列表")
    success: bool = Field(default=True, description="文档处理是否成功完成")
    message: Optional[str] = Field(None, description="处理失败时的错误信息说明")

class RetrievingRequest(BaseModel):
    query: str = Field(..., description="用户输入的检索查询文本", example="如何在软件中切换到cobas link界面？")
    retrieval_method: Literal["vector", "bm25", "hybrid"] = Field(
        default=settings.DEFAULT_RETRIEVAL_METHOD,
        description="检索方法，`vector` 为向量检索，`bm25` 为关键词检索，`hybrid` 为向量+BM25 混合检索（默认，RRF 融合）",
        example=settings.DEFAULT_RETRIEVAL_METHOD,
    )
    top_k: Optional[int] = Field(default=5, description="需要返回的最相关文档数量上限", example=5)
    similarity_threshold: Optional[float] = Field(default=0.5, description="相似度的下限阈值（小于会被筛掉）", example=0.5)
    filter: Optional[str] = Field(None, description="检索结果的过滤条件表达式", example="doc_name == 'cobas pro 用户培训指导手册 e801 v1'")
    uri: str = Field(default=settings.TEST_VECTOR_DB_URI, description="向量数据库的连接地址（支持本地和远程）", example=settings.TEST_VECTOR_DB_URI)
    collection_name: str = Field(default=settings.TEST_COLLECTION_NAME, description="向量数据库中的collection名称，根据实际情况填入", example=settings.TEST_COLLECTION_NAME)
    embedding_models: List[str] = Field(
        default=settings.DEFAULT_EMBEDDING_MODELS,
        description="用于生成查询向量的模型列表,需与索引时使用的模型对应",
        example=settings.DEFAULT_EMBEDDING_MODELS
    )
    vector_store_configs: Optional[Dict[str, VectorStoreConfig]] = Field(
        default=None,
        description="可选的模型到向量库路由配置；当不同 embedding 模型存放在不同库时使用",
    )


class SearchAgentRetrievingRequest(RetrievingRequest):
    uri: str = Field(
        default=settings.DATA_SOURCE_VECTOR_DB_URI,
        description="向量数据库的连接地址，默认使用 data_source 离线入库生成的本地库",
        example=settings.DATA_SOURCE_VECTOR_DB_URI,
    )
    collection_name: str = Field(
        default=settings.DATA_SOURCE_COLLECTION_NAME,
        description="向量数据库中的 collection 名称，默认使用 data_source 本地库对应 collection",
        example=settings.DATA_SOURCE_COLLECTION_NAME,
    )
    vector_store_configs: Optional[Dict[str, VectorStoreConfig]] = Field(
        default_factory=_default_data_source_vector_store_configs,
        description="默认按 embedding 模型路由到各自的 data_source 向量库",
    )

class TableMatchedRow(BaseModel):
    row_id: str = Field(..., description="唯一的行 ID")
    score: float = Field(..., description="行级相关性分数")
    matched_fields: List[str] = Field(default_factory=list, description="命中的字段列表")
    entity: Dict[str, Any] = Field(default_factory=dict, description="命中的结构化行内容")


class TableChatRequest(BaseModel):
    query: str = Field(..., description="用户问题", example="安全绳有哪些模型？")
    retrieval_method: Literal["keyword", "vector", "hybrid"] = Field(
        default=settings.DEFAULT_TABLE_RETRIEVAL_METHOD,
        description="表格检索方式，支持关键词、向量和混合检索",
        example=settings.DEFAULT_TABLE_RETRIEVAL_METHOD,
    )
    top_k: Optional[int] = Field(default=20, description="返回的相关行数量上限", example=20)
    similarity_threshold: Optional[float] = Field(
        default=0.15,
        description="行级相似度过滤阈值",
        example=0.15,
    )
    data_path: str = Field(
        default=settings.TABLE_DATA_JSONL_PATH,
        description="表格 JSONL 数据路径",
        example=settings.TABLE_DATA_JSONL_PATH,
    )
    searchable_fields: Optional[List[str]] = Field(
        default=None,
        description="参与检索的字段列表；不传则使用服务默认配置",
    )
    return_fields: Optional[List[str]] = Field(
        default=None,
        description="返回给前端和 LLM 的字段列表；不传则使用服务默认配置",
    )
    embedding_models: List[str] = Field(
        default=settings.DEFAULT_STRUCTURED_EMBEDDING_MODEL,
        description="表格向量检索使用的 embedding 模型列表；vector / hybrid 下仅支持一个模型",
        example=settings.DEFAULT_STRUCTURED_EMBEDDING_MODEL,
    )
    llm_config: Optional["LLMConfig"] = Field(
        default=None,
        description="可选的大模型配置，不传则使用服务默认配置",
    )

class AdelaChatRequest(BaseModel):
    query: str = Field(..., description="用户问题", example="有哪些 cuda11.0-trt7.1-fp16-T4 的模型？")
    retrieval_method: Literal["keyword", "vector", "hybrid"] = Field(
        default=settings.DEFAULT_ADELA_RETRIEVAL_METHOD,
        description="adela 检索方式，支持关键词、向量和混合检索",
        example=settings.DEFAULT_ADELA_RETRIEVAL_METHOD,
    )
    top_k: Optional[int] = Field(default=20, description="返回的相关记录数量上限", example=20)
    similarity_threshold: Optional[float] = Field(
        default=0.15,
        description="记录级相似度过滤阈值",
        example=0.15,
    )
    data_path: str = Field(
        default=settings.ADELA_DATA_JSONL_PATH,
        description="adela 规范化 JSONL 数据路径",
        example=settings.ADELA_DATA_JSONL_PATH,
    )
    source_dir: Optional[str] = Field(
        default=settings.ADELA_DATA_DIR,
        description="可选的 adela 原始 JSON 目录；当 data_path 不存在时用于自动导出",
        example=settings.ADELA_DATA_DIR,
    )
    searchable_fields: Optional[List[str]] = Field(
        default=None,
        description="参与检索的字段列表；不传则使用 adela 默认配置",
    )
    return_fields: Optional[List[str]] = Field(
        default=None,
        description="返回给前端和 LLM 的字段列表；不传则使用 adela 默认配置",
    )
    embedding_models: List[str] = Field(
        default=settings.DEFAULT_STRUCTURED_EMBEDDING_MODEL,
        description="adela 向量检索使用的 embedding 模型列表；vector / hybrid 下仅支持一个模型",
        example=settings.DEFAULT_STRUCTURED_EMBEDDING_MODEL,
    )
    llm_config: Optional["LLMConfig"] = Field(
        default=None,
        description="可选的大模型配置，不传则使用服务默认配置",
    )


class UnifiedDocumentRetrieveConfig(BaseModel):
    enabled: bool = Field(default=True, description="是否启用模型发版文档正文（ONES 工作文档 / 发版 PDF）检索")
    retrieval_method: Literal["vector", "bm25", "hybrid"] = Field(
        default=settings.DEFAULT_RETRIEVAL_METHOD,
        description="模型发版文档正文检索方法",
        example=settings.DEFAULT_RETRIEVAL_METHOD,
    )
    top_k: Optional[int] = Field(default=5, description="模型发版文档正文返回块数上限", example=5)
    similarity_threshold: Optional[float] = Field(
        default=0.5,
        description="模型发版文档正文相似度阈值",
        example=0.5,
    )
    filter: Optional[str] = Field(None, description="模型发版文档正文检索过滤条件")
    uri: str = Field(
        default=settings.DATA_SOURCE_VECTOR_DB_URI,
        description="模型发版文档正文向量库路径",
        example=settings.DATA_SOURCE_VECTOR_DB_URI,
    )
    collection_name: str = Field(
        default=settings.DATA_SOURCE_COLLECTION_NAME,
        description="模型发版文档正文向量库 collection",
        example=settings.DATA_SOURCE_COLLECTION_NAME,
    )
    embedding_models: List[str] = Field(
        default=settings.DEFAULT_EMBEDDING_MODELS,
        description="模型发版文档正文检索使用的 embedding 模型列表",
        example=settings.DEFAULT_EMBEDDING_MODELS,
    )
    vector_store_configs: Optional[Dict[str, VectorStoreConfig]] = Field(
        default_factory=_default_data_source_vector_store_configs,
        description="模型发版文档正文可选模型到向量库路由配置",
    )


class UnifiedTableRetrieveConfig(BaseModel):
    enabled: bool = Field(default=True, description="是否启用 tables 检索")
    retrieval_method: Literal["keyword", "vector", "hybrid"] = Field(
        default=settings.DEFAULT_TABLE_RETRIEVAL_METHOD,
        description="tables 检索方式",
        example=settings.DEFAULT_TABLE_RETRIEVAL_METHOD,
    )
    top_k: Optional[int] = Field(default=20, description="tables 返回行数上限", example=20)
    similarity_threshold: Optional[float] = Field(
        default=0.15,
        description="tables 相似度阈值",
        example=0.15,
    )
    data_path: str = Field(
        default=settings.TABLE_DATA_JSONL_PATH,
        description="tables JSONL 路径",
        example=settings.TABLE_DATA_JSONL_PATH,
    )
    searchable_fields: Optional[List[str]] = Field(
        default=None,
        description="tables 参与检索字段列表",
    )
    return_fields: Optional[List[str]] = Field(
        default=None,
        description="tables 返回字段列表",
    )
    embedding_models: List[str] = Field(
        default=settings.DEFAULT_STRUCTURED_EMBEDDING_MODEL,
        description="tables 向量检索模型列表",
        example=settings.DEFAULT_STRUCTURED_EMBEDDING_MODEL,
    )


class UnifiedAdelaRetrieveConfig(BaseModel):
    enabled: bool = Field(default=True, description="是否启用 adela 检索")
    retrieval_method: Literal["keyword", "vector", "hybrid"] = Field(
        default=settings.DEFAULT_ADELA_RETRIEVAL_METHOD,
        description="adela 检索方式",
        example=settings.DEFAULT_ADELA_RETRIEVAL_METHOD,
    )
    top_k: Optional[int] = Field(default=20, description="adela 返回记录数上限", example=20)
    similarity_threshold: Optional[float] = Field(
        default=0.15,
        description="adela 相似度阈值",
        example=0.15,
    )
    data_path: str = Field(
        default=settings.ADELA_DATA_JSONL_PATH,
        description="adela JSONL 路径",
        example=settings.ADELA_DATA_JSONL_PATH,
    )
    source_dir: Optional[str] = Field(
        default=settings.ADELA_DATA_DIR,
        description="adela 原始 JSON 目录（当 data_path 不存在时用于自动导出）",
        example=settings.ADELA_DATA_DIR,
    )
    searchable_fields: Optional[List[str]] = Field(
        default=None,
        description="adela 参与检索字段列表",
    )
    return_fields: Optional[List[str]] = Field(
        default=None,
        description="adela 返回字段列表",
    )
    embedding_models: List[str] = Field(
        default=settings.DEFAULT_STRUCTURED_EMBEDDING_MODEL,
        description="adela 向量检索模型列表",
        example=settings.DEFAULT_STRUCTURED_EMBEDDING_MODEL,
    )


class UnifiedPublicCloudRetrieveConfig(BaseModel):
    enabled: bool = Field(default=True, description="是否启用 public_cloud 检索")
    retrieval_method: Literal["keyword"] = Field(
        default=settings.DEFAULT_PUBLIC_CLOUD_RETRIEVAL_METHOD,
        description="public_cloud 检索方式（当前仅支持 keyword）",
        example=settings.DEFAULT_PUBLIC_CLOUD_RETRIEVAL_METHOD,
    )
    top_k: Optional[int] = Field(
        default=settings.PUBLIC_CLOUD_TOP_K,
        description="public_cloud 返回记录数上限",
        example=settings.PUBLIC_CLOUD_TOP_K,
    )
    api_url: str = Field(
        default=settings.PUBLIC_CLOUD_MODELS_API_URL,
        description="public_cloud 模型列表接口 URL",
        example=settings.PUBLIC_CLOUD_MODELS_API_URL,
    )
    api_token: str = Field(
        default=settings.PUBLIC_CLOUD_MODELS_API_TOKEN,
        description="public_cloud 接口 Bearer Token",
        example=settings.PUBLIC_CLOUD_MODELS_API_TOKEN,
    )


class UnifiedQueryRequest(BaseModel):
    query: str = Field(..., description="用户问题", example="安全绳模型在文档、表格和 adela 中都有哪些记录？")
    fused_top_k: Optional[int] = Field(default=12, description="融合后返回证据条数上限", example=12)
    rrf_k: Optional[int] = Field(default=60, description="RRF 融合参数 k", example=60)
    stream: bool = Field(default=False, description="是否开启流式回答（SSE）")
    route_with_llm: bool = Field(
        default=True,
        description="是否在检索前先让 LLM 判断需要调用哪些数据源",
    )
    document_config: UnifiedDocumentRetrieveConfig = Field(
        default_factory=UnifiedDocumentRetrieveConfig,
        description="模型发版文档正文（ONES 工作文档 / 发版 PDF）子检索配置",
    )
    table_config: UnifiedTableRetrieveConfig = Field(
        default_factory=UnifiedTableRetrieveConfig,
        description="tables 子检索配置",
    )
    adela_config: UnifiedAdelaRetrieveConfig = Field(
        default_factory=UnifiedAdelaRetrieveConfig,
        description="adela 子检索配置",
    )
    public_cloud_config: UnifiedPublicCloudRetrieveConfig = Field(
        default_factory=UnifiedPublicCloudRetrieveConfig,
        description="public_cloud 子检索配置",
    )
    llm_config: Optional["LLMConfig"] = Field(
        default=None,
        description="可选的大模型配置，不传则使用服务默认配置",
    )


class UnifiedRetrieveRequest(BaseModel):
    query: str = Field(..., description="用户问题", example="安全绳模型在文档、表格和 adela 中都有哪些记录？")
    source_types: Optional[List[Literal["document", "table", "adela", "public_cloud"]]] = Field(
        default=None,
        description="限制检索的数据源类型范围；不传时使用全部启用来源",
        example=["document", "table"],
    )
    fused_top_k: Optional[int] = Field(default=12, description="融合后返回证据条数上限", example=12)
    rrf_k: Optional[int] = Field(default=60, description="RRF 融合参数 k", example=60)
    document_config: UnifiedDocumentRetrieveConfig = Field(
        default_factory=UnifiedDocumentRetrieveConfig,
        description="模型发版文档正文（ONES 工作文档 / 发版 PDF）子检索配置",
    )
    table_config: UnifiedTableRetrieveConfig = Field(
        default_factory=UnifiedTableRetrieveConfig,
        description="tables 子检索配置",
    )
    adela_config: UnifiedAdelaRetrieveConfig = Field(
        default_factory=UnifiedAdelaRetrieveConfig,
        description="adela 子检索配置",
    )
    public_cloud_config: UnifiedPublicCloudRetrieveConfig = Field(
        default_factory=UnifiedPublicCloudRetrieveConfig,
        description="public_cloud 子检索配置",
    )


class LLMConfig(BaseModel):
    model: str = Field(default=settings.DEFAULT_LLM_MODEL, description="模型名称")
    base_url: str = Field(default=settings.DEFAULT_LLM_BASE_URL, description="模型API地址")
    api_key: Optional[str] = Field(settings.DEFAULT_LLM_API_KEY, description="模型API密钥")
    max_tokens: Optional[int] = Field(default=settings.DEFAULT_LLM_MAX_TOKENS, description="最大token数量")
    temperature: Optional[float] = Field(default=settings.DEFAULT_LLM_TEMPERATURE, description="温度")
    top_p: Optional[float] = Field(default=settings.DEFAULT_LLM_TOP_P, description="top_p")
    seed: Optional[int] = Field(default=settings.DEFAULT_LLM_SEED, description="采样随机种子（后端支持时生效）")

class SearchAgentRequest(BaseModel):
    search_engine_url: Optional[str] = Field(
        default=None,
        description="可选的搜索引擎API地址；不传时优先直接使用当前服务内的检索实现",
        example="http://127.0.0.1:6060",
    )
    retrieve_config: SearchAgentRetrievingRequest = Field(..., description="检索配置")
    llm_config: Optional[LLMConfig] = Field(
        default=None,
        description="可选的LLM配置，不传则使用服务默认配置",
    )
    max_turns: Optional[int] = Field(default=5, description="最大轮数", example=5)


class RAGChatRequest(BaseModel):
    query: str = Field(..., description="用户问题", example="safety_rope v0.2.1 追加了什么数据，标签有哪些？")
    retrieval_method: Literal["vector", "bm25", "hybrid"] = Field(
        default=settings.DEFAULT_RETRIEVAL_METHOD,
        description="检索方法，`vector` 为向量检索，`bm25` 为关键词检索，`hybrid` 为向量+BM25 混合检索（默认，RRF 融合）",
        example=settings.DEFAULT_RETRIEVAL_METHOD,
    )
    top_k: Optional[int] = Field(default=5, description="返回的最相关文档块数量", example=5)
    similarity_threshold: Optional[float] = Field(default=0.5, description="相似度过滤阈值", example=0.5)
    filter: Optional[str] = Field(None, description="检索过滤条件", example="doc_name == 'PDFs/safety_rope v0.2.1.pdf'")
    uri: str = Field(
        default=settings.DATA_SOURCE_VECTOR_DB_URI,
        description="向量数据库路径，默认使用 data_source 离线入库生成的本地库",
        example=settings.DATA_SOURCE_VECTOR_DB_URI,
    )
    collection_name: str = Field(
        default=settings.DATA_SOURCE_COLLECTION_NAME,
        description="向量数据库 collection 名称",
        example=settings.DATA_SOURCE_COLLECTION_NAME,
    )
    embedding_models: List[str] = Field(
        default=settings.DEFAULT_EMBEDDING_MODELS,
        description="用于检索的向量模型列表",
        example=settings.DEFAULT_EMBEDDING_MODELS,
    )
    vector_store_configs: Optional[Dict[str, VectorStoreConfig]] = Field(
        default_factory=_default_data_source_vector_store_configs,
        description="默认按 embedding 模型路由到各自的 data_source 向量库",
    )
    llm_config: Optional[LLMConfig] = Field(
        default=None,
        description="可选的大模型配置，不传则使用服务默认配置",
    )


class ReferenceItem(BaseModel):
    doc_name: str = Field(
        ...,
        description="去重后的来源名称（document 通常为模型发版文档文件名；adela 通常为模型名 + did）",
    )
    url: Optional[str] = Field(
        None,
        description="来源对应的跳转链接，未配置时为 null",
    )


class RAGChatTimings(BaseModel):
    retrieve_ms: float = Field(..., description="检索阶段耗时，单位毫秒")
    answer_ms: float = Field(..., description="大模型回答生成阶段耗时，单位毫秒")
    reference_ms: float = Field(..., description="reference 去重与映射阶段耗时，单位毫秒")
    total_ms: float = Field(..., description="模型发版文档正文问答总耗时，单位毫秒")


class RAGChatResponse(BaseModel):
    query: str = Field(..., description="用户问题")
    retrieved_chunks: List[ChunkNodeWithScore] = Field(..., description="检索返回的模型发版文档正文相关块")
    reference: List[ReferenceItem] = Field(default_factory=list, description="检索命中的模型发版文档来源链接列表，按文档去重")
    answer: str = Field(..., description="基于检索结果生成的回答")
    timings: RAGChatTimings = Field(..., description="模型发版文档正文问答各阶段耗时，单位毫秒")
    success: bool = Field(default=True, description="问答流程是否成功")
    message: Optional[str] = Field(None, description="失败时的错误信息")


class TableChatTimings(BaseModel):
    retrieve_ms: float = Field(..., description="检索阶段耗时，单位毫秒")
    answer_ms: float = Field(..., description="回答生成阶段耗时，单位毫秒")
    total_ms: float = Field(..., description="总耗时，单位毫秒")


class TableChatResponse(BaseModel):
    query: str = Field(..., description="用户问题")
    matched_rows: List[TableMatchedRow] = Field(
        default_factory=list,
        description="命中的结构化表格行",
    )
    answer: str = Field(..., description="基于命中行生成的回答")
    timings: TableChatTimings = Field(..., description="表格问答各阶段耗时，单位毫秒")
    success: bool = Field(default=True, description="问答流程是否成功")
    message: Optional[str] = Field(None, description="失败时的错误信息")

class AdelaChatTimings(BaseModel):
    retrieve_ms: float = Field(..., description="检索阶段耗时，单位毫秒")
    answer_ms: float = Field(..., description="回答生成阶段耗时，单位毫秒")
    total_ms: float = Field(..., description="总耗时，单位毫秒")


class AdelaChatResponse(BaseModel):
    query: str = Field(..., description="用户问题")
    matched_records: List[TableMatchedRow] = Field(
        default_factory=list,
        description="命中的 adela 结构化记录",
    )
    reference: List[ReferenceItem] = Field(
        default_factory=list,
        description="命中的 adela 部署链接列表（基于 did 去重）",
    )
    answer: str = Field(..., description="基于命中记录生成的回答")
    timings: AdelaChatTimings = Field(..., description="adela 问答各阶段耗时，单位毫秒")
    success: bool = Field(default=True, description="问答流程是否成功")
    message: Optional[str] = Field(None, description="失败时的错误信息")


class UnifiedEvidenceItem(BaseModel):
    evidence_id: str = Field(..., description="统一证据ID")
    source_type: Literal["document", "table", "adela", "public_cloud"] = Field(
        ...,
        description="证据来源类型（document=模型发版文档正文/ONES 工作文档，table=模型发版汇总表，adela=部署记录，public_cloud=公有云在线模型列表）",
    )
    score: float = Field(..., description="融合后的相关性分数")
    source_rank: int = Field(..., description="该来源内的排序名次（从1开始）")
    source_score: Optional[float] = Field(default=None, description="该来源原始分数")
    title: str = Field(..., description="证据标题")
    snippet: str = Field(..., description="证据摘要内容")
    payload: Dict[str, Any] = Field(default_factory=dict, description="证据原始结构化数据")


class UnifiedSourceStatus(BaseModel):
    source_type: Literal["document", "table", "adela", "public_cloud"] = Field(
        ...,
        description="来源类型（document=模型发版文档正文/ONES 工作文档，table=模型发版汇总表，adela=部署记录，public_cloud=公有云在线模型列表）",
    )
    enabled: bool = Field(..., description="是否启用该来源")
    success: bool = Field(..., description="该来源检索是否成功")
    retrieve_ms: float = Field(..., description="该来源检索耗时，单位毫秒")
    retrieved_count: int = Field(default=0, description="该来源原始命中数量")
    used_count: int = Field(default=0, description="该来源参与融合的数量")
    message: Optional[str] = Field(None, description="该来源失败原因")


class UnifiedRoutePlan(BaseModel):
    route_with_llm: bool = Field(..., description="是否启用 LLM 路由")
    selected_sources: List[Literal["document", "table", "adela", "public_cloud"]] = Field(
        default_factory=list,
        description="LLM 决策后实际执行检索的数据源（其中 document 表示模型发版文档正文/ONES 工作文档）",
    )
    skipped_sources: List[Literal["document", "table", "adela", "public_cloud"]] = Field(
        default_factory=list,
        description="被 LLM 跳过或配置禁用的数据源（其中 document 表示模型发版文档正文/ONES 工作文档）",
    )
    fallback_used: bool = Field(default=False, description="LLM 路由失败后是否回退到默认策略")
    reason: Optional[str] = Field(None, description="LLM 路由理由或回退说明")


class UnifiedQueryTimings(BaseModel):
    route_ms: float = Field(..., description="LLM 路由耗时，单位毫秒")
    retrieve_ms: float = Field(..., description="三路检索总耗时（并行墙钟），单位毫秒")
    fuse_ms: float = Field(..., description="融合耗时，单位毫秒")
    answer_ms: float = Field(..., description="回答生成耗时，单位毫秒")
    total_ms: float = Field(..., description="总耗时，单位毫秒")


class UnifiedQueryResponse(BaseModel):
    query: str = Field(..., description="用户问题")
    fused_evidences: List[UnifiedEvidenceItem] = Field(
        default_factory=list,
        description="融合后的统一证据列表",
    )
    reference: List[ReferenceItem] = Field(
        default_factory=list,
        description="融合证据对应的来源链接列表（包含模型发版文档来源链接与 adela 的 did 链接）",
    )
    source_status: List[UnifiedSourceStatus] = Field(
        default_factory=list,
        description="各来源检索状态与统计",
    )
    route_plan: UnifiedRoutePlan = Field(..., description="LLM 路由计划与执行情况")
    answer: str = Field(..., description="基于融合证据生成的最终回答")
    timings: UnifiedQueryTimings = Field(..., description="统一检索问答各阶段耗时，单位毫秒")
    success: bool = Field(default=True, description="问答流程是否成功")
    message: Optional[str] = Field(None, description="失败时的错误信息")


class UnifiedRetrieveTimings(BaseModel):
    retrieve_ms: float = Field(..., description="三路检索总耗时（并行墙钟），单位毫秒")
    fuse_ms: float = Field(..., description="融合耗时，单位毫秒")
    total_ms: float = Field(..., description="总耗时，单位毫秒")


class UnifiedRetrieveResponse(BaseModel):
    query: str = Field(..., description="用户问题")
    selected_sources: List[Literal["document", "table", "adela", "public_cloud"]] = Field(
        default_factory=list,
        description="本次请求实际执行检索的数据源",
    )
    fused_evidences: List[UnifiedEvidenceItem] = Field(
        default_factory=list,
        description="融合后的统一证据列表",
    )
    reference: List[ReferenceItem] = Field(
        default_factory=list,
        description="融合证据对应的来源链接列表（包含模型发版文档来源链接与 adela 的 did 链接）",
    )
    source_status: List[UnifiedSourceStatus] = Field(
        default_factory=list,
        description="各来源检索状态与统计",
    )
    timings: UnifiedRetrieveTimings = Field(..., description="统一检索各阶段耗时，单位毫秒")
    success: bool = Field(default=True, description="检索流程是否成功")
    message: Optional[str] = Field(None, description="失败时的错误信息")


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field(..., description="需要向量化的文本内容，可以是单个字符串或字符串列表")
    model: str = Field(
        default=settings.DEFAULT_SINGLE_EMBEDDING_MODEL[0],
        description="用于生成向量的模型名称",
        example=settings.DEFAULT_SINGLE_EMBEDDING_MODEL[0],
    )

class Embedding(BaseModel):
    index: int = Field(..., description="向量索引")
    embedding: List[float] = Field(..., description="文本的向量表示")
    object: str = Field(default="embedding", description="对象类型")

class Usage(BaseModel):
    prompt_tokens: int = Field(..., description="输入文本的token数量")
    completion_tokens: int = Field(default=0, description="完成的token数量")
    total_tokens: int = Field(..., description="总token数量")

class EmbeddingList(BaseModel):
    object: str = Field(default="list", description="对象类型")
    data: List[Embedding] = Field(..., description="向量列表")
    model: str = Field(..., description="使用的模型名称")
    usage: Usage = Field(..., description="token使用情况")


TableChatRequest.model_rebuild()
AdelaChatRequest.model_rebuild()
UnifiedQueryRequest.model_rebuild()
UnifiedRetrieveRequest.model_rebuild()
