from datetime import datetime

from src.core.config import get_default_llm_config, get_settings

settings = get_settings()

DEFAULT_LLM_API_CONFIG = get_default_llm_config()

# 搜索引擎配置
DEFAULT_SEARCH_CONFIG = {
    "top_k": 5,
    "uri": settings.DATA_SOURCE_VECTOR_DB_URI,
    "collection_name": settings.DATA_SOURCE_COLLECTION_NAME,
    "embedding_models": settings.DEFAULT_EMBEDDING_MODELS,
    "vector_store_configs": settings.DATA_SOURCE_VECTOR_STORE_CONFIGS,
    "search_engine_url": None,
    "filter": None
}

DEFAULT_SEARCH_AGENT_CONFIG = {
    "max_turns": 5,
    # "max_search_times": 5
}

# 系统提示词
SYSTEM_PROMPT = f"""\
今天日期：{datetime.now().strftime("%Y-%m-%d")}

你是一个专为检索增强生成(Retrieval-Augmented Generation, RAG)构建的AI智能问答系统。
你可以使用chunk_search工具从本地向量数据库对相关信息进行语义检索，帮助回答用户问题。
在未能找到足够的信息来回答用户的问题时，务必多次使用chunk_search工具。

以下是一些指导原则：
- 记住，如果需要多轮chunk_search来查找相关信息，请确保在提供最终答案前完成所有搜索任务。
- 对于复杂问题，考虑将其分解为更简单的子问题，以便进行更有效的搜索。
- 由于向量数据库存储的是切分后的文本chunk，因此部分检索结果可能是不完整的。
- 由于向量数据库通常返回的是部分检索结果而且不一定与检索query严格相关，因此需要不断调整查询策略，多次使用chunk_search工具。
- 仔细辨别检索结果的相关性，某些内容可能表面相似但实际不相关，需要进行准确判断。
- 最终答案应尽量给出原文作为回复，不要包含无关信息，确保回答与用户问题高度相关。

再次强调：在未能找到足够的信息来回答用户的问题之前，务必不断调整检索关键词，尽最大努力去检索相关信息！！！
""".strip()

# 工具定义
CHUNK_SEARCH_TOOL = {
    "name": "chunk_search",
    "description": "根据query进行chunk搜索",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "查询内容"
            },
        },
        "required": ["query"]
    }
}

CHUNK_DELETE_TOOL = {
    "name": "chunk_delete",
    "description": "删除历史对话中与用户提问无关的检索结果，防止对问题的回答造成干扰。",
    "parameters": {
        "type": "object",
        "properties": {
            "chunk_ids_to_delete": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "需要删除的chunk_id列表（若历史对话中的检索结果与用户提问均有关，则返回空列表）"
            },
        },
        "required": ["chunk_ids_to_delete"],
    },
}
