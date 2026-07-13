import json
import asyncio
from typing import Dict, List, Optional

from src.api.schemas import RetrievingRequest
from src.core.logging import get_logger
from src.rag.runtime import get_rag_service
from src.rag.search_agent.config import DEFAULT_SEARCH_CONFIG

logger = get_logger(__name__)


def format_chunk_texts(chunk_texts: List[Dict]) -> str:
    """格式化检索结果文本"""
    tool_result = []
    for res in chunk_texts:
        tool_result.append(
            {
                "doc_name": res["doc_name"],
                "chunk_id": res["id"],
                "text": res["text"].replace(" ", ""),
            }
        )
    return json.dumps(tool_result, ensure_ascii=False)


async def chunk_search(
    query: str,
    retrieve_config: RetrievingRequest,
    search_engine_url: str = None,
) -> List[Dict]:
    """异步根据query进行chunk搜索"""
    # 使用默认配置
    config = DEFAULT_SEARCH_CONFIG
    top_k = retrieve_config.top_k or config["top_k"]
    uri = retrieve_config.uri or config["uri"]
    collection_name = retrieve_config.collection_name or config["collection_name"]
    embedding_models = retrieve_config.embedding_models or config["embedding_models"]
    vector_store_configs = (
        retrieve_config.vector_store_configs or config["vector_store_configs"]
    )
    filter = retrieve_config.filter or config["filter"]
    request_data = {
        "query": query,
        "top_k": top_k,
        "uri": uri,
        "collection_name": collection_name,
        "embedding_models": embedding_models,
        "vector_store_configs": vector_store_configs,
        "filter": filter,
    }

    if search_engine_url:
        logger.info(
            "search_engine_url=%s 已传入，但当前 search_agent 优先直接使用本项目内的检索实现",
            search_engine_url,
        )

    request = RetrievingRequest(**request_data)
    rag_service = get_rag_service()
    result = await asyncio.to_thread(rag_service.retrieving, request)
    result = [entry.model_dump() for entry in result]

    for res in result:
        res.pop("index_text", None)
        res.pop("index_id", None)
        res.pop("doc_id", None)

    # 后处理检索结果
    result = _postprocess_search_results(result)

    return result


def _postprocess_search_results(result: List[Dict]) -> List[Dict]:
    """后处理检索结果：去重、清理文本、过滤长度"""
    # 去重
    dedup_result = []
    unique_text = set()
    for res in result:
        if res["text"] not in unique_text:
            unique_text.add(res["text"])
            dedup_result.append(res)

    if len(dedup_result) != len(result):
        logger.warning(
            f"检索结果去重：原始{len(result)}条 -> 去重后{len(dedup_result)}条"
        )

    result = dedup_result

    # 清理文本和去重段落
    for res in result:
        res["text"] = res["text"].replace(" ", "")
        splits = res["text"].split("\n\n")
        dedup_splits = []
        for split in splits:
            if len(split) < 10 or split not in dedup_splits:
                dedup_splits.append(split)
        res["text"] = "\n\n".join(dedup_splits)

    # 再次去重
    unique_text = set()
    dedup_result = []
    for res in result:
        if res["text"] not in unique_text:
            unique_text.add(res["text"])
            dedup_result.append(res)
    result = dedup_result

    # 过滤长度超过8000的结果
    result = [res for res in result if len(res["text"]) < 8000]

    # 简化doc_id
    for res in result:
        splits = res["id"].split("_")
        if len(splits) > 1:
            splits[1] = splits[1][:8]
            res["id"] = "_".join(splits)

    return result
