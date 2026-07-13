import json
import time
from typing import AsyncGenerator

from src.api.schemas import LLMConfig, SearchAgentRequest
from src.core.logging import get_logger
from src.rag.reference_service import PDFReferenceStore
from src.rag.search_agent.config import (
    CHUNK_DELETE_TOOL,
    CHUNK_SEARCH_TOOL,
    DEFAULT_SEARCH_AGENT_CONFIG,
    DEFAULT_LLM_API_CONFIG,
    SYSTEM_PROMPT,
)
from src.rag.search_agent.llm_utils import (
    format_chat_template,
    generate_response,
    generate_response_stream,
    process_response,
)
from src.rag.search_agent.tool_utils import chunk_search, format_chunk_texts

logger = get_logger(__name__)

# 相关数据记录
retrieve_request_count = 0
llm_request_count = 0
llm_input_token_count = 0
llm_output_token_count = 0
pdf_reference_store = PDFReferenceStore()


def _resolve_search_agent_llm_config(llm_config):
    if llm_config is not None:
        return llm_config

    return LLMConfig(
        model=DEFAULT_LLM_API_CONFIG["model"],
        base_url=DEFAULT_LLM_API_CONFIG["base_url"],
        api_key=DEFAULT_LLM_API_CONFIG["api_key"],
        max_tokens=DEFAULT_LLM_API_CONFIG["max_tokens"],
        temperature=DEFAULT_LLM_API_CONFIG["temperature"],
        top_p=DEFAULT_LLM_API_CONFIG["top_p"],
        seed=DEFAULT_LLM_API_CONFIG.get("seed"),
    )


def _should_trigger_chunk_delete(tools, turn_count, chunks_in_history_dict):
    """判断是否应该触发chunk_delete工具"""
    return (
        "chunk_delete" in [tool["name"] for tool in tools]
        and turn_count >= 2
        and len(chunks_in_history_dict) > 1
    )


async def _execute_chunk_delete(
    messages,
    tools,
    llm_api_config,
    turn_count,
    chunks_in_history_dict,
    all_visited_chunks_dict,
):
    """执行chunk_delete工具调用"""
    prefix_prompt = (
        "<think>\n由于可能存在与用户提问无关的检索结果，我需要先分析历史提供的检索结果，"
        "并调用chunk_delete工具删除与用户提问无关的检索结果。\n</think>"
    )

    formatted_chat = format_chat_template(
        messages, tools, add_generation_prompt=True, enable_thinking=True
    )
    logger.info(
        f"[第{turn_count}轮：强制调用chunk_delete工具] LLM输入长度: {len(formatted_chat + prefix_prompt)}"
    )

    start_time = time.time()
    response_text = await generate_response(
        llm_api_config, prompt=formatted_chat + prefix_prompt
    )
    global llm_request_count, llm_input_token_count, llm_output_token_count
    llm_request_count += 1
    llm_input_token_count += len(formatted_chat + prefix_prompt)
    llm_output_token_count += len(response_text)
    end_time = time.time()

    response_text = prefix_prompt + response_text
    content, tool_calls = process_response(response_text)

    # 查找有效的chunk_delete工具调用
    valid_tool_call = None
    for tool_call in tool_calls:
        if (
            tool_call.get("function", {}).get("name", "") == "chunk_delete"
            and len(
                tool_call.get("function", {})
                .get("arguments", {})
                .get("chunk_ids_to_delete", [])
            )
            > 0
        ):
            valid_tool_call = tool_call
            break

    if valid_tool_call is None:
        logger.info(
            f"[第{turn_count}轮：强制调用chunk_delete工具] LLM未调用chunk_delete工具: {response_text}"
        )
        return {"turn_count": turn_count, "tool_calls": []}

    # 执行删除操作
    chunk_ids_to_delete = valid_tool_call["function"]["arguments"][
        "chunk_ids_to_delete"
    ]
    delete_count = _delete_chunks_from_messages(
        messages, chunk_ids_to_delete, chunks_in_history_dict, all_visited_chunks_dict
    )

    # logger.info(f"[第{turn_count}轮：强制调用chunk_delete工具] 保留下来的检索结果（数量：{len(chunks_in_history_dict)}）: \n{list(chunks_in_history_dict.values())}")
    logger.info(
        f"[第{turn_count}轮：强制调用chunk_delete工具] 删除的检索结果（数量：{delete_count}），保留的检索结果（数量：{len(chunks_in_history_dict)}）"
    )
    logger.info(
        f"[第{turn_count}轮：强制调用chunk_delete工具] LLM输出时间: {end_time - start_time}秒"
    )

    return {"turn_count": turn_count, "tool_calls": tool_calls}


def _delete_chunks_from_messages(
    messages, chunk_ids_to_delete, chunks_in_history_dict, all_visited_chunks_dict
):
    """从消息中删除指定的chunk"""
    delete_count = 0

    for msg_idx, message in enumerate(messages):
        if message["role"] != "tool":
            continue

        tool_result = json.loads(message["content"])

        # 某次检索结果中只有一条检索结果，则不删除
        if len(tool_result) <= 1:
            continue

        new_tool_result = []
        for res in tool_result:
            if res.get("chunk_id", "") in chunk_ids_to_delete:
                chunks_in_history_dict.pop(res["chunk_id"], None)
                delete_count += 1
            else:
                new_tool_result.append(res)

        # 每轮搜索至少保留一条检索结果，保留首条检索结果（假设按照相关性排序）
        if len(new_tool_result) == 0:
            new_tool_result = [tool_result[0]]
            remain_chunk_id = tool_result[0]["chunk_id"]
            chunks_in_history_dict[remain_chunk_id] = all_visited_chunks_dict[
                remain_chunk_id
            ]

        messages[msg_idx]["content"] = json.dumps(new_tool_result, ensure_ascii=False)

    return delete_count


def _add_fake_tool_response(messages, turn_count, max_turns):
    """添加FAKE工具响应（用于额外的提示）"""
    if turn_count == max_turns + 1:
        # 此时搜索次数已达到max_turns次，要求直接给出最终答案
        fake_response = [
            {
                "text": "已达到最大搜索次数，无法再调用chunk_search工具，强制直接给出最终答案！！！"
            }
        ]
    elif turn_count > 1:
        # 此时搜索次数已大于1但未达到max_turns次，鼓励继续搜索
        fake_response = [
            {
                "text": "若根据已有检索结果未能给出明确的答案，务必进一步调用chunk_search工具进行搜索！！！"
            }
        ]
    else:
        return

    messages.append(
        {"role": "tool", "content": json.dumps(fake_response, ensure_ascii=False)}
    )


async def _execute_chunk_search(
    query,
    retrieve_config,
    search_engine_url,
    search_count,
    chunks_in_history_dict,
    all_visited_chunks_dict,
):
    """执行chunk搜索"""
    global retrieve_request_count

    tool_response = []

    # 如果是第一轮搜索，额外使用原始query搜索一次做兜底处理
    if search_count == 0:
        raw_tool_response = await chunk_search(
            query=retrieve_config.query,
            retrieve_config=retrieve_config,
            search_engine_url=search_engine_url,
        )
        logger.info(f"raw_tool_response: 检索到{len(raw_tool_response)}条结果")
        tool_response.extend(
            _process_search_results(
                raw_tool_response, chunks_in_history_dict, all_visited_chunks_dict
            )
        )
        retrieve_request_count += 1

    # 执行工具，使用模型指定的query进行搜索
    current_tool_response = await chunk_search(
        query=query,
        retrieve_config=retrieve_config,
        search_engine_url=search_engine_url,
    )
    logger.info(f"current_tool_response: 检索到{len(current_tool_response)}条结果")
    tool_response.extend(
        _process_search_results(
            current_tool_response, chunks_in_history_dict, all_visited_chunks_dict
        )
    )

    retrieve_request_count += 1

    return tool_response, len(current_tool_response)


def _process_search_results(
    search_results, chunks_in_history_dict, all_visited_chunks_dict
):
    """处理搜索结果，去重并更新历史记录"""
    processed_results = []

    for res in search_results:
        # 记录所有访问过的chunk
        all_visited_chunks_dict[res["id"]] = res

        # 检查是否已在历史记录中
        if res["id"] in chunks_in_history_dict:
            logger.warning(f"Warning: chunk_id {res['id']} already in history.")
            continue

        # 检查内容是否重复
        if res["text"] in [
            chunk.get("text", "") for chunk in chunks_in_history_dict.values()
        ]:
            continue

        chunks_in_history_dict[res["id"]] = res
        processed_results.append(res)

    return processed_results


def _format_tool_response(tool_response, current_response_count):
    """格式化工具响应"""
    if len(tool_response) == 0:
        # 如果当前query搜索结果为空，表示前面有重复的检索结果
        if current_response_count == 0:
            return json.dumps(
                [{"text": "未检索到相关文段，建议调整查询策略"}], ensure_ascii=False
            )
        else:
            return json.dumps(
                [{"text": "相关文段在前面均已给出，此处不再重复"}], ensure_ascii=False
            )
    else:
        return format_chunk_texts(tool_response)


def _validate_tool_call(tool_call):
    """验证工具调用的有效性"""
    tool_name = tool_call["function"]["name"]
    arguments = tool_call["function"]["arguments"]

    if tool_name != "chunk_search":
        raise AssertionError(
            f"Warning: tool_name is not chunk_search, but {tool_name}."
        )

    if "query" not in arguments:
        raise AssertionError(f"Warning: query is not in arguments, but {arguments}.")


async def run_conversation_stream(
    request: SearchAgentRequest,
) -> AsyncGenerator[str, None]:
    """运行对话，支持多轮工具调用的流式版本"""
    global retrieve_request_count, llm_request_count, llm_input_token_count, llm_output_token_count
    # 重置相关数据
    retrieve_request_count = 0
    llm_request_count = 0
    llm_input_token_count = 0
    llm_output_token_count = 0

    # 初始化消息列表
    llm_api_config = _resolve_search_agent_llm_config(request.llm_config)
    retrieve_config = request.retrieve_config
    search_engine_url = request.search_engine_url
    query = retrieve_config.query

    tools = [CHUNK_SEARCH_TOOL, CHUNK_DELETE_TOOL]
    max_turns = request.max_turns or DEFAULT_SEARCH_AGENT_CONFIG["max_turns"]

    messages = []
    messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": query})
    logger.info(f"[用户消息] {query}")

    # 初始化状态变量
    turn_count = 0
    search_count = 0
    chunks_in_history_dict = {}
    all_visited_chunks_dict = {}
    chunk_delete_history = []

    while turn_count <= max_turns + 1:
        turn_count += 1

        # 处理chunk_delete工具调用（不流式输出）
        if _should_trigger_chunk_delete(tools, turn_count, chunks_in_history_dict):
            delete_result = await _execute_chunk_delete(
                messages,
                tools,
                llm_api_config,
                turn_count,
                chunks_in_history_dict,
                all_visited_chunks_dict,
            )
            chunk_delete_history.append(delete_result)

        # 添加FAKE工具响应
        _add_fake_tool_response(messages, turn_count, max_turns)

        # 准备LLM输入
        formatted_chat = format_chat_template(messages, [CHUNK_SEARCH_TOOL], add_generation_prompt=True, enable_thinking=True)
        # formatted_chat = format_chat_template(messages, [CHUNK_SEARCH_TOOL], add_generation_prompt=True, enable_thinking=True)
        logger.info(f"[第{turn_count}轮] LLM输入长度: {len(formatted_chat)}")

        full_response_list = []
        in_thinking = False
        in_final_answer = False
        current_chunk_idx = 0
        last_end_think_chunk_idx = -1

        llm_request_count += 1
        llm_input_token_count += len(formatted_chat)
        async for chunk in generate_response_stream(
            llm_api_config, prompt=formatted_chat
        ):
            full_response_list.append(chunk)

            # 以下逻辑用于判断当前chunk是否需要进行输出
            if "<think>" in chunk and not in_thinking:
                in_thinking = True

            if "</think>" in chunk and in_thinking:
                in_thinking = False
                last_end_think_chunk_idx = current_chunk_idx

            if (
                not in_final_answer
                and last_end_think_chunk_idx >= 0
                and current_chunk_idx - last_end_think_chunk_idx > 5
            ):
                pre_final_answer_content = "".join(
                    full_response_list[last_end_think_chunk_idx:current_chunk_idx]
                )
                if "<tool_call>" not in pre_final_answer_content:
                    in_final_answer = True
                    yield pre_final_answer_content

            if in_thinking:
                if "<think>" in chunk:
                    if turn_count == 1:
                        yield chunk
                else:
                    yield chunk.replace("\n\n", "\n")
            elif in_final_answer:
                yield chunk

            current_chunk_idx += 1

        # 处理完整响应以提取工具调用
        full_response = "".join(full_response_list)
        content, tool_calls = process_response(full_response)
        llm_output_token_count += len(full_response)

        # 构建助手消息
        assistant_message = {"role": "assistant", "content": content}

        if tool_calls:
            if len(tool_calls) > 1:
                logger.warning(
                    f"Warning: tool_calls more than 1, only the first one will be used: {tool_calls}！"
                )
            # 只保留第一个工具调用
            tool_calls = [tool_calls[0]]
            assistant_message["tool_calls"] = tool_calls

        messages.append(assistant_message)
        logger.info(
            f"[第{turn_count}轮] LLM输出: \n{json.dumps(assistant_message, ensure_ascii=False, indent=4)}"
        )

        # 如果没有工具调用或达到最大轮数，结束对话
        if not tool_calls or turn_count == max_turns + 1:
            references = pdf_reference_store.resolve_doc_names(
                [chunk.get("doc_name", "") for chunk in chunks_in_history_dict.values()]
            )
            stat_dict = {
                'search_count': search_count,
                'history': messages,
                'reference': [reference.model_dump() for reference in references],
                'retrieve_request_count': retrieve_request_count,
                'llm_request_count': llm_request_count,
                'llm_input_token_count': llm_input_token_count,
                'llm_output_token_count': llm_output_token_count,
            }
            yield f"[END]{json.dumps(stat_dict, ensure_ascii=False)}"
            break

        # 处理工具调用（只处理第一个）
        if tool_calls:
            tool_call = tool_calls[0]
            _validate_tool_call(tool_call)

            # 执行chunk搜索
            sub_query = tool_call["function"]["arguments"]["query"]
            yield f"搜索：[{sub_query}]...\n"
            try:
                tool_response, current_response_count = await _execute_chunk_search(
                    sub_query,
                    retrieve_config,
                    search_engine_url,
                    search_count,
                    chunks_in_history_dict,
                    all_visited_chunks_dict,
                )
                search_count += 1

                # 格式化并添加工具响应
                fmt_tool_response = _format_tool_response(
                    tool_response, current_response_count
                )
            except Exception as exc:
                logger.exception("chunk_search 执行失败: %s", exc)
                tool_response = []
                fmt_tool_response = json.dumps(
                    [
                        {
                            "text": (
                                "chunk_search 执行失败："
                                f"{exc}"
                            )
                        }
                    ],
                    ensure_ascii=False,
                )
            messages.append({"role": "tool", "content": fmt_tool_response})

            # logger.info(f"[第{turn_count}轮] 工具调用结果（数量：{len(tool_response)}）: \n{json.dumps(messages[-1], ensure_ascii=False, indent=4)}")
            logger.info(
                f"[第{turn_count}轮] 工具调用结果长度(共{len(tool_response)}条): {[len(res['text']) for res in tool_response]}"
            )
