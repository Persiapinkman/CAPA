import json
import re
from typing import AsyncGenerator, Dict, List, Tuple

from openai import AsyncOpenAI

from src.api.schemas import LLMConfig
from src.core.logging import get_logger
from src.rag.search_agent.config import DEFAULT_LLM_API_CONFIG

logger = get_logger(__name__)


async def generate_response_stream(
    llm_api_config: LLMConfig, prompt: str = ""
) -> AsyncGenerator[str, None]:
    """
    异步流式调用模型API生成响应
    """
    default_config = DEFAULT_LLM_API_CONFIG
    client = AsyncOpenAI(
        base_url=llm_api_config.base_url or default_config["base_url"],
        api_key=llm_api_config.api_key or default_config["api_key"],
    )

    try:
        request_kwargs = {
            "model": llm_api_config.model or default_config["model"],
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "max_tokens": llm_api_config.max_tokens or default_config["max_tokens"],
        }
        if llm_api_config.temperature is not None:
            request_kwargs["temperature"] = llm_api_config.temperature
        elif default_config["temperature"] is not None:
            request_kwargs["temperature"] = default_config["temperature"]

        if llm_api_config.top_p is not None:
            request_kwargs["top_p"] = llm_api_config.top_p
        elif default_config["top_p"] is not None:
            request_kwargs["top_p"] = default_config["top_p"]

        if llm_api_config.seed is not None:
            request_kwargs["seed"] = int(llm_api_config.seed)
        elif default_config.get("seed") is not None:
            request_kwargs["seed"] = int(default_config["seed"])

        response = await client.chat.completions.create(**request_kwargs)

        async for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = delta.content if delta else None
            if content:
                yield content
    except Exception as e:
        logger.error(f"流式生成响应时出错: {e}")
        yield f"错误: {str(e)}"


async def generate_response(llm_api_config: LLMConfig, prompt: str = "") -> str:
    """
    异步调用模型API生成完整响应
    """
    default_config = DEFAULT_LLM_API_CONFIG
    client = AsyncOpenAI(
        base_url=llm_api_config.base_url or default_config["base_url"],
        api_key=llm_api_config.api_key or default_config["api_key"],
    )

    request_kwargs = {
        "model": llm_api_config.model or default_config["model"],
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": llm_api_config.max_tokens or default_config["max_tokens"],
    }
    if llm_api_config.temperature is not None:
        request_kwargs["temperature"] = llm_api_config.temperature
    elif default_config["temperature"] is not None:
        request_kwargs["temperature"] = default_config["temperature"]

    if llm_api_config.top_p is not None:
        request_kwargs["top_p"] = llm_api_config.top_p
    elif default_config["top_p"] is not None:
        request_kwargs["top_p"] = default_config["top_p"]

    if llm_api_config.seed is not None:
        request_kwargs["seed"] = int(llm_api_config.seed)
    elif default_config.get("seed") is not None:
        request_kwargs["seed"] = int(default_config["seed"])

    response = await client.chat.completions.create(**request_kwargs)

    return response.choices[0].message.content or ""


def format_chat_template(
    messages, tools=None, add_generation_prompt=False, enable_thinking=True
):
    """
    将消息历史格式化为特定的聊天模板格式
    """
    result = []

    # 处理工具部分
    if tools:
        result.append("<|im_start|>system\n")

        # 如果第一条消息是系统消息，添加到开头
        if messages and messages[0]["role"] == "system":
            result.append(messages[0]["content"] + "\n\n")

        # 添加工具描述
        result.append(
            "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        )
        result.append(
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
        )

        for tool in tools:
            result.append(
                "\n"
                + json.dumps({"type": "function", "function": tool}, ensure_ascii=False)
            )

        result.append("\n</tools>\n\n")
        result.append(
            "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
        )
        result.append(
            '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n'
        )
    else:
        # 如果没有工具但有系统消息
        if messages and messages[0]["role"] == "system":
            result.append(
                "<|im_start|>system\n" + messages[0]["content"] + "<|im_end|>\n"
            )

    # 确定最后一个用户查询的索引
    multi_step_tool = True
    last_query_index = len(messages) - 1

    for i in range(len(messages) - 1, -1, -1):
        message = messages[i]
        if (
            multi_step_tool
            and message["role"] == "user"
            and not (
                message["content"].startswith("<tool_response>")
                and message["content"].endswith("</tool_response>")
            )
        ):
            multi_step_tool = False
            last_query_index = i
            break

    # 处理所有消息
    for i, message in enumerate(messages):
        # 处理用户消息或非首条系统消息
        if message["role"] == "user" or (message["role"] == "system" and i > 0):
            result.append(
                f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            )

        # 处理助手消息
        elif message["role"] == "assistant":
            content = message["content"]
            reasoning_content = ""

            # 提取思考内容
            if (
                "reasoning_content" in message
                and message["reasoning_content"] is not None
            ):
                reasoning_content = message["reasoning_content"]
            elif "</think>" in content:
                parts = content.split("</think>")
                content = parts[-1].lstrip("\n")
                reasoning_content = (
                    parts[0].split("<think>")[-1].lstrip("\n").rstrip("\n")
                )

            # 格式化助手回复
            if i > last_query_index:
                if i == len(messages) - 1 or reasoning_content:
                    result.append(
                        f"<|im_start|>{message['role']}\n<think>\n{reasoning_content.strip()}\n</think>\n\n{content.lstrip()}"
                    )
                else:
                    result.append(f"<|im_start|>{message['role']}\n{content}")
            else:
                result.append(f"<|im_start|>{message['role']}\n{content}")

            # 处理工具调用
            if "tool_calls" in message and message["tool_calls"]:
                for j, tool_call in enumerate(message["tool_calls"]):
                    if (j == 0 and content) or j > 0:
                        result.append("\n")

                    if "function" in tool_call:
                        tool_call = tool_call["function"]

                    result.append(
                        f"<tool_call>\n{{\"name\": \"{tool_call['name']}\", \"arguments\": "
                    )

                    if isinstance(tool_call["arguments"], str):
                        result.append(tool_call["arguments"])
                    else:
                        result.append(
                            json.dumps(tool_call["arguments"], ensure_ascii=False)
                        )

                    result.append("}\n</tool_call>")

            result.append("<|im_end|>\n")

        # 处理工具消息
        elif message["role"] == "tool":
            if i == 0 or messages[i - 1]["role"] != "tool":
                result.append("<|im_start|>user")

            result.append("\n<tool_response>\n")
            result.append(message["content"])
            result.append("\n</tool_response>")

            if i == len(messages) - 1 or messages[i + 1]["role"] != "tool":
                result.append("<|im_end|>\n")

    # 添加生成提示
    if add_generation_prompt:
        result.append("<|im_start|>assistant\n")
        if enable_thinking is False:
            result.append("<think>\n\n</think>\n\n")

    return "".join(result)


def process_response(response_text: str) -> Tuple[str, List[Dict]]:
    """处理模型返回的响应，提取工具调用信息"""
    content = response_text
    tool_calls = []

    # 提取工具调用
    tool_call_pattern = r"<tool_call>\s*({.*?})\s*</tool_call>"
    tool_call_matches = re.findall(tool_call_pattern, response_text, re.DOTALL)

    for match in tool_call_matches:
        try:
            tool_call_data = json.loads(match)
            tool_calls.append(
                {
                    "function": {
                        "name": tool_call_data["name"],
                        "arguments": tool_call_data["arguments"],
                    }
                }
            )
        except json.JSONDecodeError:
            logger.warning(f"解析工具调用失败: {match}")
            continue

    # 清理响应中的工具调用部分
    if tool_calls:
        for match in tool_call_matches:
            content = content.replace(f"<tool_call>\n{match}\n</tool_call>", "").strip()

    return content, tool_calls
