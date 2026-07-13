#!/usr/bin/env python3
"""
测试streaming_agent.py的实时流式输出功能
"""

import asyncio
import sys
import time
import json
import aiohttp
from typing import List
from pathlib import Path

# 使用pathlib更优雅地处理路径
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_dir))

async def test_function_streaming():
    """测试实时流式输出功能"""
    print("开始测试实时流式输出...")

    # 模拟一个简单的查询
    query = "《昆明市呈贡区国有企业退休人员社会化管理工作实施方案》、《昆明市呈贡区经营主体集群注册登记管理办法（试行）》、《昆明市呈贡区推动花卉产业高质量发展的“花十条”措施（试行）》，哪些是同一年发布的？"
    # query = "昆明市呈贡区食品安全事故应预案和大面积停电事件应急预案哪一个发布的时间更早？"
    query = "2021到2023年三年来昆明市累计共接待国内外游客多少人？"
    query = "2021到2023年三年来昆明市交通事故死亡人数平均每年是多少？"

    # 构建请求参数
    request = {
        "search_engine_url": "http://10.151.35.34:19068",
        "retrieve_config": {
            "query": query,
            "top_k": 5,
            "similarity_threshold": 0,
            "uri": "http://10.151.35.34:19530",
            "collection_name": "INFO_CHUNK_VECTOR_10010",
            "embedding_models": ["EvoQwen2.5-VL-Retriever-3B-v1"]
        },
        "llm_config": {
            "model": "Qwen3.5-4B",
            "base_url": "http://10.111.32.253:8000/v1",
            "api_key": "token.sdc@2026",
            "max_tokens": 2048,
            "temperature": 0.001,
            "top_p": 0.001
        },
        "max_turns": 5,
    }

    output_parts: List[str] = []
    start_time = time.time()

    try:
        # 导入需要的模块
        from src.rag.search_agent.agent import run_conversation_stream
        from src.api.schemas import SearchAgentRequest

        # 将请求参数转换为SearchAgentRequest对象
        search_request = SearchAgentRequest(**request)

        print(f"开始流式输出，查询内容: {query}")
        print("-" * 50)

        async for chunk in run_conversation_stream(search_request):
            output_parts.append(chunk)
            print(chunk, end="", flush=True)

    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return

    end_time = time.time()

    # 合并所有输出
    full_output = "".join(output_parts)
    print("\n" + "=" * 50)
    print(f"完整输出:\n{full_output}")
    print(f"\n总耗时: {end_time - start_time:.2f}秒")
    print(f"输出块数量: {len(output_parts)}")

async def test_api_streaming():
    """通过API接口测试流式输出功能"""
    print("开始测试API流式输出...")
    test_url = "http://10.151.35.34:12068/api/v1/rag/chat_engine/search_agent_stream_query"

    # 模拟一个简单的查询
    query = "2021到2023年三年来昆明市交通事故死亡人数平均每年是多少？"

    # 构建请求参数
    request = {
        "retrieve_config": {
            "query": query,
            "top_k": 5,
            "similarity_threshold": 0,
            "uri": "http://10.151.35.34:19530",
            "collection_name": "INFO_CHUNK_VECTOR_10010",
            "embedding_models": ["EvoQwen2.5-VL-Retriever-3B-v1"]
        },
        "llm_config": {
            "model": "Qwen3.5-4B",
            "base_url": "http://10.111.32.253:8000/v1",
            "api_key": "token.sdc@2026",
            "max_tokens": 2048,
            "temperature": 0.001,
            "top_p": 0.001
        },
        "search_engine_url": "http://10.151.35.34:12068",
        "max_turns": 5,
    }

    output_parts: List[str] = []
    start_time = time.time()

    search_count = 0
    history = None
    try:
        print(f"开始API流式输出，查询内容: {query}")
        print("-" * 50)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                test_url,
                json=request
            ) as response:
                async for line in response.content:
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            if '[DONE]' in line:
                                break
                            data = json.loads(line[6:])
                            chunk = data.get('content', '')
                            output_parts.append(chunk)
                            print(chunk, end="", flush=True)
                            search_count = data.get('search_count', 0)
                            history = data.get('history', None)

    except Exception as e:
        print(f"\nAPI测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return

    end_time = time.time()

    # 合并所有输出
    full_output = "".join(output_parts)
    print("\n" + "=" * 50)
    print(f"完整输出:\n{full_output}")
    print(f"\n总耗时: {end_time - start_time:.2f}秒")
    print(f"输出token数量: {len(output_parts)}")
    print(f"搜索次数: {search_count}")
    # print(f"历史: {history}")

if __name__ == "__main__":
    # asyncio.run(test_real_time_streaming())
    # asyncio.run(test_function_streaming())
    asyncio.run(test_api_streaming())
