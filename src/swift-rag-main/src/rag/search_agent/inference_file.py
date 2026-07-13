#!/usr/bin/env python3
"""
处理jsonl文件，逐个调用API获取流式输出结果
"""

import json
import time
import re
import requests
from typing import List, Dict

from src.core.config import get_settings
from src.rag.search_agent.config import DEFAULT_LLM_API_CONFIG

settings = get_settings()

def remove_think_tags(text: str) -> str:
    """移除<think>xxx</think>之间的内容"""
    pattern = r'<think>.*?</think>'
    return re.sub(pattern, '', text, flags=re.DOTALL).strip()

def call_streaming_api(question: str) -> Dict[str, str]:
    """调用流式API获取结果"""
    api_url = (
        f"http://127.0.0.1:{settings.PORT}"
        f"{settings.API_V1_STR}/rag/chat_engine/search_agent_stream_query"
    )

    # 构建请求参数
    request_data = {
        "retrieve_config": {
            "query": question,
            "top_k": 5,
            "similarity_threshold": 0,
            "uri": settings.DATA_SOURCE_VECTOR_DB_URI,
            "collection_name": settings.DATA_SOURCE_COLLECTION_NAME,
            "embedding_models": settings.DEFAULT_EMBEDDING_MODELS,
        },
        "llm_config": {
            "model": DEFAULT_LLM_API_CONFIG["model"],
            "base_url": DEFAULT_LLM_API_CONFIG["base_url"],
            "api_key": DEFAULT_LLM_API_CONFIG["api_key"],
            "max_tokens": DEFAULT_LLM_API_CONFIG["max_tokens"],
            # "temperature": 0.001,
            # "top_p": 0.001
        },
        "max_turns": 5,
    }

    output_parts: List[str] = []
    search_count = 0
    history = None
    stat_dict = None

    start_time = time.time()
    first_token_time = None
    answer_first_token_time = None

    try:
        print(f"正在处理问题: {question}")

        # 发送POST请求
        response = requests.post(api_url, json=request_data, stream=True)
        response.raise_for_status()

        # 处理流式响应
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    if first_token_time is None:
                        first_token_time = time.time() - start_time
                    if '[DONE]' in line:
                        break
                    try:
                        data = json.loads(line[6:])
                        chunk = data.get('content', '')
                        output_parts.append(chunk)
                        print(chunk, end="", flush=True)
                        search_count = data.get('stat', {}).get('search_count', 0)
                        history = data.get('stat', {}).get('history', None)
                        stat_dict = data.get('stat', None)
                        if '</think>' in chunk and answer_first_token_time is None:
                            answer_first_token_time = time.time() - start_time
                    except json.JSONDecodeError:
                        continue
        print()  # 换行


    except Exception as e:
        print(f"API调用出错: {e}")
        return {
            "full_answer": f"错误: {str(e)}",
            "clean_answer": f"错误: {str(e)}",
            "success": False,
            "search_count": search_count,
            "history": history
        }

    # 合并所有输出
    full_answer = "".join(output_parts)
    clean_answer = remove_think_tags(full_answer)

    result_dict = {
        "full_answer": full_answer,
        "clean_answer": clean_answer,
        "success": True,
        "search_count": search_count,
        "history": history,
        "first_token_time": first_token_time,
        "answer_first_token_time": answer_first_token_time,
        "e2e_time": time.time() - start_time,
        "answer_len": len(clean_answer),
    }
    if stat_dict:
        result_dict.update(stat_dict)
    print(result_dict)
    return result_dict

def process_jsonl_file(input_file: str, output_file: str):
    """处理jsonl文件"""
    print(f"开始处理文件: {input_file}")

    results = []

    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"共找到 {len(lines)} 个问题")

    # 逐个处理每一行
    for i, line in enumerate(lines):
        # 解析JSON
        data = json.loads(line.strip())
        question = data.get("问题", "")

        if not question:
            print(f"第 {i} 行没有找到'问题'字段，跳过")
            continue

        print(f"\n处理第 {i}/{len(lines)} 个问题...")
        print("-" * 50)

        # 调用API
        start_time = time.time()
        api_result = call_streaming_api(question)
        end_time = time.time()

        # 构建结果
        data.update(api_result)
        data["process_time"] = f"{end_time - start_time:.2f}"

        results.append(data)

        print(f"完成，耗时: {end_time - start_time:.2f}秒")
        print("=" * 50)

        # 每处理一个就保存一次，防止中途出错丢失数据
        with open(output_file, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"\n处理完成！共处理 {len(results)} 个问题")
    print(f"结果已保存到: {output_file}")

def main():
    """主函数"""
    # 配置输入输出文件路径
    input_file = "/home/linhuifeng/Agentic_RAG/chenggong/data/test_v013_200_update0611.jsonl"  # 输入文件路径
    output_file = "test_v013_200_update0611_delete_result.jsonl"   # 输出文件路径

    # 处理文件
    process_jsonl_file(input_file, output_file)

if __name__ == "__main__":
    main()
