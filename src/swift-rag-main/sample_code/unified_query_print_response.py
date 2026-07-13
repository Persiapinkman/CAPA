import json
import os

import requests


def _default_unified_query_api_url() -> str:
    custom_url = os.getenv("SWIFT_RAG_UNIFIED_QUERY_API_URL")
    if custom_url:
        return custom_url

    api_base = os.getenv("SWIFT_RAG_API_BASE_URL")
    if api_base:
        return f"{api_base.rstrip('/')}/rag/chat_engine/unified_query"

    return "http://127.0.0.1:6060/api/v1/rag/chat_engine/unified_query"


def main() -> None:
    url = _default_unified_query_api_url()
    payload = {
        "query": "人脸识别模型1:N和1:n的区别，各自的测试精度是怎么样的？"
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
    except requests.RequestException as exc:
        print("Request failed:", exc)
        return

    print("Request URL:", url)
    print("Status Code:", response.status_code)
    print("Response Headers:")
    print(json.dumps(dict(response.headers), ensure_ascii=False, indent=2))
    print("-" * 80)
    print("Raw Response Text:")
    print(response.text)
    print("-" * 80)

    if response.status_code == 502:
        print(
            "Hint: 502 Bad Gateway 通常是网关到后端服务不可达。"
            "\n建议先探活: curl http://127.0.0.1:6060/openapi.json"
            "\n如果你必须走远端地址，请检查反向代理和后端服务连通性。"
        )
        print("-" * 80)

    try:
        data = response.json()
        print("Parsed JSON:")
        print(json.dumps(data, ensure_ascii=False, indent=2))
    except ValueError:
        print("Response is not valid JSON.")


if __name__ == "__main__":
    main()
