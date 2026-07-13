import json
import os

import requests


API_URL = os.getenv(
    "GBRAIN_RAG_UNIFIED_QUERY_URL",
    "http://127.0.0.1:6061/api/v1/rag/chat_engine/unified_query",
)


def main() -> None:
    stream = os.getenv("GBRAIN_RAG_STREAM", "").lower() in {"1", "true", "yes", "y"}
    payload = {
        "query": os.getenv("QUERY", "烟火检测有什么模型？"),
        "top_k": 12,
        "retrieval_method": "hybrid",
        "stream": stream,
    }
    if os.getenv("GBRAIN_RAG_EMBEDDING_MODEL"):
        payload["embedding_model"] = os.getenv("GBRAIN_RAG_EMBEDDING_MODEL")
    if os.getenv("GBRAIN_RAG_EMBEDDING_BACKEND"):
        payload["embedding_backend"] = os.getenv("GBRAIN_RAG_EMBEDDING_BACKEND")
    session = requests.Session()
    session.trust_env = False
    response = session.post(API_URL, json=payload, timeout=180, stream=stream)
    response.raise_for_status()
    if stream:
        final_event = None
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ").strip()
            if data == "[DONE]":
                print()
                break
            event = json.loads(data)
            content = event.get("content")
            if content:
                print(content, end="", flush=True)
            if "knowledge_base_fully_answered" in event:
                final_event = event
        if final_event is not None:
            print(f"knowledge_base_fully_answered={final_event['knowledge_base_fully_answered']}")
        return
    body = response.json()
    answer = body.get("answer")
    print(json.dumps(body, ensure_ascii=False, indent=2))
    if isinstance(answer, str) and answer.strip():
        print(answer)


if __name__ == "__main__":
    main()
