import json
import os

import requests


API_URL = os.getenv(
    "GBRAIN_RAG_UNIFIED_RETRIEVE_URL",
    "http://127.0.0.1:6061/api/v1/rag/chat_engine/unified_retrieve",
)


def main() -> None:
    payload = {
        "query": os.getenv("QUERY", "安全绳检测有什么模型"),
        "top_k": 8,
        "retrieval_method": "hybrid",
    }
    if os.getenv("GBRAIN_RAG_EMBEDDING_MODEL"):
        payload["embedding_model"] = os.getenv("GBRAIN_RAG_EMBEDDING_MODEL")
    if os.getenv("GBRAIN_RAG_EMBEDDING_BACKEND"):
        payload["embedding_backend"] = os.getenv("GBRAIN_RAG_EMBEDDING_BACKEND")
    session = requests.Session()
    session.trust_env = False
    response = session.post(API_URL, json=payload, timeout=120)
    response.raise_for_status()
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    print()


if __name__ == "__main__":
    main()
