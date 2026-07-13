import json
import os
from pathlib import Path

import requests


API_URL = os.getenv(
    "GBRAIN_RAG_UNIFIED_RETRIEVE_URL",
    "http://127.0.0.1:6061/api/v1/rag/chat_engine/unified_retrieve",
)
OUTPUT_DIR = Path(os.getenv("GBRAIN_RAG_FULL_DOC_OUTPUT_DIR", "sample_code/full_documents_output"))


def safe_filename(value: str) -> str:
    keep = []
    for char in value:
        if char.isalnum() or char in {".", "-", "_"}:
            keep.append(char)
        else:
            keep.append("_")
    name = "".join(keep).strip("_")
    return name[:160] or "document"


def main() -> None:
    payload = {
        "query": os.getenv("QUERY", "安全绳检测 v0.2.1 的输出是什么？"),
        "sources": json.loads(os.getenv("SOURCES", '["document"]')),
        "top_k": int(os.getenv("TOP_K", "8")),
        "retrieval_method": os.getenv("RETRIEVAL_METHOD", "hybrid"),
        "include_full_documents": True,
    }
    if os.getenv("GBRAIN_RAG_EMBEDDING_MODEL"):
        payload["embedding_model"] = os.getenv("GBRAIN_RAG_EMBEDDING_MODEL")
    if os.getenv("GBRAIN_RAG_EMBEDDING_BACKEND"):
        payload["embedding_backend"] = os.getenv("GBRAIN_RAG_EMBEDDING_BACKEND")

    session = requests.Session()
    session.trust_env = False
    response = session.post(API_URL, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()

    print("request:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print()
    print(f"retrieved_count: {data.get('retrieved_count')}")
    print(f"evidences: {len(data.get('evidences') or [])}")
    print(f"full_documents: {len(data.get('full_documents') or [])}")
    print(f"response_has_full_documents_key: {'full_documents' in data}")
    print()

    doc_ids = {}
    for evidence in data.get("evidences") or []:
        doc_id = evidence.get("doc_id")
        if not doc_id:
            continue
        doc_ids.setdefault(doc_id, {"count": 0, "doc_name": evidence.get("doc_name")})
        doc_ids[doc_id]["count"] += 1

    if doc_ids:
        print("matched evidence doc_ids:")
        for doc_id, detail in doc_ids.items():
            print(f"- {doc_id}: {detail['count']} evidence(s), doc_name={detail['doc_name']}")
        print()
    else:
        print("matched evidence doc_ids: none")
        print()

    if "full_documents" not in data:
        print(
            "warning: response does not contain `full_documents`; "
            "the API service is likely still running old code. Restart the server and retry."
        )
        print()
    elif data.get("evidences") and not data.get("full_documents"):
        print(
            "warning: evidences were returned but `full_documents` is empty; "
            "check whether the running server includes the new include_full_documents implementation."
        )
        print()

    for idx, doc in enumerate(data.get("full_documents") or [], start=1):
        content = doc.get("content") or ""
        print(f"[full_document {idx}]")
        print(f"doc_id: {doc.get('doc_id')}")
        print(f"doc_name: {doc.get('doc_name')}")
        print(f"source_type: {doc.get('source_type')}")
        print(f"source_path: {doc.get('source_path')}")
        print(f"chunk_count: {doc.get('chunk_count')}")
        print(f"content_chars: {len(content)}")
        print("matched_evidence_ids:", doc.get("metadata", {}).get("matched_evidence_ids", []))
        print("content_preview:")
        print(content[:1200])
        print()

    if os.getenv("SAVE_FULL_DOCUMENTS", "1") != "0":
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for idx, doc in enumerate(data.get("full_documents") or [], start=1):
            doc_name = safe_filename(str(doc.get("doc_name") or doc.get("doc_id") or f"document_{idx}"))
            output_path = OUTPUT_DIR / f"{idx:02d}_{doc_name}.txt"
            output_path.write_text(doc.get("content") or "", encoding="utf-8")
            print(f"saved: {output_path}")

        response_path = OUTPUT_DIR / "response.json"
        response_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved: {response_path}")


if __name__ == "__main__":
    main()
