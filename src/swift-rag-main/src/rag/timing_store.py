import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict

from src.core.config import get_settings
from src.core.logging import get_logger


logger = get_logger(__name__)
settings = get_settings()


class RAGChatTimingStore:
    """将 document 文档问答耗时以 JSONL 形式落盘，便于后续统计。"""

    def __init__(self, path: str, enabled: bool = True):
        self.path = Path(path)
        self.enabled = enabled
        self._lock = Lock()

    def append(self, payload: Dict[str, Any]) -> None:
        if not self.enabled:
            return

        record = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **payload,
        }

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                with self.path.open("a", encoding="utf-8") as fp:
                    fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("Failed to append RAG chat timing record to %s: %s", self.path, exc)


rag_chat_timing_store = RAGChatTimingStore(
    path=settings.RAG_CHAT_TIMING_LOG_PATH,
    enabled=settings.ENABLE_RAG_CHAT_TIMING_LOG,
)
