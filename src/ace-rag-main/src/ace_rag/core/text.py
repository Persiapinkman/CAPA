from __future__ import annotations

import re
from typing import Iterable


SOURCE_TYPES = {"document", "table", "adela"}


def dedupe_keep_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def normalize_query(text: str) -> str:
    lowered = str(text or "").lower()
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff_./+-]+", " ", lowered)
    return re.sub(r"\s+", " ", normalized).strip()


def tokenize(text: str) -> list[str]:
    lowered = str(text or "").lower()
    tokens = re.findall(r"[a-z0-9][a-z0-9_./+-]{1,80}|[\u4e00-\u9fff]{2,}", lowered)
    return dedupe_keep_order(tokens)


def infer_query_intents(query: str) -> list[str]:
    normalized = str(query or "").lower()
    intents: list[str] = []
    if any(token in normalized for token in ("did", "rid", "部署", "平台", "状态", "adela")):
        intents.append("deployment")
    if any(token in normalized for token in ("oid", "负责人", "owner", "推荐配置", "支持设备", "更新时间")):
        intents.append("release_table")
    if any(token in normalized for token in ("特征维度", "维度", "输入", "输出", "阈值", "指标", "精度", "优化", "标签", "release note")):
        intents.append("document_detail")
    if any(token in normalized for token in ("多少", "几个", "几款", "数量", "统计", "总共", "一共", "共有")):
        intents.append("aggregate")
    if any(token in normalized for token in ("是多少", "是什么", "是谁", "有哪些", "哪个", "哪一个")):
        intents.append("field_lookup")
    return dedupe_keep_order(intents)


def merge_sources(user_sources: list[str] | None, hints: list[str]) -> list[str] | None:
    valid_hints = [source for source in hints if source in SOURCE_TYPES]
    if user_sources:
        return dedupe_keep_order([*user_sources, *valid_hints])
    if valid_hints:
        return dedupe_keep_order(valid_hints)
    return None
