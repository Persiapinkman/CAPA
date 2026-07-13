from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ace_rag.api.schemas import PlaybookItem


DEFAULT_PLAYBOOK_ITEMS = [
    {
        "item_id": "pb-source-0001",
        "section": "source_routing",
        "content": "询问 did/rid、部署平台、部署状态、部署版本时，必须包含 adela 数据源；如果同时询问 OID、负责人或推荐配置，也包含 table。",
        "source_hints": ["adela", "table"],
        "query_intents": ["deployment", "field_lookup"],
        "expansion_terms": ["did", "rid", "deployment_id"],
        "tags": ["deployment", "adela", "did", "rid"],
        "confidence": 0.96,
    },
    {
        "item_id": "pb-source-0002",
        "section": "source_routing",
        "content": "询问 OID、负责人、更新时间、推荐配置、支持设备等发版汇总字段时，优先包含 table 数据源。",
        "source_hints": ["table"],
        "query_intents": ["release_table", "field_lookup"],
        "expansion_terms": ["oid", "owner", "recommended_config", "supported_device"],
        "tags": ["table", "release", "owner", "oid"],
        "confidence": 0.94,
    },
    {
        "item_id": "pb-source-0003",
        "section": "source_routing",
        "content": "询问 release note、优化点、指标、阈值、输入输出、标签解释、特征维度等文档细节时，必须包含 document 数据源。",
        "source_hints": ["document"],
        "query_intents": ["document_detail", "field_lookup"],
        "expansion_terms": ["release note", "field_summary", "important_values"],
        "tags": ["document", "release-note", "metric"],
        "confidence": 0.94,
    },
    {
        "item_id": "pb-field-0001",
        "section": "field_binding",
        "content": "回答字段值问题时，优先查看证据 payload.field_summary；如果同一证据中出现多个模型族、模型名称或平台，必须保持主体到字段值的绑定关系，不能只回答第一行。",
        "source_hints": ["document", "table", "adela"],
        "query_intents": ["field_lookup"],
        "expansion_terms": ["field_summary"],
        "tags": ["field-binding", "field_summary"],
        "confidence": 0.98,
    },
    {
        "item_id": "pb-answer-0001",
        "section": "answer_strategy",
        "content": "Playbook 只提供回答策略、业务约定和风险提示；模型名、OID、did、rid、平台、版本、指标数值等事实必须来自 evidences。证据不足时回答知识库中没有找到。",
        "source_hints": [],
        "query_intents": ["field_lookup", "deployment", "document_detail", "aggregate"],
        "expansion_terms": [],
        "tags": ["grounding", "no-hallucination"],
        "confidence": 0.99,
    },
    {
        "item_id": "pb-answer-0002",
        "section": "answer_strategy",
        "content": "如果答案使用了 Playbook 中的业务约定，可以在句末附加 [规则:pb-xxxx]；但证据标注 [证据N] 仍然必须保留。",
        "source_hints": [],
        "query_intents": ["field_lookup", "deployment", "document_detail"],
        "expansion_terms": [],
        "tags": ["citation", "audit"],
        "confidence": 0.85,
    },
    {
        "item_id": "pb-aggregate-0001",
        "section": "aggregate_semantics",
        "content": "问“多少个/几款/数量/统计”时，需要区分模型记录数、模型名称去重数和部署记录数；RD 是语料库背景，不应直接作为过滤条件。",
        "source_hints": ["table", "adela"],
        "query_intents": ["aggregate"],
        "expansion_terms": ["model_name", "deployment_record"],
        "tags": ["aggregate", "counting"],
        "confidence": 0.92,
    },
    {
        "item_id": "pb-alias-0001",
        "section": "query_expansion",
        "content": "安全绳相关问题通常需要同时检索 safety_rope；安全带/反光衣相关问题通常需要检索 safetybelt、safety_belt、waistcoat、PAR_waistcoat_safetybelt。",
        "source_hints": ["document", "table", "adela"],
        "query_intents": ["deployment", "field_lookup", "document_detail"],
        "expansion_terms": ["safety_rope", "safetybelt", "safety_belt", "waistcoat", "PAR_waistcoat_safetybelt"],
        "tags": ["alias", "safety"],
        "confidence": 0.9,
    },
]


def load_seed_items(path: Path | None = None) -> list[PlaybookItem]:
    if path and path.exists():
        return _load_jsonl(path)
    return [PlaybookItem(**item, provenance={"source": "default_seed"}) for item in DEFAULT_PLAYBOOK_ITEMS]


def _load_jsonl(path: Path) -> list[PlaybookItem]:
    items: list[PlaybookItem] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            payload: dict[str, Any] = json.loads(stripped)
            payload.setdefault("provenance", {"source": str(path), "line": line_no})
            items.append(PlaybookItem(**payload))
    return items
