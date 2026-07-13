from __future__ import annotations

import json
import hashlib
import re
from typing import Any

from openai import AsyncOpenAI

from ace_rag.api.schemas import EvidenceItem, LLMConfig, PlaybookHit, PlaybookItem
from ace_rag.core.config import get_settings
from ace_rag.core.text import dedupe_keep_order


SENSITIVE_LLM_CONFIG_KEYS = {"api_key"}


ACE_QA_PROMPT = """背景：当前语料库均为 Research & Development（简称 RD）部门内部资料，RD 指该团队/部门。
你是 v3 ACE Playbook sidecar 的回答层。请结合 <playbook> 的策略和 <evidences> 的事实证据回答用户问题。

<playbook>
{playbook}
</playbook>

<evidences>
----------
{evidences}
----------
</evidences>

<要求>
1. Playbook 中 section=online_feedback 的条目表示近期用户纠错，可作为高优先级纠错提示；除此之外，Playbook 只提供策略、业务约定、风险提示和检索/回答方法，不能作为具体事实来源。
2. 模型名、版本、OID、did、rid、平台、负责人、推荐配置、指标数值等事实必须来自 <evidences>。
3. 如果 Playbook 与 evidences 冲突，以 evidences 为准；必要时简短说明冲突。
4. 如果证据不足，请回答“抱歉,您提问的相关信息在知识库中没有找到”。
5. 必须使用中文回答，并尽量在句末用 [证据N] 标注来源。
6. 回答字段值问题时，优先查看 payload.field_summary，保持“主体 -> 字段 -> 值”的绑定关系。
7. 回答 release note、优化点、精度、指标、标签、版本变化等问题时，保留证据中的原始数字、百分比、指标名、英文 label、模型/版本字符串。
8. 当问题问“用哪个/应该用哪个/推荐哪个/现用/现在应该用哪个”且没有要求“都要/全部/列表/有哪些”时，只给出一个主推荐；表格证据中如有 count=1、现用、推荐或当前记录，应优先采用该记录，不要把其他候选、历史版本或冲突记录并列为推荐。
9. 如果使用了 Playbook 的业务约定，可附加 [规则:pb-xxxx] 便于审计。
10. 只输出最终答案，不要复述分析过程。

<问题>
{query}

<答案>
"""


ACE_PLAYBOOK_ORGANIZER_PROMPT = """你是 ACE RAG 的 Playbook 记忆整理器。请整理下列 active playbook 条目，将重复、过细、一次性的内容沉淀为更稳定的可复用规则。

<输入条目>
{items}
</输入条目>

<整理目标>
1. 合并重复或高度相似的规则，去除明显冗余。
2. 将 section=online_feedback 的单次纠错抽象成可复用规则；如果纠错只适用于完全相同问题，请保留 source_query 和关键实体，避免过度泛化。
3. 沉淀回答策略、检索路由、字段绑定、查询扩展或聚合语义规则。
4. 不要编造事实。模型名、版本、OID、did、rid、指标数值等事实只能作为检索/回答约束或来自 online_feedback 的纠错语境，不要写成无来源的业务事实。
5. seed/manual 条目可以被合并，但只有在新规则完整覆盖旧规则时才建议退役旧条目。
6. 输出必须是严格 JSON，不要 Markdown，不要解释推理过程。

<输出 JSON schema>
{{
  "items": [
    {{
      "item_id": "可省略或使用 pb-auto-org- 前缀",
      "section": "query_expansion|source_routing|field_binding|answer_strategy|aggregate_semantics",
      "content": "可直接用于 Playbook 的中文规则内容",
      "tags": ["短标签"],
      "source_hints": ["document|table|adela"],
      "query_intents": ["deployment|field_lookup|aggregate|recommendation|version_change|metrics|other"],
      "expansion_terms": ["用于检索增强的关键词"],
      "confidence": 0.0,
      "merged_from": ["被合并的原 item_id"],
      "rationale": "一句话说明合并/沉淀理由"
    }}
  ],
  "retire_item_ids": ["被新规则覆盖、可设为 inactive 的原 item_id"]
}}
"""


ALLOWED_ORGANIZER_SECTIONS = {
    "query_expansion",
    "source_routing",
    "field_binding",
    "answer_strategy",
    "aggregate_semantics",
}
ALLOWED_SOURCE_HINTS = {"document", "table", "adela"}


def strip_thinking(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    cleaned = re.sub(
        r"^(用户(?:询问|想要|问的是).*?\n+)?(我需要.*?\n+)?",
        "",
        cleaned,
        flags=re.S,
    ).strip()
    return cleaned or (text or "").strip()


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = strip_thinking(text)
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned.strip(), flags=re.I | re.S)
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    if start < 0:
        return {}
    depth = 0
    in_string = False
    escape_next = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escape_next:
                escape_next = False
            elif char == "\\":
                escape_next = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(cleaned[start : index + 1])
                    return value if isinstance(value, dict) else {}
                except json.JSONDecodeError:
                    return {}
    return {}


def evidence_important_values(evidence: EvidenceItem) -> str:
    parts = [
        evidence.snippet,
        str(evidence.payload.get("index_text") or ""),
        str(evidence.payload.get("field_summary") or ""),
    ]
    text = "\n".join(part for part in parts if part)
    if not text:
        return ""

    values: list[str] = []
    for quoted in re.findall(r"[\"“]([^\"”]{2,80})[\"”]", text):
        if re.search(r"[A-Za-z0-9_./+-]", quoted):
            values.append(f'"{quoted.strip()}"')

    compact = re.sub(r"\s+", " ", text).strip()
    metric_hints = (
        "acc",
        "accuracy",
        "map",
        "top1",
        "top-1",
        "top5",
        "far",
        "tar",
        "auc",
        "ap",
        "精度",
        "准确",
        "召回",
        "提升",
        "下降",
        "指标",
        "阈值",
        "测试集",
    )
    for candidate in re.split(r"[。；;\n]|(?:\s+[•◦▪]\s*)", compact):
        candidate = candidate.strip(" :-\t")
        lowered = candidate.lower()
        if not candidate or not re.search(r"\d|%", candidate):
            continue
        if not any(hint in lowered for hint in metric_hints):
            continue
        if len(candidate) > 220:
            candidate = candidate[:217] + "..."
        values.append(candidate)
    return "；".join(list(dict.fromkeys(value for value in values if value))[:12])


def format_evidences(evidences: list[EvidenceItem]) -> str:
    if not evidences:
        return "[]"
    blocks = []
    for idx, evidence in enumerate(evidences, start=1):
        important_values = evidence_important_values(evidence)
        payload_lines = [
            f"{key}: {value}"
            for key, value in evidence.payload.items()
            if value not in (None, "", [], {})
        ]
        lines = [
            f"[证据{idx}]",
            f"evidence_id: {evidence.evidence_id}",
            f"source_type: {evidence.source_type}",
            f"title: {evidence.title}",
            f"doc_name: {evidence.doc_name}",
            f"page_label: {evidence.page_label}",
            f"block_type: {evidence.block_type}",
            f"score: {evidence.score}",
        ]
        if important_values:
            lines.append(f"important_values: {important_values}")
        lines.extend([f"snippet: {evidence.snippet}", *payload_lines])
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def format_playbook(items: list[PlaybookHit]) -> str:
    if not items:
        return "[]"
    blocks = []
    for item in items:
        if item.section == "online_feedback":
            provenance = item.provenance or {}
            blocks.append(
                "\n".join(
                    [
                        f"[反馈:{provenance.get('feedback_id') or item.item_id}]",
                        f"section: {item.section}",
                        f"source_query: {provenance.get('source_query')}",
                        f"feedback_type: {provenance.get('feedback_type')}",
                        f"expected_evidence_ids: {json.dumps(provenance.get('expected_evidence_ids') or [], ensure_ascii=False)}",
                        f"corrected_answer: {provenance.get('corrected_answer')}",
                        f"comment: {provenance.get('comment')}",
                        f"content: {item.content}",
                    ]
                )
            )
            continue
        blocks.append(
            "\n".join(
                [
                    f"[规则:{item.item_id}]",
                    f"section: {item.section}",
                    f"confidence: {item.confidence}",
                    f"source_hints: {json.dumps(item.source_hints, ensure_ascii=False)}",
                    f"expansion_terms: {json.dumps(item.expansion_terms, ensure_ascii=False)}",
                    f"content: {item.content}",
                ]
            )
        )
    return "\n\n".join(blocks)


def format_playbook_items_for_organizer(items: list[PlaybookItem]) -> str:
    if not items:
        return "[]"
    blocks: list[str] = []
    for index, item in enumerate(items, start=1):
        provenance = item.provenance or {}
        blocks.append(
            "\n".join(
                [
                    f"[{index}] item_id: {item.item_id}",
                    f"section: {item.section}",
                    f"status: {item.status}",
                    f"confidence: {item.confidence}",
                    f"tags: {json.dumps(item.tags, ensure_ascii=False)}",
                    f"source_hints: {json.dumps(item.source_hints, ensure_ascii=False)}",
                    f"query_intents: {json.dumps(item.query_intents, ensure_ascii=False)}",
                    f"expansion_terms: {json.dumps(item.expansion_terms, ensure_ascii=False)}",
                    f"provenance_source: {provenance.get('source')}",
                    f"source_query: {provenance.get('source_query')}",
                    f"content: {item.content}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _string_list(value: Any, *, limit: int = 24) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            cleaned.append(text[:160])
    return dedupe_keep_order(cleaned)[:limit]


def _bounded_confidence(value: Any, default: float = 0.72) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = default
    return round(max(0.0, min(confidence, 1.0)), 6)


def _safe_llm_config(config: LLMConfig) -> dict[str, Any]:
    data = config.model_dump()
    for key in SENSITIVE_LLM_CONFIG_KEYS:
        if data.get(key):
            data[key] = "***"
    return data


def _auto_playbook_item_id(section: str, content: str, merged_from: list[str]) -> str:
    basis = "|".join([section, content, *merged_from])
    digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]
    return f"pb-auto-org-{digest}"


def _safe_auto_item_id(raw_item_id: Any, section: str, content: str, merged_from: list[str]) -> str:
    item_id = str(raw_item_id or "").strip()
    if item_id.startswith("pb-auto-org-") and re.fullmatch(r"[A-Za-z0-9_.:-]{8,96}", item_id):
        return item_id
    return _auto_playbook_item_id(section, content, merged_from)


def _sanitize_organized_items(
    payload: dict[str, Any],
    source_items: list[PlaybookItem],
) -> tuple[list[PlaybookItem], list[str], dict[str, Any]]:
    source_item_ids = {item.item_id for item in source_items}
    raw_items = payload.get("items") if isinstance(payload.get("items"), list) else []
    raw_retire_ids = payload.get("retire_item_ids") if isinstance(payload.get("retire_item_ids"), list) else []
    new_items: list[PlaybookItem] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        section = str(raw.get("section") or "").strip()
        if section not in ALLOWED_ORGANIZER_SECTIONS:
            continue
        content = str(raw.get("content") or "").strip()
        if len(content) < 12:
            continue
        merged_from = [item_id for item_id in _string_list(raw.get("merged_from"), limit=50) if item_id in source_item_ids]
        item_id = _safe_auto_item_id(raw.get("item_id"), section, content, merged_from)
        source_hints = [
            hint for hint in _string_list(raw.get("source_hints"), limit=8) if hint in ALLOWED_SOURCE_HINTS
        ]
        provenance = {
            "source": "playbook_auto_organizer",
            "merged_from": merged_from,
            "rationale": str(raw.get("rationale") or "").strip(),
        }
        new_items.append(
            PlaybookItem(
                item_id=item_id,
                section=section,
                content=content,
                status="active",
                tags=_string_list(raw.get("tags"), limit=20),
                source_hints=source_hints,
                query_intents=_string_list(raw.get("query_intents"), limit=12),
                expansion_terms=_string_list(raw.get("expansion_terms"), limit=32),
                confidence=_bounded_confidence(raw.get("confidence")),
                provenance=provenance,
            )
        )

    new_item_ids = {item.item_id for item in new_items}
    retire_item_ids = [
        item_id
        for item_id in _string_list(raw_retire_ids, limit=200)
        if item_id in source_item_ids and item_id not in new_item_ids
    ]
    metadata = {
        "organizer_item_count": len(source_items),
        "raw_item_count": len(raw_items),
        "accepted_item_count": len(new_items),
        "requested_retire_count": len(raw_retire_ids),
        "accepted_retire_count": len(retire_item_ids),
    }
    return new_items, retire_item_ids, metadata


def fallback_answer(query: str, evidences: list[EvidenceItem]) -> str:
    if not evidences:
        return "抱歉,您提问的相关信息在知识库中没有找到"
    lines = [f"检索到与“{query}”相关的证据，当前未调用 LLM，先给出可核对片段："]
    for idx, evidence in enumerate(evidences[:5], start=1):
        snippet = re.sub(r"\s+", " ", evidence.snippet or "").strip()
        lines.append(f"{idx}. {evidence.title}: {snippet[:240]} [证据{idx}]")
    return "\n".join(lines)


async def answer_with_llm(
    *,
    query: str,
    evidences: list[EvidenceItem],
    playbook_items: list[PlaybookHit],
    llm_config: LLMConfig | None,
) -> tuple[str, dict[str, Any]]:
    settings = get_settings()
    config = llm_config or LLMConfig()
    config_dict = _safe_llm_config(config)
    if settings.DISABLE_LLM or not config.api_key or not config.base_url:
        return fallback_answer(query, evidences), config_dict

    prompt = ACE_QA_PROMPT.format(
        playbook=format_playbook(playbook_items),
        evidences=format_evidences(evidences),
        query=query,
    )
    try:
        client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        extra_body: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}
        if config.seed is not None:
            extra_body["seed"] = config.seed
        response = await client.chat.completions.create(
            model=config.model or settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是严谨的企业知识库 RAG 问答助手。只输出最终答案，不展示推理过程。"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=config.max_tokens or settings.LLM_MAX_TOKENS,
            temperature=config.temperature,
            top_p=config.top_p,
            extra_body=extra_body,
        )
        return strip_thinking(response.choices[0].message.content or ""), config_dict
    except Exception as exc:
        answer = fallback_answer(query, evidences)
        answer += f"\n\nLLM 调用失败，已返回检索片段作为降级答案：{exc}"
        return answer, config_dict


async def organize_playbook_with_llm(
    *,
    items: list[PlaybookItem],
    llm_config: LLMConfig | None = None,
) -> tuple[list[PlaybookItem], list[str], dict[str, Any]]:
    settings = get_settings()
    config = llm_config or LLMConfig()
    config_dict = _safe_llm_config(config)
    if settings.DISABLE_LLM or not config.api_key or not config.base_url:
        return [], [], {
            "mode": "llm_disabled",
            "llm_config": config_dict,
            "reason": "ACE_RAG_LLM_API_KEY/base_url missing or ACE_RAG_DISABLE_LLM=true",
        }

    prompt = ACE_PLAYBOOK_ORGANIZER_PROMPT.format(items=format_playbook_items_for_organizer(items))
    try:
        client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        extra_body: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}
        if config.seed is not None:
            extra_body["seed"] = config.seed
        response = await client.chat.completions.create(
            model=config.model or settings.LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "你是严谨的企业知识库 Playbook 记忆整理器。只输出严格 JSON。",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=config.max_tokens or settings.LLM_MAX_TOKENS,
            temperature=0.0 if config.temperature is None else min(config.temperature, 0.2),
            top_p=config.top_p,
            extra_body=extra_body,
        )
        raw_text = response.choices[0].message.content or ""
        payload = extract_json_object(raw_text)
        new_items, retire_item_ids, metadata = _sanitize_organized_items(payload, items)
        metadata.update(
            {
                "mode": "llm",
                "llm_config": config_dict,
                "raw_response_chars": len(raw_text),
                "parse_ok": bool(payload),
            }
        )
        return new_items, retire_item_ids, metadata
    except Exception as exc:
        return [], [], {
            "mode": "llm_error",
            "llm_config": config_dict,
            "error": str(exc),
        }
