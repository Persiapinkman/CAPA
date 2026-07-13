from __future__ import annotations

import json
import re
from typing import Any, AsyncGenerator

from openai import AsyncOpenAI

from gbrain_rag.api.schemas import EvidenceItem, LLMConfig
from gbrain_rag.core.config import get_settings


UNIFIED_QA_PROMPT = """背景：当前语料库均为 Research & Development（简称 RD）部门内部资料，RD 指该团队/部门。
请根据<evidences>中的跨来源检索证据回答问题，严格遵守要求。证据已经按当前问题的回答方面标注为 primary/supporting/caveat。

<evidences>
----------
{evidences}
----------

<要求>
1. 只能依据<evidences>中的信息回答，不要编造不存在的模型、版本、平台、OID、did 或 rid。
2. document 证据表示模型发版文档正文，可能包含 PDF 正文、表格、输入输出、阈值、优化点、追加数据、标签等细节。
3. table 证据表示模型发版信息汇总表，适合负责人、OID、更新时间、推荐配置等结构化字段。
4. adela 证据表示部署记录，适合 did/rid、部署平台、部署状态、部署版本等。
5. 如果证据不足，请回答“抱歉,您提问的相关信息在知识库中没有找到”。
6. 必须使用中文回答，并尽量在句末用 [证据N] 标注来源。
7. 回答字段值问题时，例如“特征维度/维度/OID/平台/组件类型/负责人/推荐配置/支持设备/更新时间/did/rid 是多少/是什么/有哪些”，必须优先查看 payload.field_summary；field_summary 是从表格列按行抽取出的紧凑结果，优先级高于 snippet 中换行错位的原始表格文本。
8. 如果 field_summary 中同一证据列出多个模型族、模型名称或平台的字段值，且问题没有明确限定只问其中一个，答案必须覆盖这些不同取值，不能只回答第一行。
9. 字段值回答要保持“主体 -> 字段 -> 值”的绑定关系，例如“Base224 的特征维度为 512，Large336 的特征维度为 768”，不要把模型名里的数字当成字段值。
10. 回答 release note、优化点、精度、指标、标签、版本变化等问题时，必须优先查看 important_values 和 payload.index_text，保留证据中的原始数字、百分比、指标名、英文 label、模型/版本字符串，不要只做概括或翻译。
11. 有 primary evidence 时，主答案必须优先来自 primary evidence；supporting evidence 只用于补充主体、版本、字段绑定等上下文。
12. caveat evidence 只能作为限制条件或注意事项补充，不能替代主答案；如果没有 primary evidence，不要用 caveat evidence 硬答主问题，应说明证据不足。
13. 只输出最终答案，不要复述“我需要查找/分析/综合来看”等推理过程。

<问题>
{query}

<答案>
"""


KNOWLEDGE_BASE_CONFIDENCE_PROMPT = """背景：当前语料库均为 Research & Development（简称 RD）部门内部资料，RD 指该团队/部门。
请评估给定答案是否已经被知识库证据充分支持，并输出下游服务可直接使用的置信度。

<evidences>
----------
{evidences}
----------

<用户问题>
{query}

<答案>
{answer}

<评分要求>
1. 只评估“知识库证据是否足以全面回答用户问题”，不要评估语言风格。
2. 置信度必须是 0.0 到 1.0 之间的 float。
3. 如果答案完整覆盖问题中的所有关键约束，且关键事实都有 primary/supporting evidence 支持，给 0.85-1.0。
4. 如果答案大体可用但缺少次要约束、存在轻微不确定或只有 supporting evidence 支持，给 0.55-0.84。
5. 如果证据相关但不足以完整回答，或答案只覆盖少部分问题，给 0.20-0.54。
6. 如果没有相关证据、答案明确说未找到、当前未调用 LLM、LLM 调用失败或只是降级片段，给 0.0-0.19。
7. caveat evidence 只能提高限制说明可信度，不能单独支撑主答案。

只输出一个 JSON 对象，不要输出其他内容：
{{"knowledge_base_fully_answered": 0.0}}
"""


STRUCTURED_AGGREGATE_PLAN_PROMPT = """背景：当前语料库均为 Research & Development（简称 RD）部门内部资料，RD 指该团队/部门。
你是结构化统计规划 agent。请判断用户问题是否需要做结构化聚合统计，并规划执行方式。

<可用数据源>
{enabled_sources}

<数据源字段画像>
{source_profiles}

<规划规则>
1. 只有当问题是在问“多少/几个/几款/数量/总数/总共/一共/共有/统计”等全量统计时，should_aggregate 才为 true。
2. 问“RD 部门总共有多少个模型/模型数量”时，RD 是语料库整体背景，不要把 RD 当过滤条件；优先使用 table，按 model_name 去重计数。
3. 问“部署了多少模型/多少部署记录/did/rid/平台”时，优先使用 adela；模型数按 model_name 去重，部署记录数按记录数统计。
4. 如果问题包含业务/项目/方向/模型类别限定、平台或版本条件，必须输出绑定具体字段的 field_conditions。
5. 不要把用户问题直接拆成关键词扫描全表；field_conditions 必须绑定字段，并尽量使用字段画像里出现的规范值或片段。
6. 如果问题不是统计问题，或无法判断统计口径，should_aggregate=false。

<输出要求>
只输出一个 JSON 对象，不要输出其他内容。格式如下：
{{
  "should_aggregate": true/false,
  "reason": "一句话说明规划依据",
  "source_type": "table/adela/null",
  "operation": "count_unique/count_records/null",
  "count_field": "model_name/row_id/name/null",
  "dedupe_field": "model_name/name/null",
  "count_semantics": "model_variant/model_family/deployment_record/record/null",
  "confidence": "high/medium/low",
  "ambiguity": "统计口径是否有歧义；无歧义则写空字符串",
  "condition_logic": "all/any",
  "field_conditions": [
    {{"field": "字段名", "operator": "contains/equals/in", "values": ["字段值或片段"]}}
  ]
}}

<用户问题>
{query}
"""


STRUCTURED_AGGREGATE_ANSWER_PROMPT = """请基于结构化统计计划和执行结果，用简洁中文回答用户问题。

<要求>
1. 不允许改动 execution_result 中的数字。
2. 回答必须说明主结果和统计口径。
3. 如有 ambiguity 或 confidence 不是 high，要简短说明。
4. 只输出最终答案，不要展示推理过程。

<用户问题>
{query}

<统计计划>
{plan_json}

<执行结果>
{result_json}
"""


RETRIEVAL_ROUTE_PLAN_PROMPT = """背景：当前语料库均为 Research & Development（简称 RD）部门内部资料，RD 指该团队/部门。
你是检索路由规划 agent。请只根据用户问题语义，判断应该检索哪些数据源。

<可用数据源>
{enabled_sources}

<数据源说明>
{source_profiles}

<规划要求>
1. 只根据问题语义判断，不要依赖简单关键词匹配规则。
2. 默认策略是返回全部 enabled sources；只有当你能明确判断某些 source 不适用时，才返回子集。
3. 如果问题问正文细节、精度、输入输出、限制、优化、标签、模型文件说明，优先包含 document。
4. 如果问题问 OID、负责人、推荐配置、支持设备、更新时间、模型清单等结构化字段，优先包含 table。
5. 如果问题问 did/rid、部署平台、部署状态、上线版本、部署清单，优先包含 adela。
6. 如果问题存在跨源核对需求，可以同时选择多个 source。
7. 如果问题没有明确指定只看某类 source，或者你无法高置信度排除某个 source，则返回全部 enabled sources。
8. 只有在高置信度确认某些 source 与问题无关或明显不适用时，才返回较小的 source 子集。

<输出要求>
只输出一个 JSON 对象，不要输出其他内容。格式如下：
{{
  "sources": ["document", "table"],
  "reason": "一句话说明依据",
  "confidence": "high/medium/low"
}}

<用户问题>
{query}
"""


QUERY_EXPANSION_PROMPT = """请为企业知识库检索生成少量查询扩展词。

<要求>
1. 只生成可能出现在语料中的检索词、别名、英文缩写、模型/算法相关短语。
2. 不要回答问题，不要生成结论，不要生成长句。
3. 不要编造具体 OID、did、rid、版本号或模型文件名。
4. 保留用户问题中的英文、数字、版本、平台等原始 token。
5. 不要输出过泛化的类别词，例如 model、模型、算法、识别、检测、vision、object detection、YOLO。
6. 如果不确定领域别名，宁可少输出，不要猜测。
7. 最多输出 {max_terms} 个 term。

只输出 JSON：
{{"terms": ["term1", "term2"]}}

<用户问题>
{query}
"""


def _strip_thinking(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    cleaned = re.sub(
        r"^(用户(?:询问|想要|问的是).*?\n+)?(我需要.*?\n+)?",
        "",
        cleaned,
        flags=re.S,
    ).strip()
    return cleaned or (text or "").strip()


def _extract_first_json_object(text: str) -> dict[str, Any]:
    content = _strip_thinking(text)
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.S)
    if fenced:
        content = fenced.group(1)
    start = content.find("{")
    if start < 0:
        raise ValueError("No JSON object found")
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(content)):
        char = content[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
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
                return json.loads(content[start : idx + 1])
    raise ValueError("Unclosed JSON object")


def _confidence_float(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        text = str(value or "").strip()
        if text.endswith("%"):
            number = float(text[:-1]) / 100.0
        else:
            number = float(text)
    except (TypeError, ValueError):
        number = 0.0
    return round(max(0.0, min(1.0, number)), 4)


GENERIC_QUERY_EXPANSION_TERMS = {
    "model",
    "models",
    "模型",
    "算法",
    "识别",
    "检测",
    "分类",
    "推荐",
    "vision",
    "visual",
    "object detection",
    "detection",
    "classification",
    "recognition",
    "algorithm",
    "algorithms",
    "yolo",
    "cnn",
    "resnet",
    "目标检测",
    "图像识别",
    "摄像头",
    "方案",
    "替代方案",
    "能力",
    "深度学习",
    "神经网络",
}


def _resolve_llm_config(llm_config: LLMConfig | None) -> LLMConfig:
    return llm_config or LLMConfig()


async def plan_structured_aggregate(
    *,
    query: str,
    enabled_sources: list[str],
    source_profiles: str,
    llm_config: LLMConfig | None,
) -> dict[str, Any] | None:
    if not enabled_sources:
        return None
    config = _resolve_llm_config(llm_config)
    if not config.api_key or not config.base_url:
        return None
    prompt = STRUCTURED_AGGREGATE_PLAN_PROMPT.format(
        enabled_sources=", ".join(enabled_sources),
        source_profiles=source_profiles,
        query=query,
    )
    client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
    response = await client.chat.completions.create(
        model=config.model or get_settings().LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a strict JSON planning agent."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=min(int(config.max_tokens or 1024), 1024),
        temperature=config.temperature,
        top_p=config.top_p,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    payload = _extract_first_json_object(response.choices[0].message.content or "")
    source_type = str(payload.get("source_type") or "").strip()
    if source_type == "null":
        source_type = ""
    if source_type not in enabled_sources:
        return None
    operation = str(payload.get("operation") or "").strip()
    if operation not in {"count_unique", "count_records"}:
        return None
    raw_conditions = payload.get("field_conditions")
    if not isinstance(raw_conditions, list):
        raw_conditions = []
    field_conditions = []
    for condition in raw_conditions:
        if not isinstance(condition, dict):
            continue
        field = str(condition.get("field") or "").strip()
        operator = str(condition.get("operator") or "contains").strip()
        values = condition.get("values")
        if operator not in {"contains", "equals", "in"}:
            operator = "contains"
        if not isinstance(values, list):
            values = []
        cleaned_values = [str(value).strip() for value in values if str(value).strip()]
        if field and cleaned_values:
            field_conditions.append({"field": field, "operator": operator, "values": cleaned_values})
    confidence = str(payload.get("confidence") or "medium").strip()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"
    condition_logic = str(payload.get("condition_logic") or "all").strip()
    if condition_logic not in {"all", "any"}:
        condition_logic = "all"
    return {
        "should_aggregate": bool(payload.get("should_aggregate")),
        "reason": str(payload.get("reason") or "").strip(),
        "source_type": source_type,
        "operation": operation,
        "count_field": str(payload.get("count_field") or "").strip().replace("null", ""),
        "dedupe_field": str(payload.get("dedupe_field") or "").strip().replace("null", ""),
        "count_semantics": str(payload.get("count_semantics") or "").strip().replace("null", ""),
        "confidence": confidence,
        "ambiguity": str(payload.get("ambiguity") or "").strip(),
        "condition_logic": condition_logic,
        "field_conditions": field_conditions,
    }


async def plan_retrieval_sources(
    *,
    query: str,
    enabled_sources: list[str],
    source_profiles: str,
    llm_config: LLMConfig | None,
) -> dict[str, Any] | None:
    if not enabled_sources:
        return None
    config = _resolve_llm_config(llm_config)
    if not config.api_key or not config.base_url:
        return None
    prompt = RETRIEVAL_ROUTE_PLAN_PROMPT.format(
        enabled_sources=", ".join(enabled_sources),
        source_profiles=source_profiles,
        query=query,
    )
    client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
    response = await client.chat.completions.create(
        model=config.model or get_settings().LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a strict JSON routing planner."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=min(int(config.max_tokens or 512), 512),
        temperature=0.0,
        top_p=0.5,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    payload = _extract_first_json_object(response.choices[0].message.content or "")
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list):
        return None
    sources = [str(source).strip() for source in raw_sources if str(source).strip() in enabled_sources]
    if not sources:
        return None
    confidence = str(payload.get("confidence") or "medium").strip()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"
    return {
        "sources": list(dict.fromkeys(sources)),
        "reason": str(payload.get("reason") or "").strip(),
        "confidence": confidence,
    }


async def answer_structured_aggregate(
    *,
    query: str,
    plan: dict[str, Any],
    result: dict[str, Any],
    llm_config: LLMConfig | None,
) -> str | None:
    config = _resolve_llm_config(llm_config)
    if not config.api_key or not config.base_url:
        return None
    prompt = STRUCTURED_AGGREGATE_ANSWER_PROMPT.format(
        query=query,
        plan_json=json.dumps(plan, ensure_ascii=False, indent=2),
        result_json=json.dumps(result, ensure_ascii=False, indent=2),
    )
    client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
    response = await client.chat.completions.create(
        model=config.model or get_settings().LLM_MODEL,
        messages=[
            {"role": "system", "content": "你是严谨的结构化统计回答助手，只输出最终答案。"},
            {"role": "user", "content": prompt},
        ],
        max_tokens=min(int(config.max_tokens or 768), 768),
        temperature=config.temperature,
        top_p=config.top_p,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return _strip_thinking(response.choices[0].message.content or "")


async def expand_query_with_llm(
    *,
    query: str,
    llm_config: LLMConfig | None,
    max_terms: int | None = None,
) -> list[str]:
    settings = get_settings()
    config = _resolve_llm_config(llm_config)
    if settings.DISABLE_LLM or not config.api_key or not config.base_url:
        return []
    max_terms = max(1, min(int(max_terms or settings.LLM_QUERY_EXPANSION_MAX_TERMS), 16))
    prompt = QUERY_EXPANSION_PROMPT.format(query=query, max_terms=max_terms)
    client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
    response = await client.chat.completions.create(
        model=config.model or settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": "你是严格的查询扩展器，只输出 JSON。"},
            {"role": "user", "content": prompt},
        ],
        max_tokens=256,
        temperature=0.0,
        top_p=0.5,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    payload = _extract_first_json_object(response.choices[0].message.content or "")
    raw_terms = payload.get("terms") or []
    if not isinstance(raw_terms, list):
        return []
    return clean_query_expansion_terms(raw_terms, query=query, max_terms=max_terms)


def clean_query_expansion_terms(raw_terms: list[Any], *, query: str, max_terms: int) -> list[str]:
    terms: list[str] = []
    normalized_query = re.sub(r"\s+", " ", query.lower()).strip()
    compact_query = re.sub(r"\s+", "", normalized_query)
    for term in raw_terms:
        cleaned = re.sub(r"\s+", " ", str(term or "")).strip()
        if not cleaned or cleaned == query or len(cleaned) > 80:
            continue
        if any(mark in cleaned for mark in "{}[]<>"):
            continue
        lowered = cleaned.lower()
        if lowered in GENERIC_QUERY_EXPANSION_TERMS:
            continue
        if re.sub(r"\s+", "", lowered) in compact_query:
            continue
        terms.append(cleaned)
        if len(terms) >= max_terms:
            break
    return list(dict.fromkeys(terms))


def evidence_important_values(evidence: EvidenceItem) -> str:
    return _extract_important_values(evidence)


def format_evidences(evidences: list[EvidenceItem]) -> str:
    if not evidences:
        return "[]"
    grouped: dict[str, list[str]] = {"primary": [], "supporting": [], "caveat": []}
    for idx, evidence in enumerate(evidences, start=1):
        important_values = evidence_important_values(evidence)
        payload_lines = [
            f"{key}: {value}"
            for key, value in evidence.payload.items()
            if value not in (None, "", [], {})
        ]
        lines = [
            f"[证据{idx}]",
            f"source_type: {evidence.source_type}",
            f"title: {evidence.title}",
            f"doc_name: {evidence.doc_name}",
            f"page_label: {evidence.page_label}",
            f"block_type: {evidence.block_type}",
            f"score: {evidence.score}",
            f"evidence_role: {evidence.payload.get('evidence_role') or 'supporting'}",
            f"query_aspect: {evidence.payload.get('query_aspect') or ''}",
            f"section_type: {evidence.payload.get('section_type') or ''}",
            f"aspects: {evidence.payload.get('aspects') or []}",
        ]
        if important_values:
            lines.append(f"important_values: {important_values}")
        lines.extend([f"snippet: {evidence.snippet}", *payload_lines])
        role = str(evidence.payload.get("evidence_role") or "supporting")
        if role not in grouped:
            role = "supporting"
        grouped[role].append("\n".join(lines))
    sections = []
    for role, title in (
        ("primary", "primary_evidence"),
        ("supporting", "supporting_evidence"),
        ("caveat", "caveat_evidence"),
    ):
        content = "\n\n".join(grouped[role]) if grouped[role] else "[]"
        sections.append(f"<{title}>\n{content}\n</{title}>")
    return "\n\n".join(sections)


def _extract_important_values(evidence: EvidenceItem) -> str:
    parts = [
        evidence.snippet,
        str(evidence.payload.get("index_text") or ""),
        str(evidence.payload.get("field_summary") or ""),
    ]
    text = "\n".join(part for part in parts if part)
    if not text:
        return ""

    values: list[str] = []
    role = str(evidence.payload.get("evidence_role") or "")
    query_aspect = str(evidence.payload.get("query_aspect") or "")
    include_quoted_labels = query_aspect in {"label_schema", "release_change"}
    for quoted in re.findall(r"[\"“]([^\"”]{2,80})[\"”]", text):
        if include_quoted_labels and re.search(r"[A-Za-z0-9_./+-]", quoted):
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
    candidates = re.split(r"[。；;\n]|(?:\s+[•◦▪]\s*)", compact)
    for candidate in candidates:
        candidate = candidate.strip(" :-\t")
        lowered = candidate.lower()
        if not candidate or not re.search(r"\d|%", candidate):
            continue
        if query_aspect == "accuracy_metric" and any(token in lowered for token in ("标签解释", "label_", "类别标签")):
            metric_part = re.split(r"标签解释|label_", candidate, maxsplit=1)[0].strip(" :-\t")
            if metric_part and re.search(r"\d|%", metric_part):
                candidate = metric_part
                lowered = candidate.lower()
            else:
                continue
        if not any(hint in lowered for hint in metric_hints):
            continue
        if len(candidate) > 220:
            candidate = candidate[:217] + "..."
        values.append(candidate)

    deduped = list(dict.fromkeys(value for value in values if value))
    return "；".join(deduped[:12])


def fallback_answer(query: str, evidences: list[EvidenceItem]) -> str:
    if not evidences:
        return "抱歉,您提问的相关信息在知识库中没有找到"
    lines = [f"检索到与“{query}”相关的证据，当前未调用 LLM，先给出可核对片段："]
    for idx, evidence in enumerate(evidences[:5], start=1):
        lines.append(f"{idx}. {evidence.title}: {evidence.snippet[:240]} [证据{idx}]")
    return "\n".join(lines)


async def answer_with_llm(query: str, evidences: list[EvidenceItem], llm_config: LLMConfig | None) -> tuple[str, dict[str, Any]]:
    settings = get_settings()
    config = llm_config or LLMConfig()
    config_dict = config.model_dump()
    if settings.DISABLE_LLM or not config.api_key or not config.base_url:
        return fallback_answer(query, evidences), config_dict

    prompt = UNIFIED_QA_PROMPT.format(
        evidences=format_evidences(evidences),
        query=query,
    )
    try:
        client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        extra_body: dict[str, Any] = {}
        if config.seed is not None:
            extra_body["seed"] = config.seed
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
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
        content = response.choices[0].message.content or ""
        return _strip_thinking(content), config_dict
    except Exception as exc:
        answer = fallback_answer(query, evidences)
        answer += f"\n\nLLM 调用失败，已返回检索片段作为降级答案：{exc}"
        return answer, config_dict


async def score_knowledge_base_fully_answered(
    query: str,
    answer: str,
    evidences: list[EvidenceItem],
    llm_config: LLMConfig | None,
) -> float | None:
    settings = get_settings()
    config = llm_config or LLMConfig()
    if settings.DISABLE_LLM or not config.api_key or not config.base_url:
        return None

    prompt = KNOWLEDGE_BASE_CONFIDENCE_PROMPT.format(
        evidences=format_evidences(evidences),
        query=query,
        answer=answer,
    )
    client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
    response = await client.chat.completions.create(
        model=config.model or settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": "你是严格的 RAG 证据充分性评分器，只输出 JSON。"},
            {"role": "user", "content": prompt},
        ],
        max_tokens=256,
        temperature=0.0,
        top_p=0.5,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    payload = _extract_first_json_object(response.choices[0].message.content or "")
    if "knowledge_base_fully_answered" in payload:
        return _confidence_float(payload.get("knowledge_base_fully_answered"))
    if "confidence" in payload:
        return _confidence_float(payload.get("confidence"))
    return None


async def stream_answer_with_llm(
    query: str,
    evidences: list[EvidenceItem],
    llm_config: LLMConfig | None,
) -> AsyncGenerator[str, None]:
    settings = get_settings()
    config = llm_config or LLMConfig()
    if settings.DISABLE_LLM or not config.api_key or not config.base_url:
        yield fallback_answer(query, evidences)
        return

    prompt = UNIFIED_QA_PROMPT.format(
        evidences=format_evidences(evidences),
        query=query,
    )
    client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
    extra_body: dict[str, Any] = {}
    if config.seed is not None:
        extra_body["seed"] = config.seed
    extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    stream = await client.chat.completions.create(
        model=config.model or settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": "你是严谨的企业知识库 RAG 问答助手。只输出最终答案，不展示推理过程。"},
            {"role": "user", "content": prompt},
        ],
        max_tokens=config.max_tokens or settings.LLM_MAX_TOKENS,
        temperature=config.temperature,
        top_p=config.top_p,
        extra_body=extra_body,
        stream=True,
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        content = delta.content if delta else None
        if content:
            yield content
