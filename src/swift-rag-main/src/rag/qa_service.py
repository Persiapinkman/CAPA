import json
import re
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Tuple

from openai import AsyncOpenAI

from src.api.schemas import ChunkNodeWithScore, LLMConfig, TableMatchedRow, UnifiedEvidenceItem
from src.rag.search_agent.config import DEFAULT_LLM_API_CONFIG


RAG_QA_PROMPT = """背景：当前语料库均为 Research & Development（简称 RD）部门内部资料，RD 指该团队/部门。
请根据<context>中的内容，回答下面的问题，注意不要输出<context>以外的内容。

<context>
----------
{ctx_chunks}
----------

<要求>
1. 不要输出<context>以外的内容。若<context>中的内容和问题无关，请回答“抱歉,您提问的相关信息在知识库中没有找到”。
2. 请依据给定<context>合理推理，并尽量完整引用与问题直接相关的条目、条款、内容。
3. 不要输出“根据已知信息”“根据<context>”之类的提示性字段。

<问题>
{query}

<答案>
"""


TABLE_QA_PROMPT = """背景：当前语料库均为 Research & Development（简称 RD）部门内部资料，RD 指该团队/部门。
请根据<table_rows>中的结构化行内容回答问题，严格遵守要求。

<table_rows>
----------
{table_rows}
----------

<要求>
1. 只能依据<table_rows>中的内容回答，不要编造不存在的模型、OID、负责人、配置或时间。
2. 如果问题是在问“有哪些模型”“列出模型”等枚举类问题，请尽量完整列出 `model_name`。
3. 如果命中结果不足以回答，请明确回答“抱歉,您提问的相关信息在知识库中没有找到”。
4. 不要输出“根据表格”“根据已知信息”这类提示语。

<问题>
{query}

<答案>
"""

ADELA_QA_PROMPT = """背景：当前语料库均为 Research & Development（简称 RD）部门内部资料，RD 指该团队/部门。
请根据<adela_records>中的结构化部署记录回答问题，严格遵守要求。

<adela_records>
----------
{adela_records}
----------

<要求>
1. 只能依据<adela_records>中的内容回答，不要编造不存在的 did、rid、平台、版本或状态。
2. 如果问题是“有哪些模型/部署”，请尽量完整列出 `model_name`（或 `name`），并在可用时附带 `platform` 和 `version`。
3. 如果问题在问某个模型的详细信息，优先引用该行中的 `model_info`。
4. 如果命中结果不足以回答，请明确回答“抱歉,您提问的相关信息在知识库中没有找到”。
5. 不要输出“根据表格”“根据已知信息”这类提示语。

<问题>
{query}

<答案>
"""

UNIFIED_QA_PROMPT = """背景：当前语料库均为 Research & Development（简称 RD）部门内部资料，RD 指该团队/部门。
请根据<evidences>中的跨来源检索证据回答问题，严格遵守要求。

<evidences>
----------
{evidences}
----------

<要求>
1. 只能依据<evidences>中的信息回答，不要编造不存在的模型、版本、平台、OID、did 或 rid。
2. `document` 证据表示模型发版文档的正文内容，来源于 ONES 工作文档或对应发版 PDF，不是简单的摘要标题；其中常包含输入输出、阈值、优化点、追加数据、标签、背景说明、正文/table/image 细节。
3. 如果问题涉及“是否一致/有什么差异”，请先归纳来源一致项，再指出差异项。
4. 如证据不足，请明确回答“抱歉,您提问的相关信息在知识库中没有找到”。
5. 必须使用中文回答。
6. 不要输出“根据已知信息”“根据上下文”这类提示语。

<问题>
{query}

<答案>
"""

UNIFIED_ROUTE_PROMPT = """背景：当前语料库均为 Research & Development（简称 RD）部门内部资料，RD 指该团队/部门。
你是统一检索网关的数据源路由器。请根据用户问题选择需要调用的数据源，可以多选。

<数据源说明>
1. document: 模型发版文档正文，来源于 ONES 工作文档或发版 PDF，适合查当前版本输入输出、阈值、优化点、追加数据、标签、背景说明，以及正文/table/image 细节。
2. table: 模型发版信息汇总表，适合查模型清单、负责人、设备、OID、更新时间、推荐配置等结构化字段。
3. adela: 具体模型部署记录，适合查部署平台、did/rid、部署状态、部署版本等。
4. public_cloud: 公有云已部署模型列表，适合查公有云在线模型清单和模型 ID。

<可用数据源>
{enabled_sources}

<决策规则>
1. 只在可用数据源里选择。
2. 优先保证查全；需要交叉核对时可多选。
3. `document` 默认优先，不要轻易跳过。
4. 问题涉及当前版本内容、输入输出、阈值、优化点、追加数据、标签、背景说明，或是在问“发版文档里怎么写”，优先包含 document。
5. 如果不确定是否需要 document，但问题可能依赖发版文档正文才能答准，也优先带上 document。
6. 问题涉及模型列表、负责人、OID、更新时间、推荐配置等结构化字段，优先包含 table。
7. 问题涉及部署、平台、did/rid、部署版本、部署状态，优先包含 adela。
8. 问题涉及“公有云”“云上部署”“在线模型列表”“vllm 代理模型”，优先包含 public_cloud。

<输出要求>
只输出一个 JSON 对象，不要输出其他内容。格式如下：
{{
  "reason": "一句话说明选择依据",
  "document": true/false,
  "table": true/false,
  "adela": true/false,
  "public_cloud": true/false
}}

<用户问题>
{query}
"""


STRUCTURED_AGGREGATE_PLAN_PROMPT = """背景：当前语料库均为 Research & Development（简称 RD）部门内部资料，RD 指该团队/部门。
你是结构化统计规划 agent。请根据用户问题判断是否需要做全量结构化统计，并规划统计方式。

<可用数据源>
{enabled_sources}

<数据源字段>
table: 模型发版信息汇总表，字段包括 model_name, owner, target_name, algorithm_type, algorithm_name, application_scene, supported_device, recommended_config, oid, last_updated。
adela: 模型部署记录，字段包括 model_name, name, platform, status, did, rid, version, label_list。
public_cloud: 公有云在线模型列表，字段包括 id, owned_by, object, created。

<字段画像>
{source_profiles}

<规划规则>
1. 只有当问题是在问“多少/几个/数量/总数/统计”这类全量统计时，should_aggregate 才为 true。
2. 问“RD 部门总共有多少个模型/模型数量”时，RD 是语料库整体背景，不要把 RD 当过滤关键词；优先使用 table，按 model_name 去重计数。
3. 问“部署了多少模型/多少部署记录/did/rid/平台”时，优先使用 adela；模型数按 model_name 或 name 去重，部署记录数按记录数统计。
4. 问“公有云/云上/在线模型”数量时，优先使用 public_cloud，按 id 去重计数。
5. 如果问题包含业务/项目/方向/模型类别限定，例如“中车业务”“安全绳”“cuda11.0-trt7.1-fp16-T4”“车牌检测”，必须结合字段画像判断它最可能落在哪些字段上，并输出字段条件。
6. 不要把用户问题直接拆成关键词扫描全表；field_conditions 必须绑定具体字段，并优先使用字段画像里实际出现的规范值或可解释的字段片段。例如“中车业务”应规划为 application_scene contains 中车工厂项目，必要时补充 algorithm_name contains 中车工厂。
7. condition_logic 表示多条字段条件之间的关系：同一业务线可能出现在多个字段时用 any；多个限定必须同时满足时用 all。

<输出要求>
只输出一个 JSON 对象，不要输出其他内容。格式如下：
{{
  "should_aggregate": true/false,
  "reason": "一句话说明统计规划",
  "source_type": "table/adela/public_cloud/null",
  "operation": "count_unique/count_records/null",
  "count_field": "model_name/name/id/null",
  "dedupe_field": "model_name/name/id/null",
  "count_semantics": "model_variant/model_family/deployment_record/public_cloud_model/record/null",
  "breakdown_fields": ["target_name", "algorithm_name"],
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


STRUCTURED_AGGREGATE_REVIEW_PROMPT = """你是结构化统计结果自检 agent。请检查统计计划与执行摘要是否能回答用户问题，并生成简洁中文回答。

<约束>
1. 不允许改动执行摘要中的数字。
2. 回答必须包含主结果、统计口径、筛选条件摘要。
3. 如果存在歧义或置信度不是 high，要在回答中简短说明。
4. 只输出 JSON 对象，不要输出其他内容。

<用户问题>
{query}

<统计计划>
{plan_json}

<执行摘要>
{result_json}

<输出格式>
{{
  "is_plan_reasonable": true/false,
  "confidence": "high/medium/low",
  "ambiguity": "如有口径歧义则说明，否则为空字符串",
  "notes": "简短自检说明",
  "answer": "面向用户的中文最终回答"
}}
"""


SOURCE_TYPE_DISPLAY_NAMES = {
    "document": "模型发版文档正文（ONES工作文档 / 发版PDF）",
    "table": "模型发版信息汇总表",
    "adela": "Adela 部署记录",
    "public_cloud": "公有云在线模型列表",
}


SOURCE_TYPE_HINTS = {
    "document": "适合回答当前版本输入输出、阈值、优化点、追加数据、标签、背景说明、正文/table/image细节。",
    "table": "适合回答模型清单、负责人、OID、更新时间、推荐配置等结构化字段。",
    "adela": "适合回答部署平台、did/rid、部署版本、部署状态等部署信息。",
    "public_cloud": "适合回答公有云在线模型清单、模型ID、在线部署情况。",
}


def format_retrieved_chunks(chunks: List[ChunkNodeWithScore]) -> str:
    if not chunks:
        return "[]"

    formatted_chunks = []
    for idx, chunk in enumerate(chunks, start=1):
        header = chunk.metadata.get("header")
        heading = chunk.metadata.get("heading")
        formatted_chunks.append(
            "\n".join(
                [
                    f"[片段{idx}]",
                    f"doc_name: {chunk.doc_name}",
                    f"header: {header}",
                    f"heading: {heading}",
                    f"score: {chunk.score}",
                    f"index_text: {chunk.index_text}",
                ]
            )
        )

    return "\n\n".join(formatted_chunks)


def resolve_llm_config(llm_config: Optional[LLMConfig]) -> LLMConfig:
    if llm_config is not None:
        return LLMConfig(
            model=llm_config.model or DEFAULT_LLM_API_CONFIG["model"],
            base_url=llm_config.base_url or DEFAULT_LLM_API_CONFIG["base_url"],
            api_key=llm_config.api_key or DEFAULT_LLM_API_CONFIG["api_key"],
            max_tokens=llm_config.max_tokens or DEFAULT_LLM_API_CONFIG["max_tokens"],
            temperature=llm_config.temperature,
            top_p=llm_config.top_p,
            seed=llm_config.seed,
        )

    return LLMConfig(
        model=DEFAULT_LLM_API_CONFIG["model"],
        base_url=DEFAULT_LLM_API_CONFIG["base_url"],
        api_key=DEFAULT_LLM_API_CONFIG["api_key"],
        max_tokens=DEFAULT_LLM_API_CONFIG["max_tokens"],
        temperature=DEFAULT_LLM_API_CONFIG["temperature"],
        top_p=DEFAULT_LLM_API_CONFIG["top_p"],
        seed=DEFAULT_LLM_API_CONFIG.get("seed"),
    )


def extract_final_answer(content: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
    return cleaned or content.strip()


def format_matched_rows(rows: List[TableMatchedRow]) -> str:
    if not rows:
        return "[]"

    formatted_rows = []
    for idx, row in enumerate(rows, start=1):
        entity_lines = [f"{key}: {value}" for key, value in row.entity.items() if value not in (None, "")]
        formatted_rows.append(
            "\n".join(
                [
                    f"[行{idx}]",
                    f"row_id: {row.row_id}",
                    f"score: {row.score}",
                    f"matched_fields: {', '.join(row.matched_fields) if row.matched_fields else '[]'}",
                    *entity_lines,
                ]
            )
        )

    return "\n\n".join(formatted_rows)


def format_unified_evidences(evidences: List[UnifiedEvidenceItem]) -> str:
    if not evidences:
        return "[]"

    blocks = []
    for idx, evidence in enumerate(evidences, start=1):
        source_label = SOURCE_TYPE_DISPLAY_NAMES.get(evidence.source_type, evidence.source_type)
        source_hint = SOURCE_TYPE_HINTS.get(evidence.source_type, "")
        payload_lines = [
            f"{key}: {value}"
            for key, value in evidence.payload.items()
            if value not in (None, "")
        ]
        blocks.append(
            "\n".join(
                [
                    f"[证据{idx}]",
                    f"source_type: {evidence.source_type}",
                    f"source_label: {source_label}",
                    f"source_hint: {source_hint}",
                    f"evidence_id: {evidence.evidence_id}",
                    f"score: {evidence.score}",
                    f"source_rank: {evidence.source_rank}",
                    f"source_score: {evidence.source_score}",
                    f"title: {evidence.title}",
                    f"snippet: {evidence.snippet}",
                    *payload_lines,
                ]
            )
        )
    return "\n\n".join(blocks)


def _extract_first_json_object(text: str) -> dict:
    stripped = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(\{.*\})\s*```$", stripped, flags=re.S)
    if fenced:
        stripped = fenced.group(1).strip()

    try:
        return json.loads(stripped)
    except Exception:
        pass

    start = stripped.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(stripped)):
            ch = stripped[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
                continue
            if ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = stripped[start : idx + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break

    raise ValueError("No valid JSON object found in router response.")


async def answer_question(
    query: str,
    retrieved_chunks: List[ChunkNodeWithScore],
    llm_config: Optional[LLMConfig] = None,
) -> str:
    if not retrieved_chunks:
        return "抱歉,您提问的相关信息在知识库中没有找到"

    resolved = resolve_llm_config(llm_config)
    prompt = RAG_QA_PROMPT.format(
        ctx_chunks=format_retrieved_chunks(retrieved_chunks),
        query=query,
    )

    client = AsyncOpenAI(
        api_key=resolved.api_key,
        base_url=resolved.base_url,
    )
    request_kwargs = {
        "model": resolved.model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": resolved.max_tokens,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    if resolved.temperature is not None:
        request_kwargs["temperature"] = resolved.temperature
    if resolved.top_p is not None:
        request_kwargs["top_p"] = resolved.top_p
    if resolved.seed is not None:
        request_kwargs["seed"] = int(resolved.seed)

    response = await client.chat.completions.create(**request_kwargs)
    return extract_final_answer(response.choices[0].message.content or "")


async def answer_table_question(
    query: str,
    matched_rows: List[TableMatchedRow],
    llm_config: Optional[LLMConfig] = None,
) -> str:
    if not matched_rows:
        return "抱歉,您提问的相关信息在知识库中没有找到"

    resolved = resolve_llm_config(llm_config)
    prompt = TABLE_QA_PROMPT.format(
        table_rows=format_matched_rows(matched_rows),
        query=query,
    )

    client = AsyncOpenAI(
        api_key=resolved.api_key,
        base_url=resolved.base_url,
    )
    request_kwargs = {
        "model": resolved.model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": resolved.max_tokens,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    if resolved.temperature is not None:
        request_kwargs["temperature"] = resolved.temperature
    if resolved.top_p is not None:
        request_kwargs["top_p"] = resolved.top_p
    if resolved.seed is not None:
        request_kwargs["seed"] = int(resolved.seed)

    response = await client.chat.completions.create(**request_kwargs)
    return extract_final_answer(response.choices[0].message.content or "")


async def answer_adela_question(
    query: str,
    matched_rows: List[TableMatchedRow],
    llm_config: Optional[LLMConfig] = None,
) -> str:
    if not matched_rows:
        return "抱歉,您提问的相关信息在知识库中没有找到"

    resolved = resolve_llm_config(llm_config)
    prompt = ADELA_QA_PROMPT.format(
        adela_records=format_matched_rows(matched_rows),
        query=query,
    )

    client = AsyncOpenAI(
        api_key=resolved.api_key,
        base_url=resolved.base_url,
    )
    request_kwargs = {
        "model": resolved.model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": resolved.max_tokens,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    if resolved.temperature is not None:
        request_kwargs["temperature"] = resolved.temperature
    if resolved.top_p is not None:
        request_kwargs["top_p"] = resolved.top_p
    if resolved.seed is not None:
        request_kwargs["seed"] = int(resolved.seed)

    response = await client.chat.completions.create(**request_kwargs)
    return extract_final_answer(response.choices[0].message.content or "")


async def answer_unified_question(
    query: str,
    fused_evidences: List[UnifiedEvidenceItem],
    llm_config: Optional[LLMConfig] = None,
) -> str:
    if not fused_evidences:
        return "抱歉,您提问的相关信息在知识库中没有找到"

    resolved = resolve_llm_config(llm_config)
    prompt = UNIFIED_QA_PROMPT.format(
        evidences=format_unified_evidences(fused_evidences),
        query=query,
    )

    client = AsyncOpenAI(
        api_key=resolved.api_key,
        base_url=resolved.base_url,
    )
    request_kwargs = {
        "model": resolved.model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": resolved.max_tokens,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    if resolved.temperature is not None:
        request_kwargs["temperature"] = resolved.temperature
    if resolved.top_p is not None:
        request_kwargs["top_p"] = resolved.top_p
    if resolved.seed is not None:
        request_kwargs["seed"] = int(resolved.seed)

    response = await client.chat.completions.create(**request_kwargs)
    return extract_final_answer(response.choices[0].message.content or "")


async def answer_unified_question_stream(
    query: str,
    fused_evidences: List[UnifiedEvidenceItem],
    llm_config: Optional[LLMConfig] = None,
) -> AsyncGenerator[str, None]:
    if not fused_evidences:
        yield "抱歉,您提问的相关信息在知识库中没有找到"
        return

    resolved = resolve_llm_config(llm_config)
    prompt = UNIFIED_QA_PROMPT.format(
        evidences=format_unified_evidences(fused_evidences),
        query=query,
    )

    client = AsyncOpenAI(
        api_key=resolved.api_key,
        base_url=resolved.base_url,
    )
    request_kwargs = {
        "model": resolved.model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": resolved.max_tokens,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "stream": True,
    }
    if resolved.temperature is not None:
        request_kwargs["temperature"] = resolved.temperature
    if resolved.top_p is not None:
        request_kwargs["top_p"] = resolved.top_p
    if resolved.seed is not None:
        request_kwargs["seed"] = int(resolved.seed)

    stream = await client.chat.completions.create(**request_kwargs)
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        content = delta.content if delta else None
        if content:
            yield content


async def route_unified_sources(
    query: str,
    enabled_sources: List[Literal["document", "table", "adela", "public_cloud"]],
    llm_config: Optional[LLMConfig] = None,
) -> Tuple[List[Literal["document", "table", "adela", "public_cloud"]], bool, str]:
    if not enabled_sources:
        return [], True, "no enabled source"

    resolved = resolve_llm_config(llm_config)
    prompt = UNIFIED_ROUTE_PROMPT.format(
        enabled_sources=", ".join(enabled_sources),
        query=query,
    )

    client = AsyncOpenAI(
        api_key=resolved.api_key,
        base_url=resolved.base_url,
    )
    request_kwargs = {
        "model": resolved.model,
        "messages": [
            {"role": "system", "content": "You are a strict JSON router."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": min(int(resolved.max_tokens or 512), 512),
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    if resolved.temperature is not None:
        request_kwargs["temperature"] = resolved.temperature
    if resolved.top_p is not None:
        request_kwargs["top_p"] = resolved.top_p
    if resolved.seed is not None:
        request_kwargs["seed"] = int(resolved.seed)

    try:
        response = await client.chat.completions.create(**request_kwargs)
        raw = extract_final_answer(response.choices[0].message.content or "")
        payload = _extract_first_json_object(raw)

        selected: List[Literal["document", "table", "adela", "public_cloud"]] = []
        for source in ("document", "table", "adela", "public_cloud"):
            if source in enabled_sources and bool(payload.get(source)):
                selected.append(source)  # type: ignore[arg-type]

        if not selected:
            return enabled_sources, True, "router returned empty selection, fallback to all enabled sources"

        reason = str(payload.get("reason") or "").strip() or "router selected sources"
        return selected, False, reason
    except Exception as exc:
        return enabled_sources, True, f"router failed ({exc}), fallback to all enabled sources"


async def plan_structured_aggregate_query(
    query: str,
    enabled_sources: List[Literal["table", "adela", "public_cloud"]],
    llm_config: Optional[LLMConfig] = None,
    source_profiles: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not enabled_sources:
        return None

    resolved = resolve_llm_config(llm_config)
    prompt = STRUCTURED_AGGREGATE_PLAN_PROMPT.format(
        enabled_sources=", ".join(enabled_sources),
        source_profiles=source_profiles or "未提供字段画像。",
        query=query,
    )

    client = AsyncOpenAI(
        api_key=resolved.api_key,
        base_url=resolved.base_url,
    )
    request_kwargs = {
        "model": resolved.model,
        "messages": [
            {"role": "system", "content": "You are a strict JSON planning agent."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": min(int(resolved.max_tokens or 1024), 1024),
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    if resolved.temperature is not None:
        request_kwargs["temperature"] = resolved.temperature
    if resolved.top_p is not None:
        request_kwargs["top_p"] = resolved.top_p
    if resolved.seed is not None:
        request_kwargs["seed"] = int(resolved.seed)

    try:
        response = await client.chat.completions.create(**request_kwargs)
        raw = extract_final_answer(response.choices[0].message.content or "")
        payload = _extract_first_json_object(raw)

        source_type = str(payload.get("source_type") or "").strip()
        if source_type == "null":
            source_type = ""
        if source_type not in enabled_sources:
            return None

        operation = str(payload.get("operation") or "").strip()
        if operation not in {"count_unique", "count_records"}:
            return None

        count_field = str(payload.get("count_field") or "").strip()
        if count_field == "null":
            count_field = ""
        dedupe_field = str(payload.get("dedupe_field") or "").strip()
        if dedupe_field == "null":
            dedupe_field = ""
        count_semantics = str(payload.get("count_semantics") or "").strip()
        if count_semantics == "null":
            count_semantics = ""
        confidence = str(payload.get("confidence") or "medium").strip()
        if confidence not in {"high", "medium", "low"}:
            confidence = "medium"
        ambiguity = str(payload.get("ambiguity") or "").strip()

        breakdown_fields = payload.get("breakdown_fields")
        if not isinstance(breakdown_fields, list):
            breakdown_fields = []

        condition_logic = str(payload.get("condition_logic") or "all").strip()
        if condition_logic not in {"all", "any"}:
            condition_logic = "all"

        raw_conditions = payload.get("field_conditions")
        if not isinstance(raw_conditions, list):
            raw_conditions = []
        field_conditions = []
        for item in raw_conditions:
            if not isinstance(item, dict):
                continue
            field = str(item.get("field") or "").strip()
            operator = str(item.get("operator") or "contains").strip()
            values = item.get("values")
            if operator not in {"contains", "equals", "in"}:
                operator = "contains"
            if not isinstance(values, list):
                values = []
            cleaned_values = [
                str(value).strip()
                for value in values
                if str(value).strip()
            ]
            if field and cleaned_values:
                field_conditions.append(
                    {
                        "field": field,
                        "operator": operator,
                        "values": cleaned_values,
                    }
                )

        return {
            "should_aggregate": bool(payload.get("should_aggregate")),
            "reason": str(payload.get("reason") or "").strip(),
            "source_type": source_type,
            "operation": operation,
            "count_field": count_field,
            "dedupe_field": dedupe_field,
            "count_semantics": count_semantics,
            "breakdown_fields": [
                str(item).strip()
                for item in breakdown_fields
                if str(item).strip()
            ],
            "confidence": confidence,
            "ambiguity": ambiguity,
            "condition_logic": condition_logic,
            "field_conditions": field_conditions,
        }
    except Exception:
        return None


async def review_structured_aggregate_result(
    query: str,
    plan: Dict[str, Any],
    result_summary: Dict[str, Any],
    llm_config: Optional[LLMConfig] = None,
) -> Optional[Dict[str, Any]]:
    resolved = resolve_llm_config(llm_config)
    prompt = STRUCTURED_AGGREGATE_REVIEW_PROMPT.format(
        query=query,
        plan_json=json.dumps(plan, ensure_ascii=False, indent=2),
        result_json=json.dumps(result_summary, ensure_ascii=False, indent=2),
    )

    client = AsyncOpenAI(
        api_key=resolved.api_key,
        base_url=resolved.base_url,
    )
    request_kwargs = {
        "model": resolved.model,
        "messages": [
            {"role": "system", "content": "You are a strict JSON review agent."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": min(int(resolved.max_tokens or 768), 768),
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    if resolved.temperature is not None:
        request_kwargs["temperature"] = resolved.temperature
    if resolved.top_p is not None:
        request_kwargs["top_p"] = resolved.top_p
    if resolved.seed is not None:
        request_kwargs["seed"] = int(resolved.seed)

    try:
        response = await client.chat.completions.create(**request_kwargs)
        raw = extract_final_answer(response.choices[0].message.content or "")
        payload = _extract_first_json_object(raw)
        confidence = str(payload.get("confidence") or result_summary.get("confidence") or "medium").strip()
        if confidence not in {"high", "medium", "low"}:
            confidence = "medium"
        return {
            "is_plan_reasonable": bool(payload.get("is_plan_reasonable")),
            "confidence": confidence,
            "ambiguity": str(payload.get("ambiguity") or "").strip(),
            "notes": str(payload.get("notes") or "").strip(),
            "answer": str(payload.get("answer") or "").strip(),
        }
    except Exception:
        return None
