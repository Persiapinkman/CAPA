from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from gbrain_rag.core.text import text_tokens
from gbrain_rag.retrieval.aspects import GENERAL_ASPECT, infer_query_aspect


FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "target_name": ("target_name", "目标名称"),
    "algorithm_type": ("algorithm_type", "算法类型"),
    "algorithm_name": ("algorithm_name", "算法名称"),
    "application_scene": ("application_scene", "应用场景"),
    "owner": ("owner", "负责人(人员)", "负责人"),
    "model_name": ("model_name", "模型名称", "name"),
    "supported_device": ("supported_device", "支持设备"),
    "recommended_config": ("recommended_config", "推荐配置"),
    "last_updated": ("last_updated", "最近更新时间"),
    "last_updated_month": ("last_updated_month", "最近更新时间-提取年月"),
    "oid": ("oid", "OID"),
    "did": ("did",),
    "rid": ("rid",),
    "platform": ("platform",),
    "label_list": ("label_list", "labels"),
}


@dataclass(frozen=True)
class QueryIntent:
    query: str
    terms: tuple[str, ...] = ()
    target_terms: tuple[str, ...] = ()
    algorithm_terms: tuple[str, ...] = ()
    model_terms: tuple[str, ...] = ()
    platform_terms: tuple[str, ...] = ()
    wants_oid: bool = False
    wants_deployment: bool = False
    wants_recommendation: bool = False
    wants_count: bool = False
    wants_list: bool = False
    wants_latest_check: bool = False
    structured: bool = False
    exact_model: str | None = None
    semantic_terms: tuple[str, ...] = ()
    aspect: str = GENERAL_ASPECT
    answer_type: str = "general_answer"
    query_frame: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)


def canonical_value(row: dict[str, Any], field: str) -> str:
    for key in FIELD_ALIASES.get(field, (field,)):
        value = row.get(key)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, list):
            return " ".join(str(item) for item in value if item not in (None, ""))
        return str(value)
    return ""


def row_text(row: dict[str, Any], fields: tuple[str, ...] | None = None) -> str:
    if fields is None:
        values = [str(value) for value in row.values() if value not in (None, "", [], {})]
    else:
        values = [canonical_value(row, field) for field in fields]
    return "\n".join(value for value in values if value)


def expand_query_terms(query: str, extra_terms: list[str] | tuple[str, ...] | None = None) -> tuple[str, ...]:
    terms: list[str] = [query]
    terms.extend(re.findall(r"[A-Za-z0-9][A-Za-z0-9_.:/+-]{1,}", query))
    terms.extend(str(term) for term in (extra_terms or []) if str(term).strip())
    seen: dict[str, None] = {}
    return tuple(seen.setdefault(term, None) or term for term in terms if str(term).strip())


_CJK_SEQUENCE_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
_ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/+-]{1,}")

QUERY_TERM_STOPWORDS = {
    "请问",
    "请帮",
    "帮我",
    "帮忙",
    "找出",
    "推荐",
    "相关",
    "是否",
    "有没有",
    "哪个",
    "哪些",
    "什么",
    "怎么",
    "多少",
    "几个",
    "现在",
    "目前",
    "总共",
    "共有",
    "一共",
    "适配",
    "支持",
    "平台",
    "模型",
    "算法",
    "识别",
    "检测",
    "分类",
    "记录",
    "信息",
    "部门",
    "实现",
    "我要",
    "能用",
    "检测用",
    "用什么",
}


def semantic_query_terms(
    query: str,
    extra_terms: list[str] | tuple[str, ...] | None = None,
    *,
    max_terms: int = 48,
) -> tuple[str, ...]:
    """Extract compact lexical terms for structured metadata scoring."""

    terms: list[str] = []
    raw = str(query or "")
    for token in _ASCII_TOKEN_RE.findall(raw):
        terms.append(token.lower())
    for token in extra_terms or []:
        text = str(token or "").strip()
        if text:
            terms.append(text.lower())
            terms.extend(match.group(0).lower() for match in _ASCII_TOKEN_RE.finditer(text))

    for seq in _CJK_SEQUENCE_RE.findall(raw):
        seq = seq.strip()
        if len(seq) <= 8 and seq not in QUERY_TERM_STOPWORDS:
            terms.append(seq)
        max_n = min(6, len(seq))
        for n in range(2, max_n + 1):
            for idx in range(0, len(seq) - n + 1):
                gram = seq[idx : idx + n]
                if gram in QUERY_TERM_STOPWORDS:
                    continue
                terms.append(gram)

    terms.extend(token for token in text_tokens(raw) if len(token) >= 2)

    cleaned: list[str] = []
    seen: set[str] = set()
    for term in terms:
        value = term.strip().lower()
        if not value or value in QUERY_TERM_STOPWORDS:
            continue
        if len(value) == 1:
            continue
        if value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
        if len(cleaned) >= max_terms:
            break
    return tuple(cleaned)


_FIELD_CONSTRAINT_STOPWORDS = (
    "算法",
    "模型",
    "记录",
    "数量",
    "多少",
    "几个",
    "几款",
    "哪些",
    "有哪些",
    "清单",
    "列表",
    "一共",
    "总共",
    "共有",
    "部署",
    "发布",
)


def _clean_field_constraint_value(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^[\\s:：=,，。；;]+|[\\s:：=,，。；;？?]+$", "", text)
    for stopword in _FIELD_CONSTRAINT_STOPWORDS:
        idx = text.find(stopword)
        if idx > 0:
            text = text[:idx].strip()
    if text.endswith("的"):
        text = text[:-1].strip()
    return text


def _extract_field_constraint_terms(query: str, field_names: tuple[str, ...]) -> tuple[str, ...]:
    terms: list[str] = []
    field_pattern = "|".join(re.escape(name) for name in field_names)
    operator_pattern = r"(?:为|是|=|:|：|包含|包括)"
    pattern = re.compile(
        rf"(?:{field_pattern})\s*{operator_pattern}\s*"
        rf"([A-Za-z0-9_.:/+-]+|[\u4e00-\u9fff、A-Za-z0-9_.:/+-]+)",
        re.I,
    )
    for match in pattern.finditer(query):
        value = _clean_field_constraint_value(match.group(1))
        if value and value not in QUERY_TERM_STOPWORDS:
            terms.append(value)
    return tuple(dict.fromkeys(terms))


def build_query_intent(query: str, extra_terms: list[str] | tuple[str, ...] | None = None) -> QueryIntent:
    normalized = str(query or "").lower()
    terms = expand_query_terms(query, extra_terms=extra_terms)
    semantic_terms = semantic_query_terms(query, extra_terms=extra_terms)
    aspect, answer_type = infer_query_aspect(query)
    platform_terms = tuple(
        dict.fromkeys(
            re.findall(
                r"(?:cuda[\w.-]*t4|t4|p4|l4|a2|710|ascend\d+|acl-[\w.-]+|cpu-[\w.-]+)",
                normalized,
                re.I,
            )
        )
    )
    wants_oid = "oid" in normalized
    wants_deployment = any(token in normalized for token in ("部署", "did", "rid", "平台", "上线"))
    wants_recommendation = any(
        token in normalized
        for token in (
            "推荐",
            "识别是否",
            "能用",
            "请帮忙",
            "哪个模型",
            "用哪个",
            "用什么模型",
            "什么模型",
            "模型是哪一个",
            "模型是哪款",
            "该用什么",
        )
    ) or (
        any(noun in normalized for noun in ("模型", "方案"))
        and any(token in normalized for token in ("哪一个", "哪款", "可用", "现用", "现在用", "目前用"))
    )
    value_lookup_fields = (
        "特征维度",
        "维度",
        "oid",
        "负责人",
        "owner",
        "推荐配置",
        "supported_device",
        "平台",
    )
    value_lookup_phrases = ("是多少", "是什么", "是谁", "哪一个", "哪个")
    is_value_lookup = any(field in normalized for field in value_lookup_fields) and any(
        phrase in normalized for phrase in value_lookup_phrases
    )
    wants_count = (not is_value_lookup) and any(token in normalized for token in ("多少个", "多少款", "多少种", "有多少", "几个", "数量", "统计", "总共", "共有", "一共"))
    wants_list = any(token in normalized for token in ("哪些", "有哪些", "有什么", "找出", "清单", "列表"))
    wants_latest_check = any(token in normalized for token in ("最新", "目前", "现在"))
    exact_model = None
    model_match = re.search(r"([A-Za-z][A-Za-z0-9_.-]*\d+\.\d+\.\d+[A-Za-z0-9_.-]*)", query)
    if model_match:
        exact_model = model_match.group(1)

    target_terms: list[str] = list(_extract_field_constraint_terms(query, ("目标名称", "目标", "target_name")))
    algorithm_terms: list[str] = list(
        _extract_field_constraint_terms(query, ("算法名称", "算法类型", "算法", "algorithm_name", "algorithm_type"))
    )
    model_terms: list[str] = []
    if exact_model:
        model_terms.append(exact_model)

    structured = bool(
        wants_oid
        or wants_deployment
        or wants_recommendation
        or wants_count
        or wants_list
        or wants_latest_check
        or target_terms
        or algorithm_terms
        or model_terms
        or platform_terms
    )
    entity_mentions = _entity_mentions_from_query(query, semantic_terms)
    normalized_entities = tuple(
        dict.fromkeys(
            term
            for term in (*target_terms, *algorithm_terms, *model_terms, *(extra_terms or ()))
            if str(term).strip()
        )
    )
    query_frame = {
        "entity_mentions": list(entity_mentions),
        "normalized_entities": list(normalized_entities),
        "aspect": aspect,
        "answer_type": answer_type,
        "constraints": {
            "version": exact_model,
            "platform": list(platform_terms),
        },
    }
    return QueryIntent(
        query=query,
        terms=terms,
        target_terms=tuple(dict.fromkeys(target_terms)),
        algorithm_terms=tuple(dict.fromkeys(algorithm_terms)),
        model_terms=tuple(dict.fromkeys(model_terms)),
        platform_terms=tuple(dict.fromkeys(platform_terms)),
        wants_oid=wants_oid,
        wants_deployment=wants_deployment,
        wants_recommendation=wants_recommendation,
        wants_count=wants_count,
        wants_list=wants_list,
        wants_latest_check=wants_latest_check,
        structured=structured,
        exact_model=exact_model,
        semantic_terms=semantic_terms,
        aspect=aspect,
        answer_type=answer_type,
        query_frame=query_frame,
    )


def _entity_mentions_from_query(query: str, semantic_terms: tuple[str, ...]) -> tuple[str, ...]:
    mentions: list[str] = []
    text = str(query or "").strip()
    stop_phrases = (
        "在哪些情况下",
        "哪些情况下",
        "什么情况下",
        "精度如何",
        "精度怎么样",
        "输入输出",
        "用哪个模型",
        "是什么",
        "有哪些",
    )
    head = text
    for phrase in stop_phrases:
        idx = head.find(phrase)
        if idx > 0:
            head = head[:idx]
            break
    head = re.sub(r"(的)?(精度|输入|输出|边界|限制|模型文件|oid|OID|部署|负责人).*$", "", head).strip(" 的？?，,。")
    if 2 <= len(head) <= 40 and any(token in head for token in ("模型", "算法", "检测", "识别", "分类", "属性", "安全绳")):
        mentions.append(head)
    for pattern in (
        r"([A-Za-z][A-Za-z0-9_.+-]*(?:\s+v?\d+(?:\.\d+){1,3})?)",
        r"([\u4e00-\u9fffA-Za-z0-9_+-]{2,}(?:模型|算法|检测|识别|分类|属性))",
    ):
        for match in re.finditer(pattern, text):
            value = match.group(1).strip(" 的？?，,。")
            if value and value not in QUERY_TERM_STOPWORDS:
                mentions.append(value)
    if not mentions:
        mentions.extend(term for term in semantic_terms[:6] if len(term) >= 3)
    return tuple(dict.fromkeys(mentions[:8]))


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms if term)


def _platform_matches(text: str, platform_terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    for term in platform_terms:
        token = term.lower()
        if token in {"t4", "p4", "l4", "a2"}:
            if re.search(rf"(?:^|[-_/]){re.escape(token)}(?:$|[-_/])", lowered):
                return True
            continue
        if token and token in lowered:
            return True
    return False


def _ascii_term_matches(token: str, value: str) -> bool:
    if not token:
        return False
    lowered = value.lower()
    token = token.lower()
    if not re.search(r"[a-z0-9]", token):
        return token in lowered
    if token == "struct":
        return bool(re.search(r"(?<![a-z0-9])struct(?![a-z0-9])", lowered))
    return token in lowered


STRUCTURED_FIELD_WEIGHTS = {
    "target_name": 1.9,
    "algorithm_type": 1.35,
    "algorithm_name": 1.75,
    "application_scene": 0.8,
    "model_name": 0.85,
    "label_list": 0.75,
}


def _term_field_overlap_score(term: str, row: dict[str, Any]) -> float:
    token = str(term or "").strip().lower()
    if not token or token in QUERY_TERM_STOPWORDS:
        return 0.0
    score = 0.0
    for field, weight in STRUCTURED_FIELD_WEIGHTS.items():
        value = canonical_value(row, field).lower()
        if not value:
            continue
        if _ascii_term_matches(token, value):
            length_boost = 1.0 + min(len(token), 8) * 0.06
            score += weight * length_boost
            if field == "model_name" and re.search(r"[a-z0-9_+-]{4,}", token) and token in value:
                score += 3.0
    return score


def _row_count_value(row: dict[str, Any]) -> str:
    return str(row.get("count") or row.get("计数") or "").strip()


def _query_prefers_current_record(intent: QueryIntent) -> bool:
    compact = re.sub(r"\s+", "", intent.query.lower())
    return bool(
        intent.wants_latest_check
        or intent.wants_recommendation
        or any(token in compact for token in ("现用", "当前", "现在", "目前", "应该用", "推荐"))
    )


def score_structured_row(row: dict[str, Any], intent: QueryIntent, source_type: str) -> float:
    if not intent.structured:
        return 0.0
    score = 0.0
    semantic_score = 0.0
    normalized = intent.query.lower()
    full_text = row_text(row).lower()
    target_text = canonical_value(row, "target_name")
    algorithm_name = canonical_value(row, "algorithm_name")
    algorithm_type = canonical_value(row, "algorithm_type")
    model_name = canonical_value(row, "model_name")
    platform = canonical_value(row, "platform") or canonical_value(row, "supported_device") or canonical_value(row, "recommended_config")
    if source_type == "table" and not (target_text or algorithm_name or model_name):
        return 0.0
    if source_type == "adela" and not model_name:
        return 0.0

    for term in intent.semantic_terms:
        lowered = term.lower()
        if not lowered or lowered == intent.query.lower() or lowered in {"rd", "t4", "p4", "l4"}:
            continue
        field_score = _term_field_overlap_score(lowered, row)
        if field_score > 0:
            score += field_score
            semantic_score += field_score
            continue
        if lowered and lowered in full_text:
            score += 0.25
            if len(lowered) >= 3:
                semantic_score += 0.25
    if intent.target_terms and _contains_any(target_text, intent.target_terms):
        score += 4.0
        semantic_score += 4.0
    elif intent.target_terms and (intent.wants_oid or intent.wants_count or intent.wants_list):
        return 0.0
    if intent.algorithm_terms and (
        _contains_any(algorithm_name, intent.algorithm_terms) or _contains_any(algorithm_type, intent.algorithm_terms)
    ):
        score += 4.0
        semantic_score += 4.0
    elif intent.algorithm_terms and (intent.wants_oid or "检测" in intent.query):
        return 0.0
    if intent.model_terms and _contains_any(model_name, intent.model_terms):
        score += 4.5
        semantic_score += 4.5
    if intent.exact_model and intent.exact_model.lower() in model_name.lower():
        score += 5.0
        semantic_score += 5.0
    if source_type == "table" and "识别" in intent.query and "识别" not in f"{algorithm_name}{algorithm_type}":
        score -= 3.0
    if source_type == "table" and "特征" in intent.query and "特征" not in f"{algorithm_name}{algorithm_type}{model_name}":
        score -= 3.0
    if intent.platform_terms and _platform_matches(platform, intent.platform_terms):
        score += 3.0
    elif intent.platform_terms and (intent.wants_deployment or intent.wants_oid):
        return 0.0
    if intent.wants_oid:
        oid = canonical_value(row, "oid").strip().lower()
        if oid and oid not in {"无", "null", "none", "nan", "未记录"}:
            score += 2.0 if "识别" in intent.query else 5.0
        elif "识别" not in intent.query:
            score -= 3.0
    if intent.wants_deployment and source_type == "adela":
        score += 1.5
    if intent.wants_recommendation and source_type == "table":
        score += 1.0
    if source_type == "table" and _query_prefers_current_record(intent):
        count_value = _row_count_value(row)
        if count_value == "1":
            score += 4.0
        elif count_value == "0":
            score -= 4.0
    if any(token in normalized for token in ("高精度", "准确率", "更高精度")):
        model_text = f"{model_name} {algorithm_name}".lower()
        if "large" in model_text:
            score += 1.8
        if "base" in model_text and "large" not in model_text:
            score -= 0.8
    if "通用" in normalized and not any(token in normalized for token in ("高精度", "准确率", "更高精度")):
        model_text = f"{model_name} {algorithm_name}".lower()
        if "base" in model_text:
            score += 0.9
        if "large" in model_text:
            score -= 0.4
    if any(token in normalized for token in ("轻量", "轻量一点", "小模型")):
        model_text = f"{model_name} {algorithm_name}".lower()
        if "small" in model_text or "base" in model_text:
            score += 1.3
        if "large" in model_text:
            score -= 0.8
    if not intent.wants_count and semantic_score <= 0:
        return 0.0
    return max(score, 0.0)


def parse_date(value: str) -> datetime | None:
    text = str(value or "").strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y年%m月%d日", "%Y年%m月"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None
