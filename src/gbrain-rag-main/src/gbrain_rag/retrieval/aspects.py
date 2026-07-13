from __future__ import annotations

import re
from typing import Any

from gbrain_rag.core.types import Chunk


Aspect = str


ACCURACY_ASPECT = "accuracy_metric"
LIMITATION_ASPECT = "limitation"
INPUT_OUTPUT_ASPECT = "input_output"
MODEL_ARTIFACT_ASPECT = "model_artifact"
DEPLOYMENT_ASPECT = "deployment"
RELEASE_CHANGE_ASPECT = "release_change"
OWNER_METADATA_ASPECT = "owner_metadata"
LABEL_ASPECT = "label_schema"
PERFORMANCE_ASPECT = "performance_latency"
GENERAL_ASPECT = "general"


_METRIC_RE = re.compile(
    r"\b(?:acc|accuracy|macc|mprec|ma|map|top\s*-?\s*1|top1|top\s*-?\s*5|top5|"
    r"precision|recall|f1(?:-score)?|auc|tar|far|fppi|ap|iou)\b",
    re.I,
)
_METRIC_VALUE_RE = re.compile(
    r"(?:\b\d+(?:\.\d+)?\s*%|\b0\.\d+\b|\b1\.0+\b|\+\d+(?:\.\d+)?\s*%)"
)
_MODEL_ARTIFACT_RE = re.compile(r"\.(?:model|onnx|pt|pth|safetensors)\b", re.I)
_OID_RE = re.compile(r"\b[0-9a-f]{32,}\b", re.I)
_PLATFORM_RE = re.compile(r"\b(?:cuda|trt|acl|ascend|cpu|ppl|nart|fp16|fp32|int8|t4|p4|l4)\b", re.I)


def infer_query_aspect(query: str) -> tuple[Aspect, str]:
    """Infer the main answer aspect from a user query.

    The rules are intentionally domain-level rather than entity-level. For
    example, they distinguish "精度如何" from "哪些场景精度无法保证" without
    knowing whether the entity is safety_rope, face, or a vehicle model.
    """

    text = str(query or "").lower()
    compact = re.sub(r"\s+", "", text)

    limitation_terms = (
        "边界",
        "限制",
        "局限",
        "无法保证",
        "不能保证",
        "失败",
        "误报",
        "漏报",
        "适用条件",
        "前提条件",
        "哪些情况",
        "什么情况",
        "什么场景",
        "场景要求",
        "目标要求",
        "要求是什么",
    )
    if any(term in compact for term in limitation_terms):
        return LIMITATION_ASPECT, "limitation_summary"

    if any(term in compact for term in ("输入", "输出", "入参", "出参", "怎么给", "如何给", "预测结果")):
        return INPUT_OUTPUT_ASPECT, "io_spec"

    if any(term in compact for term in ("模型文件", "模型列表", "oid", "特征维度", "组件类型", "推荐配置", "支持设备")):
        return MODEL_ARTIFACT_ASPECT, "artifact_or_field_lookup"

    if any(term in compact for term in ("部署", "adela", "did", "rid", "上线", "平台状态")):
        return DEPLOYMENT_ASPECT, "deployment_lookup"

    if any(term in compact for term in ("负责人", "owner", "更新时间", "发布时间", "发布人")):
        return OWNER_METADATA_ASPECT, "metadata_lookup"

    label_terms = ("标签", "label", "类别", "分类结果", "有哪些类", "检测类别")
    if any(term in compact for term in label_terms):
        return LABEL_ASPECT, "label_schema"

    accuracy_terms = (
        "精度",
        "准确率",
        "准确",
        "召回",
        "f1",
        "map",
        "acc",
        "top1",
        "top-1",
        "指标",
        "测试结果",
        "评测",
    )
    if any(term in compact for term in accuracy_terms):
        return ACCURACY_ASPECT, "evaluation_summary"

    if any(term in compact for term in ("耗时", "性能", "延迟", "吞吐", "内存", "显存", "设备占用")):
        return PERFORMANCE_ASPECT, "runtime_performance"

    if any(term in compact for term in ("优化", "提升", "追加", "相比", "变化", "release", "note", "版本")):
        return RELEASE_CHANGE_ASPECT, "release_change_summary"

    return GENERAL_ASPECT, "general_answer"


def classify_chunk_aspects(chunk: Chunk) -> tuple[Aspect, ...]:
    text = _chunk_text(chunk)
    compact = re.sub(r"\s+", "", text.lower())
    metadata = chunk.metadata or {}
    aspects: list[Aspect] = []

    if chunk.source_type == "adela":
        aspects.append(DEPLOYMENT_ASPECT)
        if metadata.get("benchmark_info") or "benchmark_info" in compact:
            aspects.append(ACCURACY_ASPECT)

    if chunk.source_type == "table":
        if any(key in metadata for key in ("owner", "负责人(人员)", "last_updated", "最近更新时间")):
            aspects.append(OWNER_METADATA_ASPECT)
        if any(key in metadata for key in ("oid", "OID", "model_name", "模型名称", "supported_device", "recommended_config")):
            aspects.append(MODEL_ARTIFACT_ASPECT)
        if metadata.get("label_list") or metadata.get("labels"):
            aspects.append(LABEL_ASPECT)

    if _contains_limitation(compact):
        aspects.append(LIMITATION_ASPECT)
    if _contains_accuracy_metric(text, compact):
        aspects.append(ACCURACY_ASPECT)
    if _contains_input_output(compact):
        aspects.append(INPUT_OUTPUT_ASPECT)
    if _contains_model_artifact(text, compact):
        aspects.append(MODEL_ARTIFACT_ASPECT)
    if _contains_deployment(text, compact):
        aspects.append(DEPLOYMENT_ASPECT)
    if _contains_release_change(compact):
        aspects.append(RELEASE_CHANGE_ASPECT)
    if _contains_labels(text, compact):
        aspects.append(LABEL_ASPECT)
    if _contains_performance(text, compact):
        aspects.append(PERFORMANCE_ASPECT)
    if _contains_owner_metadata(compact):
        aspects.append(OWNER_METADATA_ASPECT)

    return tuple(dict.fromkeys(aspects)) or (GENERAL_ASPECT,)


def chunk_section_type(chunk: Chunk, aspects: tuple[Aspect, ...] | None = None) -> str:
    aspects = aspects or classify_chunk_aspects(chunk)
    text = _chunk_text(chunk)
    compact = re.sub(r"\s+", "", text.lower())

    if LIMITATION_ASPECT in aspects and any(term in compact for term in ("无法保证", "算法边界", "前提条件", "场景要求", "目标要求")):
        return "algorithm_boundary"
    if ACCURACY_ASPECT in aspects and any(term in compact for term in ("精度测试", "模型精度", "测试结果", "评测")):
        return "accuracy_eval"
    if ACCURACY_ASPECT in aspects and _contains_release_change(compact):
        return "release_note"
    if ACCURACY_ASPECT in aspects:
        return "accuracy_metric"
    if LIMITATION_ASPECT in aspects:
        return "algorithm_boundary"
    if INPUT_OUTPUT_ASPECT in aspects:
        return "input_output"
    if MODEL_ARTIFACT_ASPECT in aspects:
        return "model_artifact"
    if DEPLOYMENT_ASPECT in aspects:
        return "deployment"
    if LABEL_ASPECT in aspects:
        return "label_schema"
    if PERFORMANCE_ASPECT in aspects:
        return "performance_eval"
    if OWNER_METADATA_ASPECT in aspects:
        return "owner_metadata"
    if RELEASE_CHANGE_ASPECT in aspects:
        return "release_note"
    return "general"


def answerability_score(query_aspect: Aspect, chunk: Chunk) -> tuple[float, dict[str, Any]]:
    aspects = classify_chunk_aspects(chunk)
    section_type = chunk_section_type(chunk, aspects)
    score = 0.0

    if query_aspect == GENERAL_ASPECT:
        score = 0.0
    elif query_aspect in aspects:
        score += 0.7
    elif _compatible_aspect(query_aspect, aspects):
        score += 0.35
    else:
        score -= 0.12

    text = _chunk_text(chunk)
    compact = re.sub(r"\s+", "", text.lower())

    if query_aspect == ACCURACY_ASPECT:
        if _contains_accuracy_metric(text, compact):
            score += 0.5
        if _METRIC_VALUE_RE.search(text):
            score += 0.2
        if section_type in {"accuracy_eval", "release_note", "accuracy_metric"}:
            score += 0.25
        if LIMITATION_ASPECT in aspects and ACCURACY_ASPECT not in aspects:
            score -= 0.5
        elif LIMITATION_ASPECT in aspects and "无法保证" in compact:
            score -= 0.18
        if LABEL_ASPECT in aspects and ACCURACY_ASPECT not in aspects:
            score -= 0.22
        if MODEL_ARTIFACT_ASPECT in aspects and ACCURACY_ASPECT not in aspects:
            score -= 0.25
    elif query_aspect == LIMITATION_ASPECT:
        if LIMITATION_ASPECT in aspects:
            score += 0.45
        if any(term in compact for term in ("无法保证", "边界", "前提条件", "场景要求", "目标要求")):
            score += 0.35
        if ACCURACY_ASPECT in aspects and LIMITATION_ASPECT not in aspects:
            score -= 0.3
    elif query_aspect == INPUT_OUTPUT_ASPECT:
        if _contains_input_output(compact):
            score += 0.45
    elif query_aspect == MODEL_ARTIFACT_ASPECT:
        if _contains_model_artifact(text, compact):
            score += 0.35
    elif query_aspect == DEPLOYMENT_ASPECT:
        if chunk.source_type == "adela":
            score += 0.35
        if _contains_deployment(text, compact):
            score += 0.25
    elif query_aspect == LABEL_ASPECT:
        if _contains_labels(text, compact):
            score += 0.35
    elif query_aspect == PERFORMANCE_ASPECT:
        if _contains_performance(text, compact):
            score += 0.35
    elif query_aspect == RELEASE_CHANGE_ASPECT:
        if _contains_release_change(compact):
            score += 0.35
        if _contains_accuracy_metric(text, compact):
            score += 0.12

    if chunk.source_type in {"table", "adela"} and query_aspect in {
        ACCURACY_ASPECT,
        LIMITATION_ASPECT,
        INPUT_OUTPUT_ASPECT,
        LABEL_ASPECT,
        RELEASE_CHANGE_ASPECT,
        PERFORMANCE_ASPECT,
    } and query_aspect not in aspects:
        score -= 0.2

    score = max(-1.0, min(1.8, score))
    return score, {
        "query_aspect": query_aspect,
        "chunk_aspects": list(aspects),
        "section_type": section_type,
        "evidence_role": evidence_role(query_aspect, aspects, score),
    }


def evidence_role(query_aspect: Aspect, chunk_aspects: tuple[Aspect, ...], score: float) -> str:
    if query_aspect != GENERAL_ASPECT and query_aspect in chunk_aspects and score >= 0.7:
        return "primary"
    if query_aspect == ACCURACY_ASPECT and LIMITATION_ASPECT in chunk_aspects and ACCURACY_ASPECT not in chunk_aspects:
        return "caveat"
    if query_aspect == LIMITATION_ASPECT and ACCURACY_ASPECT in chunk_aspects and LIMITATION_ASPECT not in chunk_aspects:
        return "supporting"
    if LIMITATION_ASPECT in chunk_aspects and score < 0.45:
        return "caveat"
    return "supporting"


def _chunk_text(chunk: Chunk) -> str:
    metadata_text = "\n".join(str(value) for value in (chunk.metadata or {}).values() if value not in (None, "", [], {}))
    return "\n".join(
        part
        for part in (
            chunk.doc_name,
            chunk.title or "",
            chunk.text,
            metadata_text,
        )
        if part
    )


def _compatible_aspect(query_aspect: Aspect, aspects: tuple[Aspect, ...]) -> bool:
    if query_aspect == ACCURACY_ASPECT and RELEASE_CHANGE_ASPECT in aspects:
        return True
    if query_aspect == RELEASE_CHANGE_ASPECT and ACCURACY_ASPECT in aspects:
        return True
    if query_aspect == MODEL_ARTIFACT_ASPECT and DEPLOYMENT_ASPECT in aspects:
        return True
    return False


def _contains_accuracy_metric(text: str, compact: str) -> bool:
    strong_metric = bool(
        re.search(
            r"\b(?:acc|accuracy|macc|mprec|ma|map|top\s*-?\s*1|top1|top\s*-?\s*5|top5|"
            r"precision|recall|f1(?:-score)?|auc|tar|far|fppi|ap)\b",
            text,
            flags=re.I,
        )
    )
    if _contains_limitation(compact) and not strong_metric and not any(
        word in compact for word in ("指标", "评测", "测试结果", "模型精度", "精度测试", "无掉点", "提升", "下降")
    ):
        return False
    if bool(_METRIC_RE.search(text)):
        return True
    metric_context = (
        "准确率",
        "召回",
        "指标",
        "评测",
        "测试集",
        "测试结果",
        "模型精度",
        "精度测试",
        "无掉点",
        "提升",
        "下降",
    )
    if any(word in compact for word in metric_context) and (
        _METRIC_VALUE_RE.search(text) is not None or any(word in compact for word in ("指标", "评测", "测试"))
    ):
        return True
    if "精度" in compact and _METRIC_VALUE_RE.search(text) is not None and "无法保证" not in compact:
        return True
    return False


def _contains_limitation(compact: str) -> bool:
    return any(
        term in compact
        for term in (
            "算法边界",
            "边界",
            "前提条件",
            "场景要求",
            "目标要求",
            "无法保证",
            "不能保证",
            "暂时无法保证",
            "适用场景",
            "要求：",
            "要求:",
        )
    )


def _contains_input_output(compact: str) -> bool:
    return any(term in compact for term in ("输入：", "输入:", "输出：", "输出:", "模型输入", "模型输出", "预测分类结果"))


def _contains_model_artifact(text: str, compact: str) -> bool:
    return (
        bool(_MODEL_ARTIFACT_RE.search(text))
        or bool(_OID_RE.search(text))
        or "模型文件列表" in compact
        or "模型列表" in compact
        or "oid" in compact
        or "特征维度" in compact
        or bool(_PLATFORM_RE.search(text))
    )


def _contains_deployment(text: str, compact: str) -> bool:
    return any(term in compact for term in ("adela", "部署", "did", "rid", "release_id", "deployment")) or "部署地址" in compact


def _contains_release_change(compact: str) -> bool:
    return any(term in compact for term in ("release", "highlight", "优化", "追加", "提升", "相比", "版本", "掉点"))


def _contains_labels(text: str, compact: str) -> bool:
    return any(term in compact for term in ("标签解释", "label", "类别标签", "检测类别", "分类标签")) or bool(
        re.search(r'"[A-Za-z0-9_+-]{3,}"', text)
    )


def _contains_performance(text: str, compact: str) -> bool:
    return any(term in compact for term in ("性能测试", "模型性能", "推理耗时", "耗时", "内存占用", "设备占用", "吞吐")) or bool(
        re.search(r"\bms\b|\bqps\b", text, flags=re.I)
    )


def _contains_owner_metadata(compact: str) -> bool:
    return any(term in compact for term in ("owner", "负责人", "发布时间", "更新时间", "最近更新时间"))
