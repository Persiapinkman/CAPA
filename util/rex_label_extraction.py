"""
从用户需求（及可选结构化规划）中抽取 Rex-Omni 开集检测标签，不依赖类别硬编码表。
支持单目标检测与多类别（如头盔颜色九分类）场景。
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from util.vlm_service import VLMService

_DEFAULT_API_BASE = os.environ.get(
    "DEMO_MIGRATION_ADVISOR_API_BASE",
    os.environ.get("DEMO_ANSWER_API_BASE", os.environ.get("DEMO_LLM_API_BASE", "http://10.111.32.253:8000/v1")),
)
_DEFAULT_API_KEY = os.environ.get(
    "DEMO_MIGRATION_ADVISOR_API_KEY",
    os.environ.get("DEMO_ANSWER_API_KEY", os.environ.get("DEMO_LLM_API_KEY", "token.sdc@2026")),
)
_DEFAULT_MODEL = os.environ.get("DEMO_MIGRATION_ADVISOR_MODEL", os.environ.get("DEMO_ANSWER_MODEL", "Qwen3.5-4B"))

REX_LABEL_EXTRACTION_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "rex_detection_label_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "task_mode": {
                    "type": "string",
                    "enum": ["single_object", "multi_class"],
                    "description": "单目标检测或多类别/多属性分类检测",
                },
                "display_label": {
                    "type": "string",
                    "description": "面向用户的检测任务简述",
                },
                "object": {
                    "type": "string",
                    "description": "检测对象，如头盔、缺陷、行人",
                },
                "classes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "class_key": {
                                "type": "string",
                                "description": "英文 snake_case 类别键",
                            },
                            "label": {
                                "type": "string",
                                "description": "用户侧类别名，如黑、白、迷彩",
                            },
                            "detect_phrase": {
                                "type": "string",
                                "description": "送入 Rex Detect 的单一短语，如 black helmet、白色头盔，需自然语言完整表达",
                            },
                            "tokens": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                                "maxItems": 6,
                                "description": "同义词/备用词，用于别名映射",
                            },
                        },
                        "required": ["class_key", "label", "detect_phrase", "tokens"],
                        "additionalProperties": False,
                    },
                    "minItems": 1,
                    "maxItems": 20,
                },
            },
            "required": ["task_mode", "display_label", "object", "classes"],
            "additionalProperties": False,
        },
    },
}


def _extract_first_json_object(text: str) -> dict:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9]*\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw).strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except (TypeError, ValueError):
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        data = json.loads(raw[start : end + 1])
        if isinstance(data, dict):
            return data
    raise ValueError("no json object found")


def _slug_class_key(label: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(label or "").strip()).strip("_").lower()
    return s or "detect_object"


def _normalize_class_row(row: Any, *, object_name: str = "", index: int = 0) -> dict | None:
    if not isinstance(row, dict):
        return None
    label = str(row.get("label") or "").strip()
    if not label:
        return None
    class_key = _slug_class_key(str(row.get("class_key") or f"{object_name}_{label}" or label))
    detect_phrase = str(row.get("detect_phrase") or "").strip()
    tokens_raw = row.get("tokens")
    tokens: list[str] = []
    seen: set[str] = set()

    def _add_token(t: str) -> None:
        s = str(t or "").strip()
        if not s:
            return
        key = s.lower()
        if key in seen:
            return
        seen.add(key)
        tokens.append(s)

    if detect_phrase:
        _add_token(detect_phrase)
    if isinstance(tokens_raw, list):
        for item in tokens_raw:
            _add_token(str(item or ""))
    _add_token(label)
    if object_name:
        _add_token(object_name)
        if not detect_phrase:
            detect_phrase = f"{label}{object_name}".strip()
            if object_name not in label and label not in object_name:
                detect_phrase = f"{label}色{object_name}" if len(label) <= 2 else f"{label} {object_name}"
    if not detect_phrase and tokens:
        detect_phrase = tokens[0]
    if not detect_phrase:
        detect_phrase = label
    if detect_phrase.lower() not in seen:
        tokens.insert(0, detect_phrase)
    return {
        "class_key": class_key,
        "label": label,
        "detect_phrase": detect_phrase,
        "tokens": tokens[:8],
    }


def normalize_detection_targets(data: dict) -> dict:
    """统一为含 classes[] 的结构，兼容旧版单类字段。"""
    obj = data if isinstance(data, dict) else {}
    object_name = str(obj.get("object") or "").strip()
    display_label = str(obj.get("display_label") or object_name or "").strip()
    classes_in = obj.get("classes")
    classes: list[dict] = []
    if isinstance(classes_in, list):
        for i, row in enumerate(classes_in):
            normalized = _normalize_class_row(row, object_name=object_name, index=i)
            if normalized:
                classes.append(normalized)
    if not classes:
        legacy_label = str(obj.get("display_label") or obj.get("target_label") or "").strip()
        legacy_tokens = obj.get("tokens") if isinstance(obj.get("tokens"), list) else []
        legacy_key = _slug_class_key(str(obj.get("class_key") or legacy_label))
        row = _normalize_class_row(
            {"class_key": legacy_key, "label": legacy_label or legacy_key, "tokens": legacy_tokens},
            object_name=object_name,
        )
        if row:
            classes.append(row)
    if not classes:
        raise ValueError("no detection classes extracted")

    task_mode = str(obj.get("task_mode") or "").strip()
    if len(classes) > 1:
        task_mode = "multi_class"
    elif task_mode not in ("single_object", "multi_class"):
        task_mode = "single_object"

    flat_tokens: list[str] = []
    seen_t: set[str] = set()
    for c in classes:
        for t in c.get("tokens") or []:
            tl = str(t).lower()
            if tl not in seen_t:
                seen_t.add(tl)
                flat_tokens.append(str(t))

    return {
        "task_mode": task_mode,
        "display_label": display_label or object_name or classes[0]["label"],
        "object": object_name or classes[0]["label"],
        "classes": classes,
        "class_key": str(classes[0].get("class_key") or ""),
        "tokens": flat_tokens[:24],
    }


def extract_rex_detection_labels(
    user_query: str,
    *,
    plan: dict | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> dict:
    """
    用 LLM 理解用户需求并抽取 Rex-Omni 检测类别。
    若用户给出枚举属性（如头盔颜色：黑、白、灰…），每个枚举值为一个 class。
    """
    query = str(user_query or "").strip()
    if not query:
        raise ValueError("user_query is required for label extraction")

    plan_obj = plan if isinstance(plan, dict) else {}
    abstract = (
        plan_obj.get("abstract_requirement") if isinstance(plan_obj.get("abstract_requirement"), dict) else {}
    )
    structured_input = {
        "user_query": query,
        "retrieval_plan": {
            "object": str(abstract.get("object") or "").strip(),
            "attribute": str(abstract.get("attribute") or "").strip(),
            "task_type": str(abstract.get("task_type") or "").strip(),
            "scene": str(abstract.get("scene") or "").strip(),
            "constraints": abstract.get("constraints") if isinstance(abstract.get("constraints"), list) else [],
        },
    }
    system = (
        "你是视觉开集检测任务解析器。根据用户问题（及可选需求抽象），抽取 Rex-Omni 需要检测的类别。"
        "要求："
        "1) 先识别检测对象 object（如头盔、缺陷、行人），再识别要区分的属性或类别；"
        "2) 若用户明确列出枚举值（如颜色：黑、白、灰、红、蓝、黄、绿、紫、迷彩），"
        "   必须为每个枚举值生成一个 class，task_mode=multi_class，不得合并成一个笼统类别；"
        "3) 每个 class 的 label 必须与用户原文枚举值一致（如黑、红、迷彩，不得擅自改写成 pink 等其它词）；"
        "4) detect_phrase 必须是单一完整检测短语（如 black helmet、白色头盔、camouflage helmet），"
        "   将用于 Rex 的 Detect 提示，不要只写单个颜色字；"
        "   tokens 可列同义词，供后续别名匹配；"
        "5) 若仅需检测单一目标且无子类枚举，task_mode=single_object，classes 仅 1 项；"
        "6) class_key 为英文 snake_case，全局唯一；"
        "7) 不要把迁移/模型咨询/是否支持等问句当作检测类别；"
        "8) 只输出 JSON。"
    )
    user = json.dumps(structured_input, ensure_ascii=False, indent=2)
    use_model = model or _DEFAULT_MODEL
    vlm = VLMService(
        api_key=str(api_key or _DEFAULT_API_KEY).strip(),
        base_url=str(base_url or _DEFAULT_API_BASE).rstrip("/"),
    )
    raw = vlm.generate_text(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model=use_model,
        response_format=REX_LABEL_EXTRACTION_FORMAT,
    )
    data = _extract_first_json_object(raw)
    return normalize_detection_targets(data)


def summarize_label_hits(
    coco: dict,
    *,
    classes: list[dict],
) -> dict:
    """按图片统计各类别是否命中（至少 1 个框）。"""
    cat_id_to_name = {
        int(c["id"]): str(c.get("name") or "")
        for c in coco.get("categories", [])
        if isinstance(c, dict) and c.get("id") is not None
    }
    id_to_file = {
        int(im["id"]): str(im.get("file_name") or "")
        for im in coco.get("images", [])
        if isinstance(im, dict) and im.get("id") is not None
    }
    expected = [str(c.get("class_key") or "") for c in classes if isinstance(c, dict)]
    label_by_key = {str(c.get("class_key") or ""): str(c.get("label") or "") for c in classes if isinstance(c, dict)}

    per_image: dict[int, dict] = {}
    for iid, fname in id_to_file.items():
        per_image[iid] = {
            "image_id": iid,
            "file_name": fname,
            "hits": [],
            "misses": [],
            "boxes_by_label": {},
            "num_boxes": 0,
        }

    for ann in coco.get("annotations", []) if isinstance(coco.get("annotations"), list) else []:
        if not isinstance(ann, dict):
            continue
        iid = int(ann.get("image_id", -1))
        if iid not in per_image:
            continue
        cat_name = cat_id_to_name.get(int(ann.get("category_id", -1)), "")
        bbox = ann.get("bbox")
        if not cat_name or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x, y, w, h = bbox
        box = [float(x), float(y), float(x + w), float(y + h)]
        row = per_image[iid]
        row["num_boxes"] = int(row.get("num_boxes") or 0) + 1
        boxes_map = row.setdefault("boxes_by_label", {})
        boxes_map.setdefault(cat_name, []).append(box)
        if cat_name not in row["hits"]:
            row["hits"].append(cat_name)

    per_image_out: list[dict] = []
    aggregate_hits: dict[str, int] = {k: 0 for k in expected if k}
    for iid in sorted(per_image.keys()):
        row = per_image[iid]
        hits = row.get("hits") if isinstance(row.get("hits"), list) else []
        misses = [k for k in expected if k and k not in hits]
        row["misses"] = misses
        row["hit_labels"] = [label_by_key.get(k, k) for k in hits]
        row["miss_labels"] = [label_by_key.get(k, k) for k in misses]
        per_image_out.append(row)
        for k in hits:
            if k in aggregate_hits:
                aggregate_hits[k] += 1

    return {
        "expected_classes": classes,
        "expected_class_keys": expected,
        "per_image": per_image_out,
        "aggregate_hits": aggregate_hits,
        "total_images": len(per_image_out),
    }
