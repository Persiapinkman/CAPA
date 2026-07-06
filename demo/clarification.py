from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

from util.vlm_service import VLMService


REPLY_TYPE_CANCEL = "cancel"
REPLY_TYPE_REPLAN = "replan"
REPLY_TYPE_SLOT_UPDATE = "slot_update"

ACTION_NONE = "none"
ACTION_CANCEL = "cancel"
ACTION_REPLAN = "replan"
ACTION_DIRECT_TOOL = "direct_tool"
ACTION_STILL_PENDING = "still_pending"
ACTION_RESOLVED = "resolved"

REPLY_CLASSIFIER_MODEL = "Qwen3.5-4B"
REPLY_CLASSIFIER_API_BASE = "http://10.111.32.253:8000/v1"
REPLY_CLASSIFIER_API_KEY = "token.sdc@2026"

REPLY_TYPE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "clarification_reply_type",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reply_type": {
                    "type": "string",
                    "enum": [REPLY_TYPE_SLOT_UPDATE, REPLY_TYPE_CANCEL, REPLY_TYPE_REPLAN],
                }
            },
            "required": ["reply_type"],
            "additionalProperties": False,
        },
    },
}


def _clean_slot_map(value) -> dict[str, str]:
    src = value if isinstance(value, dict) else {}
    out: dict[str, str] = {}
    for key, raw in src.items():
        name = str(key or "").strip()
        text = str(raw or "").strip()
        if name and text:
            out[name] = text
    return out


def _clean_slot_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _has_tool_arg_value(tool_args: dict, key: str) -> bool:
    if not isinstance(tool_args, dict):
        return False
    if key not in tool_args:
        return False
    value = tool_args.get(key)
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _required_slots_for_tool(
    *,
    action: str,
    qwen_detection_action: str,
    rex_detection_action: str,
    pipeline_eval_action: str,
    flux_action: str,
    adela_cli_action: str,
    tool_args: dict,
) -> list[str]:
    if action in {qwen_detection_action, rex_detection_action}:
        return ["image", "label"]
    if action == pipeline_eval_action:
        return ["image"]
    if action == flux_action and bool(tool_args.get("source_image_required")):
        return ["image"]
    if action == adela_cli_action:
        missing = []
        if not _has_tool_arg_value(tool_args, "rawmodel_id") and not _has_tool_arg_value(tool_args, "model_name"):
            missing.append("model_name")
        if not _has_tool_arg_value(tool_args, "platform"):
            missing.append("platform")
        if not _has_tool_arg_value(tool_args, "eval_type"):
            missing.append("eval_type")
        return missing
    return []


def _tool_precondition_question(
    *,
    action: str,
    missing_slots: list[str],
    qwen_detection_action: str,
    rex_detection_action: str,
    pipeline_eval_action: str,
    flux_action: str,
    adela_cli_action: str,
) -> str:
    missing = _clean_slot_list(missing_slots)
    if action in {qwen_detection_action, rex_detection_action}:
        if missing == ["image"]:
            return "要继续做目标检测，请先上传一张待检测图片。"
        if missing == ["label"]:
            return "要继续做目标检测，请补充你要检测的目标名称，例如“黑猫”或“安全帽”。"
        if missing:
            return "要继续做目标检测，请上传待检测图片，并补充要检测的目标名称。"
    if action == pipeline_eval_action and missing:
        return "要继续做检测评测流水线，请先上传一张参考图片。"
    if action == flux_action and missing:
        return "这次图像生成需要参考图，请先上传一张参考图片。"
    if action == adela_cli_action:
        if missing == ["model_name"]:
            return "要继续做 Adela 部署评测，请先提供模型名称；如果你已知 rawmodel_id，也可以直接给 rawmodel_id。"
        if missing == ["platform"]:
            return "要继续做 Adela 部署评测，请补充目标部署平台，例如 cuda11.0-trt7.1-int8-T4。"
        if missing == ["eval_type"]:
            return "要继续做 Adela 部署评测，请说明评测类型：0 表示精度评测，1 表示性能评测。"
        if missing:
            return "要继续做 Adela 部署评测，请补充模型名称或 rawmodel_id、目标平台 platform，以及评测类型（0=精度，1=性能）。"
    if missing:
        return "当前还缺少继续执行所需的信息，请补充。"
    return ""


def _normalize_slot_value(slot: str, raw_value):
    name = str(slot or "").strip()
    text = "" if raw_value is None else str(raw_value).strip()
    if not name or not text:
        return None
    if name == "rawmodel_id":
        try:
            return int(text)
        except (TypeError, ValueError):
            return text
    if name == "eval_type":
        low = text.lower()
        if text == "0" or low in {
            "精度",
            "精度评测",
            "准确率",
            "accuracy",
            "precision",
            "normal_precision",
            "normal precision",
        }:
            return 0
        if text == "1" or low in {
            "性能",
            "性能评测",
            "速度",
            "performance",
            "speed",
            "latency",
            "normal_performance",
            "normal performance",
        }:
            return 1
        try:
            return int(text)
        except (TypeError, ValueError):
            return text
    return text


def normalize_adela_eval_type_arg(raw_value) -> int | None:
    """将工具参数里的 eval_type 规范为 0 或 1；无法识别时返回 None。"""
    value = _normalize_slot_value("eval_type", raw_value)
    return value if value in (0, 1) else None


def _extract_explicit_adela_rawmodel_id(*values) -> int | None:
    for raw in values:
        text = str(raw or "").strip()
        if not text:
            continue
        if text.isdigit():
            try:
                parsed = int(text)
            except (TypeError, ValueError):
                parsed = 0
            if parsed > 0:
                return parsed
        match = __import__("re").search(
            r"(?:^|[^a-zA-Z])(?:rawmodel_id|model_id|id)\s*(?:是|为|=|:)?\s*(\d+)",
            text,
            flags=__import__("re").I,
        )
        if match:
            try:
                parsed = int(match.group(1))
            except (TypeError, ValueError):
                parsed = 0
            if parsed > 0:
                return parsed
    return None


def _normalize_eval_type_hint(raw_value) -> str:
    value = _normalize_slot_value("eval_type", raw_value)
    if value == 0:
        return "0"
    if value == 1:
        return "1"
    return ""


def _looks_like_adela_platform(text: str) -> bool:
    value = str(text or "").strip().lower()
    if not value:
        return False
    markers = ("cuda", "trt", "acl", "ascend", "halnn", "stpu", "int8", "int4", "fp16", "fp32")
    return any(item in value for item in markers) and ("-" in value or "." in value)


def _parse_adela_slot_form_reply(user_text: str) -> dict | None:
    text = str(user_text or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    if str(data.get("_structured_type") or "").strip() != "adela_slot_form":
        return None
    args: dict = {}
    model_name = str(data.get("model_name") or "").strip()
    platform = str(data.get("platform") or "").strip()
    rawmodel_id = _extract_explicit_adela_rawmodel_id(data.get("rawmodel_id"))
    eval_type = _normalize_slot_value("eval_type", data.get("eval_type"))
    if model_name:
        args["model_name"] = model_name
    if rawmodel_id is not None:
        args["rawmodel_id"] = rawmodel_id
    if platform:
        args["platform"] = platform
    if eval_type in (0, 1):
        args["eval_type"] = eval_type
    args["finish_after_tool"] = bool(data.get("finish_after_tool", False))
    return args


def _normalize_adela_candidate_model_names(values, *, limit: int = 5) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def build_adela_clarification_task_state(
    *,
    task_state: dict,
    known_slots: dict[str, str] | None = None,
    missing_slots: list[str] | None = None,
    tool_args: dict | None = None,
    model_resolution_status: str = "",
    candidate_model_names: list[str] | None = None,
    resolution_message: str = "",
) -> dict:
    src = task_state if isinstance(task_state, dict) else {}
    return {
        "candidate_tool": str(src.get("candidate_tool") or "").strip(),
        "known_slots": _clean_slot_map(known_slots if isinstance(known_slots, dict) else src.get("known_slots")),
        "missing_slots": _clean_slot_list(missing_slots if isinstance(missing_slots, list) else src.get("missing_slots")),
        "tool_args": dict(tool_args) if isinstance(tool_args, dict) else (dict(src.get("tool_args")) if isinstance(src.get("tool_args"), dict) else {}),
        "original_user_text": str(src.get("original_user_text") or "").strip(),
        "model_resolution_status": str(model_resolution_status or src.get("model_resolution_status") or "").strip(),
        "candidate_model_names": _normalize_adela_candidate_model_names(
            candidate_model_names if isinstance(candidate_model_names, list) else src.get("candidate_model_names")
        ),
        "resolution_message": str(resolution_message or src.get("resolution_message") or "").strip(),
    }


def _legacy_task_state(
    *,
    src: dict,
    normalize_action: Callable[[str], str],
) -> dict:
    tool_args = src.get("action_input")
    if not isinstance(tool_args, dict):
        tool_args = {}
    known_slots = _clean_slot_map(src.get("known_slots"))
    if not known_slots:
        for key, raw in tool_args.items():
            name = str(key or "").strip()
            if not name or name == "finish_after_tool":
                continue
            text = str(raw or "").strip()
            if text:
                known_slots[name] = text
    return {
        "candidate_tool": normalize_action(str(src.get("candidate_tool") or src.get("tool_action") or "").strip()),
        "known_slots": known_slots,
        "missing_slots": _clean_slot_list(src.get("missing_slots") if isinstance(src.get("missing_slots"), list) else src.get("missing_requirements")),
        "tool_args": tool_args,
        "original_user_text": str(src.get("original_user_text") or "").strip(),
    }


def normalize_pending_clarification(
    value,
    *,
    normalize_thread_id: Callable[[str], str],
    normalize_action: Callable[[str], str],
) -> dict:
    src = value if isinstance(value, dict) else {}
    status = str(src.get("status") or "").strip().lower()
    question = str(src.get("clarification_question") or "").strip()
    if status != "pending" or not question:
        return {}
    raw_task_state = src.get("task_state")
    task_state_src = raw_task_state if isinstance(raw_task_state, dict) else _legacy_task_state(
        src=src,
        normalize_action=normalize_action,
    )
    return {
        "status": "pending",
        "source": str(src.get("source") or "").strip() or "planner_clarify",
        "query_id": str(src.get("query_id") or "").strip(),
        "thread_id": normalize_thread_id(str(src.get("thread_id") or "").strip()),
        "clarification_question": question,
        "task_state": {
            **build_adela_clarification_task_state(
                task_state={
                    **task_state_src,
                    "candidate_tool": normalize_action(str(task_state_src.get("candidate_tool") or "").strip()),
                }
            ),
        },
    }


def clear_pending_clarification(session: dict) -> None:
    session["pending_clarification"] = {}


def get_pending_clarification(
    session: dict,
    *,
    normalize_thread_id: Callable[[str], str],
    normalize_action: Callable[[str], str],
) -> dict:
    pending = normalize_pending_clarification(
        session.get("pending_clarification"),
        normalize_thread_id=normalize_thread_id,
        normalize_action=normalize_action,
    )
    session["pending_clarification"] = pending
    return pending


def build_tool_precondition_clarification(
    *,
    action: str,
    action_input: dict,
    user_text: str,
    image_path: str,
    normalize_action: Callable[[str], str],
    qwen_detection_action: str,
    rex_detection_action: str,
    pipeline_eval_action: str,
    flux_action: str,
    adela_cli_action: str,
) -> dict | None:
    normalized_action = normalize_action(action)
    payload = dict(action_input) if isinstance(action_input, dict) else {}
    if normalized_action == adela_cli_action:
        explicit_id = _extract_explicit_adela_rawmodel_id(
            payload.get("rawmodel_id"),
            payload.get("model_name"),
            user_text,
        )
        if explicit_id is not None:
            payload["rawmodel_id"] = explicit_id
        if normalize_adela_eval_type_arg(payload.get("eval_type")) is None:
            inferred = adela_eval_type_from_text(str(user_text or ""))
            if inferred in (0, 1):
                payload["eval_type"] = inferred
    known_slots = {
        str(key).strip(): str(value).strip()
        for key, value in payload.items()
        if str(key).strip() and str(key).strip() != "finish_after_tool" and str(value).strip()
    }
    missing_slots = _required_slots_for_tool(
        action=normalized_action,
        qwen_detection_action=qwen_detection_action,
        rex_detection_action=rex_detection_action,
        pipeline_eval_action=pipeline_eval_action,
        flux_action=flux_action,
        adela_cli_action=adela_cli_action,
        tool_args=payload,
    )
    if image_path and Path(image_path).is_file() and "image" in missing_slots:
        missing_slots = [item for item in missing_slots if item != "image"]
        known_slots["image"] = "[uploaded]"
    for slot in list(missing_slots):
        normalized_value = _normalize_slot_value(slot, payload.get(slot))
        if normalized_value is None or str(normalized_value).strip() == "":
            continue
        payload[slot] = normalized_value
        known_slots[slot] = str(normalized_value).strip()
        missing_slots = [item for item in missing_slots if item != slot]
    if not missing_slots:
        return None
    return {
        "source": "tool_precondition",
        "clarification_question": _tool_precondition_question(
            action=normalized_action,
            missing_slots=missing_slots,
            qwen_detection_action=qwen_detection_action,
            rex_detection_action=rex_detection_action,
            pipeline_eval_action=pipeline_eval_action,
            flux_action=flux_action,
            adela_cli_action=adela_cli_action,
        ) or "当前还缺少继续执行所需的信息，请补充。",
        "task_state": {
            "candidate_tool": normalized_action,
            "known_slots": known_slots,
            "missing_slots": missing_slots,
            "tool_args": dict(payload),
            "original_user_text": str(user_text or "").strip(),
        },
    }


def activate_pending_clarification(
    *,
    session: dict,
    emit: Callable[[dict], None],
    thought: str,
    question: str,
    source: str,
    step_index: int,
    run_stamp: str,
    query_id: str,
    thread_id: str,
    normalize_action: Callable[[str], str],
    task_state: dict | None = None,
) -> dict:
    cleaned_question = str(question or "").strip() or "请补充更多信息。"
    src_task_state = task_state if isinstance(task_state, dict) else {}
    pending = {
        "status": "pending",
        "source": str(source or "").strip() or "planner_clarify",
        "query_id": str(query_id or "").strip(),
        "thread_id": str(thread_id or "").strip(),
        "clarification_question": cleaned_question,
        "task_state": build_adela_clarification_task_state(
            task_state={
                **src_task_state,
                "candidate_tool": normalize_action(str(src_task_state.get("candidate_tool") or "").strip()),
            }
        ),
    }
    session["pending_clarification"] = pending
    emit(
        {
            "type": "meta",
            "flow": "clarify",
            "decision": {
                "action": "clarify",
                "reason": thought,
                "direct_reply": "",
            },
            "run_stamp": run_stamp,
            "step_index": step_index,
        }
    )
    emit(
        {
            "type": "clarification",
            "text": cleaned_question,
            "source": pending["source"],
            "missing_slots": pending["task_state"]["missing_slots"],
            "task_state": pending["task_state"],
        }
    )
    return {
        "final_answer": cleaned_question,
        "assistant_text": cleaned_question,
        "assistant_event_type": "clarification",
        "pending_clarification": pending,
        "query_completed": False,
    }


def _merge_clarification_text(original_text: str, reply_text: str) -> str:
    base = str(original_text or "").strip()
    reply = str(reply_text or "").strip()
    if base and reply:
        return f"{base}\n\n用户补充说明：{reply}"
    return reply or base


def _parse_reply_type_response(raw: str) -> str:
    try:
        data = json.loads(str(raw or "").strip())
    except Exception:
        return REPLY_TYPE_SLOT_UPDATE
    if not isinstance(data, dict):
        return REPLY_TYPE_SLOT_UPDATE
    reply_type = str(data.get("reply_type") or "").strip()
    if reply_type in {REPLY_TYPE_SLOT_UPDATE, REPLY_TYPE_CANCEL, REPLY_TYPE_REPLAN}:
        return reply_type
    return REPLY_TYPE_SLOT_UPDATE


def _classify_reply_type(
    *,
    task_state: dict,
    last_clarification_question: str,
    user_reply: str,
) -> str:
    system_prompt = (
        "你是 clarification reply classifier。\n"
        "任务：根据 task_state、上一条 clarification 问题、以及用户当前回复，判断该回复属于哪一类。\n"
        "只允许输出 JSON，字段 reply_type 只能是以下三种之一：\n"
        "1) slot_update：用户在补充之前缺失的工具参数或上传图片；\n"
        "2) cancel：用户明确表示取消、不继续；\n"
        "3) replan：用户不是在补参数，而是在改需求、换目标、重开话题或要求重新理解任务。\n"
        "判断原则：\n"
        "- 如果用户在回答上一条澄清问题，优先判为 slot_update；\n"
        "- 如果用户说“取消/算了/不用了/不做了”等，判为 cancel；\n"
        "- 如果用户明显改变原目标、工具方向或任务意图，判为 replan；\n"
        "- 若不确定，但回复看起来像在补充参数，判为 slot_update。"
    )
    user_payload = {
        "task_state": task_state if isinstance(task_state, dict) else {},
        "last_clarification_question": str(last_clarification_question or "").strip(),
        "user_reply": str(user_reply or "").strip(),
    }
    try:
        vlm = VLMService(
            api_key=REPLY_CLASSIFIER_API_KEY,
            base_url=REPLY_CLASSIFIER_API_BASE.rstrip("/"),
        )
        raw = vlm.generate_text(
            prompt=system_prompt + "\n\n" + json.dumps(user_payload, ensure_ascii=False, indent=2),
            model=REPLY_CLASSIFIER_MODEL,
            response_format=REPLY_TYPE_RESPONSE_FORMAT,
        )
        return _parse_reply_type_response(raw)
    except Exception:
        return REPLY_TYPE_SLOT_UPDATE


def _merge_slot_update(
    *,
    task_state: dict,
    user_text: str,
    image_path: str,
) -> tuple[dict[str, str], dict, list[str]]:
    known_slots = _clean_slot_map(task_state.get("known_slots"))
    tool_args = dict(task_state.get("tool_args")) if isinstance(task_state.get("tool_args"), dict) else {}
    missing_slots = _clean_slot_list(task_state.get("missing_slots"))
    reply = str(user_text or "").strip()
    has_image = bool(image_path and Path(image_path).is_file())

    if "image" in missing_slots and has_image:
        known_slots["image"] = "[uploaded]"
        missing_slots = [item for item in missing_slots if item != "image"]

    if reply and "label" in missing_slots:
        normalized = _normalize_slot_value("label", reply)
        known_slots["label"] = str(normalized).strip()
        tool_args["label"] = normalized
        missing_slots = [item for item in missing_slots if item != "label"]
    elif reply and "task_text" in missing_slots:
        normalized = _normalize_slot_value("task_text", reply)
        known_slots["task_text"] = str(normalized).strip()
        tool_args["task_text"] = normalized
        missing_slots = [item for item in missing_slots if item != "task_text"]
    elif reply and len(missing_slots) == 1:
        slot = missing_slots[0]
        normalized = _normalize_slot_value(slot, reply)
        known_slots[slot] = str(normalized).strip()
        tool_args[slot] = normalized
        missing_slots = []

    return known_slots, tool_args, missing_slots


def _adela_eval_type_is_explicit(user_text: str) -> bool:
    hint = _normalize_eval_type_hint(user_text)
    return bool(hint)


def adela_eval_hits_from_text(user_text: str) -> tuple[bool, bool]:
    """
    从用户自然语言中判断是否包含精度类(acc)、性能类(perf)意图。
    返回 (acc_hit, perf_hit)。二者同时为 True 时，上层仅执行单一 eval_type（见 adela_eval_sequence_from_text），不再依次跑 0 与 1。
    注意：已命中性能类时，不再用「含评测/benchmark」或「部署+多少」等泛化规则把 acc 一并打开，
    避免「问性能却跑精度+性能」双段。
    """
    text = str(user_text or "").strip()
    if not text:
        return False, False
    low = text.lower()
    perf_tokens = (
        "性能",
        "速度",
        "fps",
        "吞吐",
        "延迟",
        "latency",
        "推理速度",
        "inference speed",
        "normal_performance",
    )
    acc_tokens = (
        "精度",
        "准确率",
        "准度",
        "准确度",
        "accuracy",
        "precision",
        "normal_precision",
        "mAP",
    )
    perf_hit = any(t in text for t in perf_tokens) or any(
        t in low for t in ("performance", "speed", "throughput")
    )
    acc_hit = any(t in text for t in acc_tokens) or re.search(r"\bmap\b", low) is not None
    bench_hit = "benchmark" in low or "评测" in text
    deploy_acc = bool("部署" in text and re.search(r"(多少|怎样|如何|怎么样|是什么)", text))
    # 已明确性能类诉求时，不因「评测」「部署+多少」等泛化规则再并入精度，否则会误触发 eval_type=0 再 1 的双段执行。
    if not perf_hit:
        acc_hit = acc_hit or bench_hit or deploy_acc
    return acc_hit, perf_hit


def adela_eval_sequence_from_text(user_text: str) -> list[int]:
    """由用户问题得到单项评测类型序列（至多一段 Adela CLI）。同时含精度与性能意图时只取精度 0。"""
    acc_hit, perf_hit = adela_eval_hits_from_text(user_text)
    if acc_hit and perf_hit:
        return [0]
    if perf_hit:
        return [1]
    if acc_hit:
        return [0]
    return []


def adela_eval_type_from_text(user_text: str) -> int | None:
    """
    解析单一 eval_type（槽位/补全用）。同时含精度与性能时与 adela_eval_sequence_from_text 一致，取 0（精度）。
    """
    text = str(user_text or "").strip()
    if not text:
        return None
    value = _normalize_slot_value("eval_type", text)
    if value in (0, 1):
        return value
    seq = adela_eval_sequence_from_text(text)
    if len(seq) == 1:
        return seq[0]
    return None


def _adela_followup_question(*, model_name: str, candidate_names: list[str]) -> str:
    model = str(model_name or "").strip()
    names = _normalize_adela_candidate_model_names(candidate_names)
    if names:
        options = "\n".join(f"{idx + 1}. {name}" for idx, name in enumerate(names))
        return (
            f"没有找到与“{model}”完全一致的模型名称。请确认你要的是下面哪个模型：\n"
            f"{options}\n"
            "请直接回复完整模型名称。"
        )
    return (
        f"当前 adela 平台上没有找到名为“{model}”的模型。"
        "如有需要，请到 monolith 平台上训练模型，并上传到 adela 平台。"
    )


def handle_pending_reply(
    *,
    pending: dict,
    user_text: str,
    image_path: str,
    normalize_action: Callable[[str], str],
    decision_type_tool: str,
    qwen_detection_action: str,
    rex_detection_action: str,
    pipeline_eval_action: str,
    flux_action: str,
    adela_cli_action: str,
) -> dict:
    if not isinstance(pending, dict) or not pending:
        return {"action": ACTION_NONE}

    task_state = pending.get("task_state") if isinstance(pending.get("task_state"), dict) else {}
    reply_type = _classify_reply_type(
        task_state=task_state,
        last_clarification_question=str(pending.get("clarification_question") or "").strip(),
        user_reply=str(user_text or "").strip(),
    )
    query_id = str(pending.get("query_id") or "").strip()
    thread_id = str(pending.get("thread_id") or "").strip()
    original_user_text = str(task_state.get("original_user_text") or "").strip()
    candidate_tool = normalize_action(str(task_state.get("candidate_tool") or "").strip())

    if reply_type == REPLY_TYPE_CANCEL:
        return {
            "action": ACTION_CANCEL,
            "reply_type": reply_type,
            "query_id": query_id,
            "thread_id": thread_id,
        }

    if reply_type == REPLY_TYPE_REPLAN:
        return {
            "action": ACTION_REPLAN,
            "reply_type": reply_type,
            "query_id": query_id,
            "thread_id": thread_id,
            "effective_text": _merge_clarification_text(original_user_text, user_text),
        }

    if candidate_tool == adela_cli_action:
        known_slots = _clean_slot_map(task_state.get("known_slots"))
        tool_args = dict(task_state.get("tool_args")) if isinstance(task_state.get("tool_args"), dict) else {}
        missing_slots = _clean_slot_list(task_state.get("missing_slots"))
        reply = str(user_text or "").strip()
        form_args = _parse_adela_slot_form_reply(reply)
        if form_args is not None:
            for key, value in form_args.items():
                if key == "eval_type" and value in (0, 1):
                    tool_args[key] = value
                elif key == "finish_after_tool":
                    tool_args[key] = bool(value)
                elif value is not None and str(value).strip():
                    tool_args[key] = value
            if form_args.get("rawmodel_id"):
                tool_args["model_name"] = str(form_args.get("model_name") or "").strip()
            known_slots = {
                str(key).strip(): str(value).strip()
                for key, value in tool_args.items()
                if str(key).strip() and str(key).strip() != "finish_after_tool" and str(value).strip()
            }
            missing_slots = []
            if not _has_tool_arg_value(tool_args, "rawmodel_id") and not _has_tool_arg_value(tool_args, "model_name"):
                missing_slots.append("model_name")
            if not _has_tool_arg_value(tool_args, "platform"):
                missing_slots.append("platform")
            if normalize_adela_eval_type_arg(tool_args.get("eval_type")) is None:
                missing_slots.append("eval_type")
            if not missing_slots:
                return {
                    "action": ACTION_DIRECT_TOOL,
                    "reply_type": reply_type,
                    "query_id": query_id,
                    "thread_id": thread_id,
                    "effective_text": _merge_clarification_text(original_user_text, user_text),
                    "resolved_task_state": build_adela_clarification_task_state(
                        task_state=task_state,
                        known_slots=known_slots,
                        missing_slots=[],
                        tool_args=tool_args,
                    ),
                    "forced_first_step": {
                        "thought": "用户已通过 Adela 参数表单补齐所需参数，继续执行工具。",
                        "decision_type": decision_type_tool,
                        "action": candidate_tool,
                        "action_input": tool_args,
                        "final_answer": "",
                    },
                }
            next_pending = {
                "status": "pending",
                "source": str(pending.get("source") or "").strip() or "tool_precondition",
                "query_id": query_id,
                "thread_id": thread_id,
                "clarification_question": _tool_precondition_question(
                    action=candidate_tool,
                    missing_slots=missing_slots,
                    qwen_detection_action=qwen_detection_action,
                    rex_detection_action=rex_detection_action,
                    pipeline_eval_action=pipeline_eval_action,
                    flux_action=flux_action,
                    adela_cli_action=adela_cli_action,
                ) or "当前还缺少继续执行所需的信息，请补充。",
                "task_state": build_adela_clarification_task_state(
                    task_state=task_state,
                    known_slots=known_slots,
                    missing_slots=missing_slots,
                    tool_args=tool_args,
                ),
            }
            return {
                "action": ACTION_STILL_PENDING,
                "reply_type": reply_type,
                "query_id": query_id,
                "thread_id": thread_id,
                "pending_clarification": next_pending,
                "clarification_question": str(next_pending.get("clarification_question") or ""),
            }
        explicit_id = _extract_explicit_adela_rawmodel_id(
            tool_args.get("rawmodel_id"),
            reply,
        )
        consumed_reply = False
        if explicit_id is not None:
            known_slots["rawmodel_id"] = str(explicit_id)
            tool_args["rawmodel_id"] = explicit_id
            consumed_reply = True
        normalized_eval_type = _normalize_slot_value("eval_type", reply)
        if (
            not consumed_reply
            and "platform" in missing_slots
            and reply
            and not _adela_eval_type_is_explicit(reply)
            and _looks_like_adela_platform(reply)
        ):
            known_slots["platform"] = reply
            tool_args["platform"] = reply
            missing_slots = [x for x in missing_slots if x != "platform"]
            consumed_reply = True
        if not consumed_reply and "eval_type" in missing_slots and normalized_eval_type in (0, 1):
            known_slots["eval_type"] = str(normalized_eval_type)
            tool_args["eval_type"] = normalized_eval_type
            missing_slots = [x for x in missing_slots if x != "eval_type"]
            consumed_reply = True
        if not consumed_reply and "model_name" in missing_slots and reply and explicit_id is None:
            known_slots["model_name"] = reply
            tool_args["model_name"] = reply
            missing_slots = [x for x in missing_slots if x != "model_name"]

        if missing_slots:
            next_pending = {
                "status": "pending",
                "source": str(pending.get("source") or "").strip() or "tool_precondition",
                "query_id": query_id,
                "thread_id": thread_id,
                "clarification_question": _tool_precondition_question(
                    action=candidate_tool,
                    missing_slots=missing_slots,
                    qwen_detection_action=qwen_detection_action,
                    rex_detection_action=rex_detection_action,
                    pipeline_eval_action=pipeline_eval_action,
                    flux_action=flux_action,
                    adela_cli_action=adela_cli_action,
                ) or "当前还缺少继续执行所需的信息，请补充。",
                "task_state": build_adela_clarification_task_state(
                    task_state=task_state,
                    known_slots=known_slots,
                    missing_slots=missing_slots,
                    tool_args=tool_args,
                ),
            }
            return {
                "action": ACTION_STILL_PENDING,
                "reply_type": reply_type,
                "query_id": query_id,
                "thread_id": thread_id,
                "pending_clarification": next_pending,
                "clarification_question": str(next_pending.get("clarification_question") or ""),
            }

        return {
            "action": ACTION_DIRECT_TOOL,
            "reply_type": reply_type,
            "query_id": query_id,
            "thread_id": thread_id,
            "effective_text": _merge_clarification_text(original_user_text, user_text),
            "resolved_task_state": build_adela_clarification_task_state(
                task_state=task_state,
                known_slots=known_slots,
                missing_slots=[],
                tool_args=tool_args,
            ),
            "forced_first_step": {
                "thought": "用户已补齐 Adela 所需参数，继续执行工具。",
                "decision_type": decision_type_tool,
                "action": candidate_tool,
                "action_input": tool_args,
                "final_answer": "",
            },
        }

    known_slots, tool_args, missing_slots = _merge_slot_update(
        task_state=task_state,
        user_text=user_text,
        image_path=image_path,
    )

    if missing_slots:
        next_pending = {
            "status": "pending",
            "source": str(pending.get("source") or "").strip() or "tool_precondition",
            "query_id": query_id,
            "thread_id": thread_id,
            "clarification_question": _tool_precondition_question(
                action=candidate_tool,
                missing_slots=missing_slots,
                qwen_detection_action=qwen_detection_action,
                rex_detection_action=rex_detection_action,
                pipeline_eval_action=pipeline_eval_action,
                flux_action=flux_action,
                adela_cli_action=adela_cli_action,
            ) or "当前还缺少继续执行所需的信息，请补充。",
            "task_state": {
                "candidate_tool": candidate_tool,
                "known_slots": known_slots,
                "missing_slots": missing_slots,
                "tool_args": tool_args,
                "original_user_text": original_user_text,
            },
        }
        return {
            "action": ACTION_STILL_PENDING,
            "reply_type": reply_type,
            "query_id": query_id,
            "thread_id": thread_id,
            "pending_clarification": next_pending,
            "clarification_question": str(next_pending.get("clarification_question") or ""),
        }

    return {
        "action": ACTION_DIRECT_TOOL,
        "reply_type": reply_type,
        "query_id": query_id,
        "thread_id": thread_id,
        "effective_text": _merge_clarification_text(original_user_text, user_text),
        "forced_first_step": {
            "thought": "用户已补充澄清信息，继续执行上次已确定的工具动作。",
            "decision_type": decision_type_tool,
            "action": candidate_tool,
            "action_input": tool_args,
            "final_answer": "",
        },
    }
