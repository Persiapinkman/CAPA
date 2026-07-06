"""
Demo agent brain module.

主要功能：
- 承载 Planner/Answerer 的核心决策逻辑与 LLM 调用封装。
- 将 ReAct 规划步骤标准化为结构化 action（Thought -> Action -> Action Input）。
- 汇总工具动作的兼容映射，并提供统一的 action 归一化能力。
- 提供 `AgentOrchestrator` 主控循环，串联 memory、prompt 与 tool execution。

主要模块（核心能力）：
- Prompt 构建适配：调用 `prompts.py` 的模板函数。
- Planner 路由：`choose_agent_step_llm` / `choose_agent_step_with_fallback`。
- Answer 生成：answerer 与通用问答输出封装。
- 编排层：`AgentOrchestrator`。
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import clarification as clarification_state  # noqa: E402
from util.vlm_service import VLMService  # noqa: E402
from prompts import (  # noqa: E402
    build_agent_system_prompt as prompt_build_agent_system_prompt,
    build_agent_user_prompt as prompt_build_agent_user_prompt,
    build_answer_system_prompt as prompt_build_answer_system_prompt,
    build_answer_resolution_judger_system_prompt as prompt_build_answer_resolution_judger_system_prompt,
    build_answer_resolution_judger_user_prompt as prompt_build_answer_resolution_judger_user_prompt,
    build_answer_user_prompt as prompt_build_answer_user_prompt,
    build_rewrite_query_system_prompt as prompt_build_rewrite_query_system_prompt,
    build_rewrite_query_user_prompt as prompt_build_rewrite_query_user_prompt,
)
from tools import registry as tool_registry  # noqa: E402
from tools import schemas as tool_schemas  # noqa: E402

# Native agent tool names
TOOL_RAG_ANSWER = tool_registry.TOOL_RAG_ANSWER
TOOL_RE_QUESTION = tool_registry.TOOL_RE_QUESTION
TOOL_ANSWERER = tool_registry.TOOL_ANSWERER
TOOL_EVIDENCE_SYNTHESIS_ANSWER = getattr(
    tool_registry,
    "TOOL_EVIDENCE_SYNTHESIS_ANSWER",
    "evidence_synthesis_answer",
)
TOOL_FLUX_IMAGE_GENERATION = tool_registry.TOOL_FLUX_IMAGE_GENERATION
TOOL_QWEN_DETECTION = tool_registry.TOOL_QWEN_DETECTION
TOOL_REXOMNI_DETECTION = tool_registry.TOOL_REXOMNI_DETECTION
TOOL_PIPELINE_EVAL = tool_registry.TOOL_PIPELINE_EVAL
TOOL_MIGRATION_ADVISOR = tool_registry.TOOL_MIGRATION_ADVISOR
TOOL_ADELA_CLI_EVAL = tool_registry.TOOL_ADELA_CLI_EVAL
ACTION_FINAL_ANSWER = tool_registry.ACTION_FINAL_ANSWER

# Legacy action ids kept for compatibility with the current demo executor.
ACTION_RAG_ANSWER = tool_registry.ACTION_RAG_ANSWER
ACTION_RE_QUESTION = tool_registry.ACTION_RE_QUESTION
ACTION_ANSWERER = tool_registry.ACTION_ANSWERER
ACTION_FLUX_IMAGE_GENERATION = tool_registry.ACTION_FLUX_IMAGE_GENERATION
ACTION_QWEN_OPEN_SET_DETECTION = tool_registry.ACTION_QWEN_OPEN_SET_DETECTION
ACTION_REXOMNI_OPEN_SET_DETECTION = tool_registry.ACTION_REXOMNI_OPEN_SET_DETECTION
ACTION_TARGET_DETECTION_EVALUATION = tool_registry.ACTION_TARGET_DETECTION_EVALUATION
ACTION_MIGRATION_ADVISOR = tool_registry.ACTION_MIGRATION_ADVISOR
ACTION_ADELA_CLI_EVAL = tool_registry.ACTION_ADELA_CLI_EVAL

DEMO_ROUTE_MODEL = os.environ.get("DEMO_ROUTE_MODEL", "Qwen3.5-35B-A3B")
DEMO_ROUTE_API_BASE = os.environ.get("DEMO_ROUTE_API_BASE", "http://10.111.32.253:8000/v1")
DEMO_ROUTE_API_KEY = os.environ.get("DEMO_ROUTE_API_KEY", "token.sdc@2026")
DEMO_ANSWER_MODEL = os.environ.get("DEMO_ANSWER_MODEL", "Qwen3.5-4B")
DEMO_ANSWER_API_BASE = os.environ.get("DEMO_ANSWER_API_BASE", DEMO_ROUTE_API_BASE)
DEMO_ANSWER_API_KEY = os.environ.get("DEMO_ANSWER_API_KEY", DEMO_ROUTE_API_KEY)
DEMO_REQUESTION_MODEL = (
    os.environ.get("DEMO_REQUESTION_MODEL")
    or os.environ.get("DEMO_REQUESTION_MODEL")
    or "Qwen3.5-4B"
)
AGENT_MAX_STEPS = int(os.environ.get("DEMO_AGENT_MAX_STEPS", "10"))
LLM_DEBUG_ENABLED = str(os.environ.get("DEMO_LLM_DEBUG_ENABLED", "1")).strip().lower() not in {
    "0",
    "false",
    "off",
    "no",
}
RAG_RESOLUTION_JUDGE_ENABLED = str(
    os.environ.get("DEMO_RAG_RESOLUTION_JUDGE_ENABLED", "0")
).strip().lower() not in {
    "0",
    "false",
    "off",
    "no",
}
KNOWLEDGE_BASE_SCORE_THRESHOLD = float(
    os.environ.get("DEMO_KB_ANSWER_THRESHOLD", "0.97")
)
LLM_DEBUG_DIR = ROOT / "demo" / "llm_debug"
BEIJING_TZ = timezone(timedelta(hours=8))
DECISION_TYPE_TOOL = "tool"
DECISION_TYPE_END = "end"
DECISION_TYPE_CLARIFY = "clarify"
VALID_DECISION_TYPES = {DECISION_TYPE_TOOL, DECISION_TYPE_END, DECISION_TYPE_CLARIFY}
END_REASON_RECHECK_DONE = "recheck_done"
END_REASON_MEMORY_HIT = "memory_hit"
VALID_END_REASONS = {END_REASON_RECHECK_DONE, END_REASON_MEMORY_HIT}

AGENT_STEP_RESPONSE_FORMAT = tool_schemas.build_agent_step_response_format(
    [*tool_registry.get_declared_tool_names(), ACTION_FINAL_ANSWER]
)
ANSWERER_INPUT_SCHEMA = tool_schemas.ANSWERER_INPUT_SCHEMA
ANSWERER_OUTPUT_SCHEMA = tool_schemas.ANSWERER_OUTPUT_SCHEMA
ANSWERER_RESPONSE_FORMAT = tool_schemas.ANSWERER_RESPONSE_FORMAT
REWRITE_QUERY_RESPONSE_FORMAT = tool_schemas.REWRITE_QUERY_RESPONSE_FORMAT
ANSWER_RESOLUTION_JUDGER_RESPONSE_FORMAT = tool_schemas.ANSWER_RESOLUTION_JUDGER_RESPONSE_FORMAT

VALID_AGENT_ACTIONS = set(tool_registry.get_declared_tool_names()) | {ACTION_FINAL_ANSWER}


def get_tool_schemas() -> list[dict]:
    return tool_registry.get_tool_schemas()


def flow_for_action(action: str) -> str:
    return tool_registry.flow_for_action(action)


def to_legacy_action(action: str) -> str:
    return tool_registry.to_legacy_action(action)


def normalize_agent_action(action: str) -> str:
    return tool_registry.normalize_tool_action(action)


def _extract_first_json_object(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("empty response")

    start = raw.find("{")
    if start < 0:
        raise ValueError("no json object found")

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(raw)):
        ch = raw[idx]
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
                return raw[start : idx + 1]
    raise ValueError("unterminated json object")


def _repair_json_text(text: str) -> list[str]:
    raw = str(text or "").strip()
    candidates: list[str] = []
    if not raw:
        return candidates

    candidates.append(raw)

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.S)
    if fence_match:
        candidates.append(fence_match.group(1).strip())

    try:
        candidates.append(_extract_first_json_object(raw))
    except Exception:
        pass

    expanded: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if not item or item in seen:
            continue
        seen.add(item)
        expanded.append(item)

        trimmed = re.sub(r",(\s*[}\]])", r"\1", item)
        trimmed = trimmed.strip()
        if trimmed and trimmed not in seen:
            seen.add(trimmed)
            expanded.append(trimmed)

    return expanded


def _normalize_agent_step_data(data: dict) -> dict:
    if not isinstance(data, dict):
        raise ValueError("planner output is not an object")

    raw_decision_type = str(data.get("decision_type") or "").strip().lower()
    raw_end_reason = str(data.get("end_reason") or "").strip().lower()
    action = normalize_agent_action(str(data.get("action") or "").strip())
    thought = str(data.get("thought") or "").strip()
    final_answer = str(data.get("final_answer") or "").strip()
    clarification_question = str(data.get("clarification_question") or "").strip()
    action_input = data.get("action_input")
    if not isinstance(action_input, dict):
        action_input = {}

    if raw_decision_type in VALID_DECISION_TYPES:
        decision_type = raw_decision_type
    else:
        # 兼容旧协议：旧版 decision_type=plain 及 action=final_answer 都视为 tool(answerer)。
        if raw_decision_type == "plain" or action == ACTION_FINAL_ANSWER:
            decision_type = DECISION_TYPE_TOOL
        else:
            decision_type = DECISION_TYPE_TOOL

    if decision_type == DECISION_TYPE_TOOL:
        if action == ACTION_FINAL_ANSWER:
            action = TOOL_ANSWERER
            action_input = {}
        if action not in VALID_AGENT_ACTIONS or action == ACTION_FINAL_ANSWER:
            raise ValueError(f"invalid tool action: {action}")
        end_reason = ""
        final_answer = str(data.get("final_answer") or "").strip()
        clarification_question = ""
    elif decision_type == DECISION_TYPE_END:
        action = ACTION_FINAL_ANSWER
        action_input = {}
        clarification_question = ""
        final_answer = str(data.get("final_answer") or "").strip()
        end_reason = (
            raw_end_reason
            if raw_end_reason in VALID_END_REASONS
            else END_REASON_MEMORY_HIT
        )
    else:
        action = ""
        action_input = {}
        end_reason = ""
        final_answer = ""
        if not clarification_question:
            raise ValueError("clarification_question is required when decision_type=clarify")

    return {
        "thought": thought,
        "decision_type": decision_type,
        "action": action,
        "action_input": action_input,
        "end_reason": end_reason,
        "final_answer": final_answer,
        "clarification_question": clarification_question,
    }


def _parse_agent_step_response(raw: str) -> dict:
    last_exc: Exception | None = None
    for candidate in _repair_json_text(raw):
        try:
            data = json.loads(candidate)
            return _normalize_agent_step_data(data)
        except Exception as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise ValueError("planner response is empty")


def _parse_rewrite_query_response(raw: str) -> str:
    last_exc: Exception | None = None
    for candidate in _repair_json_text(raw):
        try:
            data = json.loads(candidate)
            if not isinstance(data, dict):
                raise ValueError("rewrite output is not an object")
            rewritten_query = str(data.get("rewritten_query") or "").strip()
            if not rewritten_query:
                raise ValueError("rewritten_query is empty")
            return rewritten_query
        except Exception as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise ValueError("rewrite response is empty")


def build_agent_system_prompt(*, max_steps: int = AGENT_MAX_STEPS) -> str:
    tools_json = json.dumps(get_tool_schemas(), ensure_ascii=False, indent=2)
    return prompt_build_agent_system_prompt(max_steps=max_steps, tools_json=tools_json)


def build_agent_user_prompt(
    text: str,
    image_path: str | None,
    *,
    planner_context: dict | None = None,
    step_index: int = 1,
    max_steps: int = AGENT_MAX_STEPS,
) -> str:
    return prompt_build_agent_user_prompt(
        text,
        image_path,
        planner_context=planner_context,
        step_index=step_index,
        max_steps=max_steps,
    )


def build_answer_system_prompt() -> str:
    return prompt_build_answer_system_prompt()


def build_answer_user_prompt(
    user_query: str,
    *,
    evidence: dict | None = None,
    mode: str = "direct",
) -> str:
    return prompt_build_answer_user_prompt(
        user_query,
        evidence=evidence,
        mode=mode,
    )


def build_rewrite_query_system_prompt() -> str:
    return prompt_build_rewrite_query_system_prompt()


def build_rewrite_query_user_prompt(
    *,
    query: str,
    rewrite_reason: str = "",
    context_hint: str = "",
    retrieval_round: int = 1,
) -> str:
    return prompt_build_rewrite_query_user_prompt(
        query=query,
        rewrite_reason=rewrite_reason,
        context_hint=context_hint,
        retrieval_round=retrieval_round,
    )


def build_answer_resolution_judger_system_prompt() -> str:
    return prompt_build_answer_resolution_judger_system_prompt()


def build_answer_resolution_judger_user_prompt(
    *,
    user_query: str,
    candidate_answer: str,
    retrieved_chunks: list[dict] | None = None,
) -> str:
    return prompt_build_answer_resolution_judger_user_prompt(
        user_query=user_query,
        candidate_answer=candidate_answer,
        retrieved_chunks=retrieved_chunks,
    )


def _parse_answer_resolution_judger_response(raw: str) -> dict:
    last_exc: Exception | None = None
    for candidate in _repair_json_text(raw):
        try:
            data = json.loads(candidate)
            if not isinstance(data, dict):
                raise ValueError("judger output is not an object")
            return {
                "resolved": bool(data.get("resolved")),
                "reason": str(data.get("reason") or "").strip(),
                "clarification_question": str(data.get("clarification_question") or "").strip(),
            }
        except Exception as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise ValueError("judger output is empty")


def _sanitize_final_answer(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    text = re.sub(r"```[\s\S]*?```", "", text).strip()

    def _is_cot_like_paragraph(p: str) -> bool:
        lower = p.lower()
        cot_markers = (
            "thinking process",
            "reasoning:",
            "analysis:",
            "analyze the request",
            "chain of thought",
            "let me think",
            "let's think",
            "final check",
            "final polish",
            "revised plan",
            "step 1",
            "step 2",
            "step 3",
            "推理过程",
            "思考过程",
            "分析过程",
            "下面是分析",
        )
        return any(m in lower for m in cot_markers)

    def _extract_tail_chinese_answer(s: str) -> str:
        paras = [p.strip() for p in re.split(r"\n\s*\n+", s) if p.strip()]
        if not paras:
            return ""
        good: list[str] = []
        for p in reversed(paras):
            if _is_cot_like_paragraph(p):
                if good:
                    break
                continue
            chinese_like = sum(1 for ch in p if "\u4e00" <= ch <= "\u9fff")
            if chinese_like >= 10:
                good.append(p)
                if len(good) >= 2:
                    break
                continue
            if good:
                break
        if not good:
            return ""
        good.reverse()
        return "\n\n".join(good).strip()

    markers = [
        "Thinking Process:",
        "Thinking process:",
        "Reasoning:",
        "Analysis:",
        "Let's think",
        "Let me think",
        "推理过程：",
        "思考过程：",
        "分析过程：",
        "下面是分析：",
    ]
    cut_positions = [text.find(marker) for marker in markers if text.find(marker) >= 0]
    if cut_positions:
        first_pos = min(cut_positions)
        # marker 出现在首位时，不能直接截断为空；优先尝试提取末尾中文正文。
        if first_pos == 0:
            recovered = _extract_tail_chinese_answer(text)
            if recovered:
                return recovered
        else:
            text = text[:first_pos].strip()

    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    if lines and (lines[0].startswith("Thinking Process") or lines[0].startswith("推理过程")):
        lines = []

    if not lines:
        return ""

    last_line = lines[-1]
    if len(lines) >= 2:
        chinese_like = sum(1 for ch in last_line if "\u4e00" <= ch <= "\u9fff")
        if chinese_like >= 8 and len(last_line) <= 240:
            return last_line

    joined = "\n".join(lines).strip()
    if len(joined) > 400:
        joined = joined[:400].rstrip() + "…"
    return joined


def _safe_debug_name(value: str, fallback: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("._")
    return text or fallback


def _bj_date_prefix() -> str:
    return datetime.now(BEIJING_TZ).strftime("%Y%m%d")


def _write_llm_debug_record(
    *,
    session_id: str,
    run_stamp: str,
    stage: str,
    step_index: int,
    payload: dict,
) -> None:
    if not LLM_DEBUG_ENABLED:
        return
    try:
        sid = _safe_debug_name(session_id, "unknown_session")
        stamp = _safe_debug_name(run_stamp, "unknown_run")
        stage_name = _safe_debug_name(stage, "unknown_stage")
        seq = max(1, int(step_index or 1))
        date_prefix = _bj_date_prefix()
        stamp_tail = re.sub(r"^\d{8}_", "", stamp).strip("_")
        if not stamp_tail:
            stamp_tail = "unknown_run"
        target_dir = LLM_DEBUG_DIR / date_prefix
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"{date_prefix}_{sid}_{stamp_tail}_{seq:02d}_{stage_name}.json"
        body = {
            "session_id": session_id,
            "run_stamp": run_stamp,
            "stage": stage,
            "step_index": seq,
            "created_at": datetime.now(BEIJING_TZ).isoformat(timespec="seconds"),
            "payload": payload,
        }
        file_path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")

    except Exception:
        # 调试日志不应影响主流程。
        return


def choose_agent_step_llm(
    text: str,
    image_path: str | None,
    *,
    planner_context: dict | None = None,
    step_index: int = 1,
    max_steps: int = AGENT_MAX_STEPS,
    model: str | None = None,
    debug_meta: dict | None = None,
) -> dict:
    use_model = model or DEMO_ROUTE_MODEL
    system_prompt = build_agent_system_prompt(max_steps=max_steps)
    user_prompt = build_agent_user_prompt(
        text,
        image_path,
        planner_context=planner_context,
        step_index=step_index,
        max_steps=max_steps,
    )
    planner_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    image_paths = [image_path] if image_path and Path(image_path).is_file() else None
    debug_info = debug_meta if isinstance(debug_meta, dict) else {}
    pctx = planner_context if isinstance(planner_context, dict) else {}
    session_id = str(
        debug_info.get("session_id") or pctx.get("session_id") or ""
    ).strip() or "unknown_session"
    run_stamp = str(debug_info.get("run_stamp") or "").strip() or "unknown_run"
    _write_llm_debug_record(
        session_id=session_id,
        run_stamp=run_stamp,
        stage="planner_request",
        step_index=step_index,
        payload={
            "model": use_model,
            "response_format": AGENT_STEP_RESPONSE_FORMAT,
            "image_paths": image_paths or [],
            "messages": planner_messages,
            "run_dir": str(debug_info.get("run_dir") or ""),
        },
    )
    vlm = VLMService(api_key=DEMO_ROUTE_API_KEY, base_url=DEMO_ROUTE_API_BASE.rstrip("/"))
    raw = vlm.generate_text(
        messages=planner_messages,
        image_paths=image_paths,
        model=use_model,
        response_format=AGENT_STEP_RESPONSE_FORMAT,
    )
    _write_llm_debug_record(
        session_id=session_id,
        run_stamp=run_stamp,
        stage="planner_response",
        step_index=step_index,
        payload={"raw_response": raw, "run_dir": str(debug_info.get("run_dir") or "")},
    )
    try:
        return _parse_agent_step_response(raw)
    except Exception as first_exc:
        retry_messages = list(planner_messages)
        retry_messages.append(
            {
                "role": "user",
                "content": (
                    f"你的上一条输出不符合 JSON 格式约束或业务规则，解析失败。\n"
                    f"【解析错误详情】：{str(first_exc)}\n\n"
                    "请立即反思错误原因，并严格按照给定的 JSON Schema 重新输出合法的结果。不要包含任何 markdown 语法或额外文本。"
                ),
            }
        )
        _write_llm_debug_record(
            session_id=session_id,
            run_stamp=run_stamp,
            stage="planner_retry_request",
            step_index=step_index,
            payload={
                "model": use_model,
                "response_format": AGENT_STEP_RESPONSE_FORMAT,
                "image_paths": image_paths or [],
                "messages": retry_messages,
                "parse_error": str(first_exc),
                "run_dir": str(debug_info.get("run_dir") or ""),
            },
        )
        retry_raw = vlm.generate_text(
            messages=retry_messages,
            image_paths=image_paths,
            model=use_model,
            response_format=AGENT_STEP_RESPONSE_FORMAT,
        )
        _write_llm_debug_record(
            session_id=session_id,
            run_stamp=run_stamp,
            stage="planner_retry_response",
            step_index=step_index,
            payload={"raw_response": retry_raw, "run_dir": str(debug_info.get("run_dir") or "")},
        )
        try:
            return _parse_agent_step_response(retry_raw)
        except Exception as retry_exc:
            raise ValueError(
                f"planner schema parse failed after retry: first={first_exc}; retry={retry_exc}"
            ) from retry_exc


def generate_final_answer_llm(
    *,
    answerer_input: dict,
    mode: str = "direct",
    model: str | None = None,
    debug_meta: dict | None = None,
    emit_chunk: Callable[[str], None] | None = None,
) -> str:
    answerer_input = _normalize_answerer_input(answerer_input)
    _validate_answerer_input(answerer_input)
    use_model = model or DEMO_ANSWER_MODEL
    system_prompt = build_answer_system_prompt()
    user_prompt = build_answer_user_prompt(
        str(answerer_input.get("user_query") or "").strip(),
        evidence=answerer_input.get("evidence") if isinstance(answerer_input.get("evidence"), dict) else {},
        mode=mode,
    )
    prompt = system_prompt + "\n\n" + user_prompt
    debug_info = debug_meta if isinstance(debug_meta, dict) else {}
    session_id = str(debug_info.get("session_id") or "").strip() or "unknown_session"
    run_stamp = str(debug_info.get("run_stamp") or "").strip() or "unknown_run"
    step_index = int(debug_info.get("step_index") or 1)
    _write_llm_debug_record(
        session_id=session_id,
        run_stamp=run_stamp,
        stage=str(debug_info.get("stage") or "answerer_request"),
        step_index=step_index,
        payload={
            "model": use_model,
            "response_format": ANSWERER_RESPONSE_FORMAT,
            "stream": True,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "combined_prompt": prompt,
            "mode": mode,
            "answerer_input": answerer_input,
            "run_dir": str(debug_info.get("run_dir") or ""),
        },
    )
    vlm = VLMService(api_key=DEMO_ANSWER_API_KEY, base_url=DEMO_ANSWER_API_BASE.rstrip("/"))
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    raw_parts: list[str] = []
    streamed_text = ""
    completion_stream = vlm.client.chat.completions.create(
        model=use_model,
        messages=messages,
        response_format=ANSWERER_RESPONSE_FORMAT,
        stream=True,
    )
    for chunk in completion_stream:
        try:
            piece = chunk.choices[0].delta.content
        except Exception:
            piece = None
        if not isinstance(piece, str) or not piece:
            continue
        raw_parts.append(piece)
        if emit_chunk is None:
            continue
        now_text = _extract_streaming_final_answer_text("".join(raw_parts))
        if len(now_text) > len(streamed_text):
            emit_chunk(now_text[len(streamed_text) :])
            streamed_text = now_text
    raw = "".join(raw_parts).strip()
    _write_llm_debug_record(
        session_id=session_id,
        run_stamp=run_stamp,
        stage=str(debug_info.get("stage_response") or "answerer_response"),
        step_index=step_index,
        payload={"raw_response": raw, "run_dir": str(debug_info.get("run_dir") or "")},
    )
    if not raw:
        raise ValueError("answerer streaming output is empty")
    try:
        final_text = _parse_answerer_output_response(raw)
    except Exception:
        text = _sanitize_final_answer(raw)
        if text:
            final_text = text
        else:
            raise
    if emit_chunk is not None:
        if len(final_text) > len(streamed_text):
            emit_chunk(final_text[len(streamed_text) :])
    return final_text


def _validate_answerer_input(answerer_input: dict) -> None:
    if not isinstance(answerer_input, dict):
        raise ValueError("answerer_input must be object")
    user_query = str(answerer_input.get("user_query") or "").strip()
    if not user_query:
        raise ValueError("answerer_input.user_query is required")
    evidence = answerer_input.get("evidence")
    if evidence is None:
        return
    if not isinstance(evidence, dict):
        raise ValueError("answerer_input.evidence must be object|null")
    if not isinstance(evidence.get("retrieved_chunks"), list):
        raise ValueError("answerer_input.evidence.retrieved_chunks must be array")
    if not isinstance(evidence.get("query_trajectories"), list):
        raise ValueError("answerer_input.evidence.query_trajectories must be array")


def _normalize_answerer_input(answerer_input: dict | None) -> dict:
    src = answerer_input if isinstance(answerer_input, dict) else {}
    evidence_src = src.get("evidence")
    evidence_obj = evidence_src if isinstance(evidence_src, dict) else {}
    normalized = {
        "user_query": str(src.get("user_query") or "").strip(),
        "evidence": {
            "retrieved_chunks": (
                evidence_obj.get("retrieved_chunks")
                if isinstance(evidence_obj.get("retrieved_chunks"), list)
                else []
            ),
            "query_trajectories": (
                evidence_obj.get("query_trajectories")
                if isinstance(evidence_obj.get("query_trajectories"), list)
                else []
            ),
        },
    }
    return normalized


def _parse_answerer_output_response(raw: str) -> str:
    last_exc: Exception | None = None
    for candidate in _repair_json_text(raw):
        try:
            data = json.loads(candidate)
            if not isinstance(data, dict):
                raise ValueError("answerer output is not object")
            final_answer = str(data.get("final_answer") or "").strip()
            if not final_answer:
                raise ValueError("final_answer is empty")
            return final_answer
        except Exception as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise ValueError("answerer output is empty")


def generate_final_answer_with_fallback(
    *,
    answerer_input: dict,
    mode: str = "direct",
    model: str | None = None,
    debug_meta: dict | None = None,
    emit_chunk: Callable[[str], None] | None = None,
) -> str:
    debug_info = debug_meta if isinstance(debug_meta, dict) else {}
    session_id = str(debug_info.get("session_id") or "").strip() or "unknown_session"
    run_stamp = str(debug_info.get("run_stamp") or "").strip() or "unknown_run"
    step_index = int(debug_info.get("step_index") or 1)
    answer_error: str = ""

    try:
        answer = generate_final_answer_llm(
            answerer_input=answerer_input,
            mode=mode,
            model=model,
            debug_meta=debug_meta,
            emit_chunk=emit_chunk,
        )
        if answer:
            return answer
    except Exception as exc:
        answer_error = str(exc or "").strip()

    _write_llm_debug_record(
        session_id=session_id,
        run_stamp=run_stamp,
        stage="answerer_fallback",
        step_index=step_index,
        payload={
            "mode": mode,
            "reason": "empty_after_sanitize_or_exception",
            "error": answer_error,
            "has_answerer_input": bool(isinstance(answerer_input, dict) and answerer_input),
            "run_dir": str(debug_info.get("run_dir") or ""),
        },
    )

    return "当前已完成分析，但暂时无法生成更完整的答复。"


def normalize_knowledge_base_score(value) -> float:
    """将 knowledge_base_fully_answered 规范为 [0, 1]；缺失或无法解析视为 0（未命中）。"""
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, score))
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"true", "1", "yes", "y"}:
            return 1.0
        if raw in {"false", "0", "no", "n", ""}:
            return 0.0
        try:
            return normalize_knowledge_base_score(float(raw))
        except ValueError:
            return 0.0
    return 0.0


def is_rag_miss(observation: dict | None) -> bool:
    obs = observation if isinstance(observation, dict) else {}
    score = normalize_knowledge_base_score(obs.get("knowledge_base_fully_answered"))
    return score < KNOWLEDGE_BASE_SCORE_THRESHOLD


def rewrite_query_llm(
    *,
    query: str,
    rewrite_reason: str = "",
    context_hint: str = "",
    retrieval_round: int = 1,
    model: str | None = None,
) -> str:
    use_model = model or DEMO_REQUESTION_MODEL
    system_prompt = build_rewrite_query_system_prompt()
    user_prompt = build_rewrite_query_user_prompt(
        query=query,
        rewrite_reason=rewrite_reason,
        context_hint=context_hint,
        retrieval_round=retrieval_round,
    )
    prompt = system_prompt + "\n\n" + user_prompt
    vlm = VLMService(api_key=DEMO_ROUTE_API_KEY, base_url=DEMO_ROUTE_API_BASE.rstrip("/"))
    raw = vlm.generate_text(
        prompt=prompt,
        model=use_model,
        response_format=REWRITE_QUERY_RESPONSE_FORMAT,
    )
    try:
        text = _parse_rewrite_query_response(raw)
    except Exception:
        text = _sanitize_final_answer(raw)
    text = re.sub(r"\s+", " ", str(text or "").strip())
    return text[:300].strip()


def rewrite_query_with_fallback(
    *,
    query: str,
    rewrite_reason: str = "",
    context_hint: str = "",
    retrieval_round: int = 1,
    model: str | None = None,
) -> str:
    source = str(query or "").strip()
    if not source:
        return ""
    try:
        rewritten = rewrite_query_llm(
            query=source,
            rewrite_reason=rewrite_reason,
            context_hint=context_hint,
            retrieval_round=retrieval_round,
            model=model,
        )
        if rewritten:
            return rewritten
    except Exception:
        pass
    return source


def judge_answer_resolution_llm(
    *,
    user_query: str,
    candidate_answer: str,
    retrieved_chunks: list[dict] | None = None,
    model: str | None = None,
) -> dict:
    use_model = model or DEMO_ANSWER_MODEL
    system_prompt = build_answer_resolution_judger_system_prompt()
    user_prompt = build_answer_resolution_judger_user_prompt(
        user_query=user_query,
        candidate_answer=candidate_answer,
        retrieved_chunks=retrieved_chunks,
    )
    prompt = system_prompt + "\n\n" + user_prompt
    vlm = VLMService(api_key=DEMO_ANSWER_API_KEY, base_url=DEMO_ANSWER_API_BASE.rstrip("/"))
    raw = vlm.generate_text(
        prompt=prompt,
        model=use_model,
        response_format=ANSWER_RESOLUTION_JUDGER_RESPONSE_FORMAT,
    )
    return _parse_answer_resolution_judger_response(raw)


def judge_answer_resolution_with_fallback(
    *,
    user_query: str,
    candidate_answer: str,
    retrieved_chunks: list[dict] | None = None,
    model: str | None = None,
) -> dict:
    try:
        judged = judge_answer_resolution_llm(
            user_query=user_query,
            candidate_answer=candidate_answer,
            retrieved_chunks=retrieved_chunks,
            model=model,
        )
        if isinstance(judged, dict):
            return judged
    except Exception:
        pass
    return {
        "resolved": True,
        "reason": "judger_fallback_resolved",
        "clarification_question": "",
    }


def choose_agent_step_with_fallback(
    text: str,
    image_path: str | None,
    *,
    planner_context: dict | None = None,
    step_index: int = 1,
    max_steps: int = AGENT_MAX_STEPS,
    model: str | None = None,
    debug_meta: dict | None = None,
) -> dict:
    try:
        return choose_agent_step_llm(
            text,
            image_path,
            planner_context=planner_context,
            step_index=step_index,
            max_steps=max_steps,
            model=model,
            debug_meta=debug_meta,
        )
    except Exception as exc:
        return {
            "thought": f"Agent 决策异常：{exc}。本轮停止工具调用，直接向用户说明。",
            "decision_type": DECISION_TYPE_TOOL,
            "action": TOOL_ANSWERER,
            "action_input": {},
            "final_answer": "当前决策失败，请重试一次；如果问题涉及图片，请重新附带图片发送。",
        }


def choose_route_with_fallback(
    text: str,
    image_path: str | None,
    *,
    model: str | None = None,
) -> dict:
    step = choose_agent_step_with_fallback(
        text,
        image_path,
        planner_context={},
        step_index=1,
        max_steps=AGENT_MAX_STEPS,
        model=model,
    )
    action = step.get("action")
    decision_type = str(step.get("decision_type") or DECISION_TYPE_TOOL).strip().lower()
    if decision_type == DECISION_TYPE_CLARIFY:
        return {
            "action": ACTION_ANSWERER,
            "reason": str(step.get("thought") or "").strip(),
            "direct_reply": str(step.get("clarification_question") or "").strip(),
            "detection_label": "",
            "flux_user_prompt": "",
            "flux_num_images": 1,
        }
    if decision_type == DECISION_TYPE_END or action == ACTION_FINAL_ANSWER:
        return {
            "action": ACTION_RAG_ANSWER,
            "reason": str(step.get("thought") or "").strip(),
            "direct_reply": str(step.get("final_answer") or "").strip(),
            "detection_label": "",
            "flux_user_prompt": "",
            "flux_num_images": 1,
        }

    tool_action = to_legacy_action(str(action or "").strip())
    action_input = step.get("action_input") or {}
    flux_num_images = action_input.get("num_images", 1)
    try:
        flux_num_images = max(1, min(5, int(flux_num_images)))
    except (TypeError, ValueError):
        flux_num_images = 1

    return {
        "action": tool_action,
        "reason": str(step.get("thought") or "").strip(),
        "direct_reply": "",
        "general_query": str(action_input.get("query") or "").strip(),
        "detection_label": str(action_input.get("label") or "").strip(),
        "flux_user_prompt": str(action_input.get("task_text") or "").strip(),
        "flux_num_images": flux_num_images,
    }


def collect_rag_observations_from_session(session: dict, *, limit: int = 8) -> list[dict]:
    raw_ledger = session.get("raw_ledger")
    if not isinstance(raw_ledger, list):
        return []
    active_tid = str(session.get("active_thread_id") or "").strip()
    out: list[dict] = []
    for item in raw_ledger:
        if not isinstance(item, dict):
            continue
        if active_tid and str(item.get("thread_id") or "").strip() not in ("", active_tid):
            continue
        if str(item.get("event_type") or "").upper() != "OBSERVATION":
            continue
        payload = item.get("payload")
        if not isinstance(payload, dict):
            continue
        action = normalize_agent_action(str(payload.get("_action") or item.get("observation") or "").strip())
        if action != TOOL_RAG_ANSWER:
            continue
        out.append(
            {
                "query": str((payload.get("_action_input") or {}).get("query") or "").strip(),
                "answer": str(payload.get("answer") or "").strip(),
                "summary": str(payload.get("summary") or "").strip(),
                "references": payload.get("references") if isinstance(payload.get("references"), list) else [],
                "retrieved_chunks": payload.get("retrieved_chunks")
                if isinstance(payload.get("retrieved_chunks"), list)
                else [],
            }
        )
    return out[-max(1, limit) :]


def _load_rag_retrieved_chunks_from_run_dir(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / "rag_response.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    chunks = data.get("retrieved_chunks")
    if not isinstance(chunks, list):
        return []
    return [item for item in chunks if isinstance(item, dict)]


def _trim_text(value: str, limit: int = 400) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _split_text_for_stream(text: str, chunk_size: int = 36) -> list[str]:
    raw = str(text or "")
    if not raw:
        return []
    size = max(8, int(chunk_size or 36))
    return [raw[i : i + size] for i in range(0, len(raw), size)]


def _decode_partial_json_string(content: str) -> str:
    raw = str(content or "")
    if not raw:
        return ""
    for end in range(len(raw), -1, -1):
        candidate = raw[:end]
        try:
            return str(json.loads(f'"{candidate}"'))
        except Exception:
            continue
    return raw.replace('\\"', '"').replace("\\n", "\n")


def _extract_streaming_final_answer_text(raw: str) -> str:
    text = str(raw or "")
    if not text:
        return ""
    key_pos = text.find('"final_answer"')
    if key_pos < 0:
        return ""
    colon_pos = text.find(":", key_pos)
    if colon_pos < 0:
        return ""
    i = colon_pos + 1
    while i < len(text) and text[i] in " \t\r\n":
        i += 1
    if i >= len(text) or text[i] != '"':
        return ""
    i += 1
    buf: list[str] = []
    escape = False
    while i < len(text):
        ch = text[i]
        if escape:
            buf.append(ch)
            escape = False
            i += 1
            continue
        if ch == "\\":
            escape = True
            buf.append(ch)
            i += 1
            continue
        if ch == '"':
            break
        buf.append(ch)
        i += 1
    return _decode_partial_json_string("".join(buf))


def _observation_without_answer(observation: dict | None) -> dict:
    obs = dict(observation) if isinstance(observation, dict) else {}
    obs.pop("answer", None)
    return obs


def _compact_trajectory_observation_for_answerer(observation: dict | None) -> dict:
    obs = observation if isinstance(observation, dict) else {}
    compact: dict = {}
    summary = str(obs.get("summary") or "").strip()
    final_answer = str(obs.get("final_answer") or "").strip()
    answer = str(obs.get("answer") or "").strip()
    if summary:
        compact["summary"] = summary
    elif final_answer:
        compact["summary"] = final_answer
    elif answer:
        compact["summary"] = answer
    references = obs.get("references")
    if isinstance(references, list) and references:
        compact["references"] = references
    return compact


def _compact_query_trajectories_for_answerer(query_trajectories: list) -> list[dict]:
    if not isinstance(query_trajectories, list):
        return []
    out: list[dict] = []
    for item in query_trajectories:
        if not isinstance(item, dict):
            continue
        steps_in = item.get("steps")
        compact_steps: list[dict] = []
        if isinstance(steps_in, list):
            for st in steps_in:
                if not isinstance(st, dict):
                    continue
                compact_steps.append(
                    {
                        "step_id": str(st.get("step_id") or "").strip(),
                        "action": str(st.get("action") or "").strip(),
                        "observation": _compact_trajectory_observation_for_answerer(st.get("observation")),
                    }
                )
        out.append(
            {
                "query_id": str(item.get("query_id") or "").strip(),
                "query": str(item.get("query") or "").strip(),
                "result_summary": str(item.get("result_summary") or "").strip(),
                "steps": compact_steps,
            }
        )
    return out


def _final_text_from_memory(*, working_memory: list[dict], thought: str, fallback: str) -> str:
    if isinstance(working_memory, list) and working_memory:
        latest = working_memory[-1]
        if isinstance(latest, dict):
            points = latest.get("key_points")
            if isinstance(points, list) and points:
                first = points[0]
                if isinstance(first, dict):
                    text = str(first.get("text") or first.get("summary") or "").strip()
                else:
                    text = str(first or "").strip()
                if text:
                    return _trim_text(text, 1000)
    if str(thought or "").strip():
        return _trim_text(str(thought or "").strip(), 1000)
    return fallback


def _is_self_intro_query(text: str) -> bool:
    q = re.sub(r"\s+", "", str(text or "").strip().lower())
    if not q:
        return False
    patterns = (
        "你是谁",
        "你是干什么的",
        "你能做什么",
        "你可以做什么",
        "你会什么",
        "介绍一下你自己",
        "自我介绍",
        "你的功能",
        "你的能力",
        "你能帮我做什么",
    )
    return any(p in q for p in patterns)


SELF_INTRO_FIXED_ANSWER = (
    "你好，我是 RD-Claw Agent Demo！当前主要有三方面能力：\n"
    "1、自主通用问答：可直接回答通用知识问题。\n"
    "例如：\"解释条件随机场解码\"、\"苹果是当季水果吗\"。\n"
    "2、企业知识检索问答：可基于 ONES 发版文档、Adela 平台部署模型信息、评测结论和模型汇总表格进行回答。\n"
    "例如：\"safety_rope v0.2.1 新增了哪些标签？\"、\"某模型在 Adela 上的部署平台和推荐阈值是什么？\"。\n"
    "3、视觉能力：可执行图片生成、图片标注、评测报告生成。\n"
    "例如：\"生成 3 张河道巡检海报\"、\"标注图片中的横幅并输出结果\"、\"对检测结果生成评测报告\"。"
)


class AgentOrchestrator:
    """Main agent loop orchestrator."""

    def __init__(self, *, emit, execute_tool):
        self._emit = emit
        self._execute_tool = execute_tool

    def run(
        self,
        *,
        text: str,
        image_path: str,
        api_key: str,
        api_base: str,
        run_dir: Path,
        run_stamp: str,
        session: dict,
        forced_first_step: dict | None = None,
    ) -> dict:
        import memory_system as ms
        from tools.contracts import ToolCall, ToolExecutionContext

        emit = self._emit
        if _is_self_intro_query(text):
            emit(
                {
                    "type": "meta",
                    "flow": "self_intro",
                    "decision": {"action": "self_intro", "reason": ""},
                    "run_stamp": run_stamp,
                    "step_index": 1,
                }
            )
            for chunk in _split_text_for_stream(SELF_INTRO_FIXED_ANSWER):
                emit({"type": "final_answer", "text": chunk})
            return {"final_answer": SELF_INTRO_FIXED_ANSWER}

        final_text = ""
        effective_image_path = ""
        active_query_id = str(session.get("active_query_id") or "").strip()
        active_thread_id = str(session.get("active_thread_id") or "").strip()
        if image_path and Path(image_path).is_file():
            effective_image_path = str(Path(image_path).resolve())

        planner_context = ms.ContextBuilder.build_prompt_context(
            session, text=text, effective_image_path=effective_image_path
        )
        rag_round_count = 0
        last_rag_miss = False
        last_rag_query = str(text or "").strip()
        latest_rewritten_query = ""
        must_do_requestion = False
        rag_round_trace: list[dict] = []
        current_planner_ctx: dict = dict(planner_context)
        pending_forced_step = dict(forced_first_step) if isinstance(forced_first_step, dict) else None

        for step_index in range(1, AGENT_MAX_STEPS + 1):
            working_memory = ms.MemoryProjector.derive_working_memory(
                session, ms.SESSION_MEMORY_LIMIT
            )
            if pending_forced_step is not None:
                step = dict(pending_forced_step)
                pending_forced_step = None
            elif must_do_requestion and rag_round_count < 3:
                step = {
                    "thought": "上一轮 RAG 未命中，先执行 re_question 做小步改写，再继续检索。",
                    "decision_type": DECISION_TYPE_TOOL,
                    "action": TOOL_RE_QUESTION,
                    "action_input": {
                        "query": last_rag_query or str(text or "").strip(),
                        "rewrite_reason": "rag_miss",
                        "retrieval_round": max(1, rag_round_count),
                        "context_hint": str(text or "").strip(),
                        "finish_after_tool": False,
                    },
                    "final_answer": "",
                }
            elif latest_rewritten_query and last_rag_miss and rag_round_count < 3:
                step = {
                    "thought": "使用上一轮 re_question 的改写结果继续 rag_answer 检索。",
                    "decision_type": DECISION_TYPE_TOOL,
                    "action": TOOL_RAG_ANSWER,
                    "action_input": {
                        "query": latest_rewritten_query,
                        "finish_after_tool": False,
                    },
                    "final_answer": "",
                }
            else:
                planner_ctx = dict(planner_context)
                planner_ctx["rag_loop_state"] = {
                    "rag_round_count": rag_round_count,
                    "last_rag_miss": last_rag_miss,
                    "last_rag_query": last_rag_query,
                    "latest_rewritten_query": latest_rewritten_query,
                    "rag_max_rounds": 3,
                }
                step = choose_agent_step_with_fallback(
                    text,
                    effective_image_path or None,
                    planner_context=planner_ctx,
                    step_index=step_index,
                    max_steps=AGENT_MAX_STEPS,
                    debug_meta={
                        "session_id": str(session.get("session_id") or ""),
                        "run_stamp": run_stamp,
                        "run_dir": str(run_dir),
                    },
                )
                current_planner_ctx = planner_ctx

            decision_type = str(step.get("decision_type") or DECISION_TYPE_TOOL).strip().lower()
            if decision_type not in VALID_DECISION_TYPES:
                decision_type = DECISION_TYPE_TOOL

            if decision_type == DECISION_TYPE_CLARIFY:
                tid = str(session.get("active_thread_id") or active_thread_id).strip()
                ms.LedgerStore.append_event(
                    session,
                    event_type="PLAN_DECISION",
                    observation="clarify",
                    payload={
                        "action": "clarify",
                        "thought": str(step.get("thought") or "").strip(),
                        "clarification_question": str(step.get("clarification_question") or "").strip(),
                    },
                    thread_id=tid or None,
                )
                ms.LedgerStore.sync_ledger_cursor(session)
                return clarification_state.activate_pending_clarification(
                    session=session,
                    emit=emit,
                    thought=str(step.get("thought") or "").strip(),
                    question=str(step.get("clarification_question") or "").strip(),
                    source="planner_clarify",
                    step_index=step_index,
                    run_stamp=run_stamp,
                    query_id=active_query_id,
                    thread_id=tid,
                    normalize_action=normalize_agent_action,
                    task_state={
                        "candidate_tool": "",
                        "known_slots": {},
                        "missing_slots": [],
                        "tool_args": {},
                        "original_user_text": str(text or "").strip(),
                    },
                )

            if decision_type == DECISION_TYPE_END:
                end_reason = str(step.get("end_reason") or "").strip().lower()
                if end_reason == END_REASON_RECHECK_DONE:
                    return {"final_answer": ""}
                if end_reason == END_REASON_MEMORY_HIT:
                    query_trajectories = (
                        current_planner_ctx.get("query_trajectories")
                        if isinstance(current_planner_ctx.get("query_trajectories"), list)
                        else []
                    )
                    compact_query_trajectories = _compact_query_trajectories_for_answerer(
                        query_trajectories
                    )
                    answerer_input = {
                        "user_query": str(text or "").strip(),
                        "evidence": {
                            "retrieved_chunks": [],
                            "query_trajectories": compact_query_trajectories,
                        },
                    }
                    streamed_final_parts: list[str] = []
                    final_text = generate_final_answer_with_fallback(
                        answerer_input=answerer_input,
                        mode="memoryquery_trajectories",
                        debug_meta={
                            "session_id": str(session.get("session_id") or ""),
                            "run_stamp": run_stamp,
                            "run_dir": str(run_dir),
                            "step_index": step_index,
                            "stage": "answerer_memory_end_request",
                            "stage_response": "answerer_memory_end_response",
                        },
                        emit_chunk=lambda piece: (
                            streamed_final_parts.append(piece),
                            emit({"type": "final_answer", "text": piece}),
                        ),
                    )
                    if not final_text:
                        final_text = "已完成分析。"
                    if not streamed_final_parts:
                        for chunk in _split_text_for_stream(final_text):
                            emit({"type": "final_answer", "text": chunk})
                    return {"final_answer": final_text}

                final_text = str(step.get("final_answer") or "").strip() or _final_text_from_memory(
                    working_memory=working_memory,
                    thought=str(step.get("thought") or ""),
                    fallback="任务已结束。",
                )
                if final_text:
                    emit({"type": "final_answer", "text": final_text})
                return {"final_answer": final_text}

            if normalize_agent_action(str(step.get("action") or "").strip()) == TOOL_RAG_ANSWER:
                action_input = step.get("action_input")
                if not isinstance(action_input, dict):
                    action_input = {}
                if not str(action_input.get("query") or "").strip() and latest_rewritten_query:
                    action_input["query"] = latest_rewritten_query
                step["action_input"] = action_input

            precondition_clarify = clarification_state.build_tool_precondition_clarification(
                action=str(step.get("action") or "").strip(),
                action_input=step.get("action_input") if isinstance(step.get("action_input"), dict) else {},
                user_text=text,
                image_path=effective_image_path,
                normalize_action=normalize_agent_action,
                qwen_detection_action=TOOL_QWEN_DETECTION,
                rex_detection_action=TOOL_REXOMNI_DETECTION,
                pipeline_eval_action=TOOL_PIPELINE_EVAL,
                flux_action=TOOL_FLUX_IMAGE_GENERATION,
                adela_cli_action=TOOL_ADELA_CLI_EVAL,
            )
            if precondition_clarify is not None:
                tid = str(session.get("active_thread_id") or active_thread_id).strip()
                ms.LedgerStore.append_event(
                    session,
                    event_type="PLAN_DECISION",
                    observation="clarify",
                    payload={
                        "action": "clarify",
                        "thought": str(step.get("thought") or "").strip(),
                        "candidate_tool": normalize_agent_action(str(step.get("action") or "").strip()),
                        "clarification_question": str(precondition_clarify.get("clarification_question") or "").strip(),
                        "missing_slots": (
                            ((precondition_clarify.get("task_state") or {}).get("missing_slots"))
                            if isinstance(precondition_clarify.get("task_state"), dict)
                            else []
                        ),
                    },
                    thread_id=tid or None,
                )
                ms.LedgerStore.sync_ledger_cursor(session)
                return clarification_state.activate_pending_clarification(
                    session=session,
                    emit=emit,
                    thought=str(step.get("thought") or "").strip(),
                    question=str(precondition_clarify.get("clarification_question") or "").strip(),
                    source="tool_precondition",
                    step_index=step_index,
                    run_stamp=run_stamp,
                    query_id=active_query_id,
                    thread_id=tid,
                    normalize_action=normalize_agent_action,
                    task_state=precondition_clarify.get("task_state") if isinstance(precondition_clarify.get("task_state"), dict) else {},
                )

            tool_call = ToolCall.from_step(step)
            action = normalize_agent_action(tool_call.action)
            thought = tool_call.thought
            planner_summary = tool_call.final_answer

            if action == TOOL_ANSWERER:
                mode = str((tool_call.action_input or {}).get("mode") or "direct").strip()
                judger_chunks: list[dict] = []
                emit(
                    {
                        "type": "meta",
                        "flow": "direct_answer",
                        "decision": {
                            "action": TOOL_ANSWERER,
                            "reason": (
                                "RAG 三轮未命中，正在调用 LLM 进行通用回答..."
                                if mode == "rag_evidence"
                                else "正在调用 LLM 进行回答..."
                            ),
                            "direct_reply": "",
                        },
                        "run_stamp": run_stamp,
                        "step_index": step_index,
                    }
                )
                if mode == "rag_evidence":
                    all_chunks: list[dict] = []
                    for item in rag_round_trace:
                        if not isinstance(item, dict):
                            continue
                        chunks = item.get("retrieved_chunks")
                        if isinstance(chunks, list):
                            all_chunks.extend(chunks)
                    judger_chunks = list(all_chunks)
                    answerer_input = {
                        "user_query": str(text or "").strip(),
                        "evidence": {
                            "retrieved_chunks": all_chunks,
                        },
                    }
                else:
                    rag_obs_list = collect_rag_observations_from_session(session, limit=1)
                    latest_rag_obs = rag_obs_list[-1] if rag_obs_list else {}
                    latest_rag_obs = latest_rag_obs if isinstance(latest_rag_obs, dict) else {}
                    answerer_input = {
                        "user_query": str(text or "").strip(),
                        "evidence": {
                            "retrieved_chunks": (
                                latest_rag_obs.get("retrieved_chunks")
                                if isinstance(latest_rag_obs.get("retrieved_chunks"), list)
                                else []
                            ),
                        },
                    }
                    judger_chunks = list(answerer_input["evidence"]["retrieved_chunks"])
                streamed_final_parts: list[str] = []
                final_text = generate_final_answer_with_fallback(
                    answerer_input=answerer_input,
                    mode=mode,
                    debug_meta={
                        "session_id": str(session.get("session_id") or ""),
                        "run_stamp": run_stamp,
                        "run_dir": str(run_dir),
                        "step_index": step_index,
                        "stage": "answerer_tool_request",
                        "stage_response": "answerer_tool_response",
                    },
                    emit_chunk=lambda piece: (
                        streamed_final_parts.append(piece),
                        emit({"type": "final_answer", "text": piece}),
                    ),
                )
                if not final_text:
                    final_text = "已完成分析。"
                if mode == "rag_evidence" and RAG_RESOLUTION_JUDGE_ENABLED:
                    judged = judge_answer_resolution_with_fallback(
                        user_query=str(text or "").strip(),
                        candidate_answer=final_text,
                        retrieved_chunks=judger_chunks,
                    )
                    if not bool(judged.get("resolved")):
                        return clarification_state.activate_pending_clarification(
                            session=session,
                            emit=emit,
                            thought=str(judged.get("reason") or thought).strip() or "当前回答仍未解决用户问题，需要进一步澄清。",
                            question=str(judged.get("clarification_question") or "").strip(),
                            source="rag_judger",
                            step_index=step_index,
                            run_stamp=run_stamp,
                            query_id=active_query_id,
                            thread_id=str(session.get("active_thread_id") or active_thread_id).strip(),
                            normalize_action=normalize_agent_action,
                            task_state={
                                "candidate_tool": "",
                                "known_slots": {},
                                "missing_slots": [],
                                "tool_args": {},
                                "original_user_text": str(text or "").strip(),
                            },
                        )
                if not streamed_final_parts:
                    for chunk in _split_text_for_stream(final_text):
                        emit({"type": "final_answer", "text": chunk})
                emit(
                    {
                        "type": "meta",
                        "flow": "direct_answer",
                        "decision": {
                            "action": ACTION_FINAL_ANSWER,
                            "reason": thought,
                            "direct_reply": final_text,
                        },
                        "run_stamp": run_stamp,
                        "step_index": step_index,
                    }
                )
                return {"final_answer": final_text}

            if action == ACTION_FINAL_ANSWER:
                tid = str(session.get("active_thread_id") or "").strip()
                ms.LedgerStore.append_event(
                    session,
                    event_type="PLAN_DECISION",
                    observation=action,
                    payload={
                        "action": action,
                        "action_input": tool_call.action_input,
                        "thought": thought,
                    },
                    thread_id=tid or None,
                )
                fa_ev = ms.LedgerStore.append_event(
                    session,
                    event_type="ASSISTANT_OUTPUT",
                    observation="final_answer",
                    payload={"planner_summary": planner_summary, "thought": thought},
                    thread_id=tid or None,
                )
                ms.QueryTrajectoryStore.append_step(
                    session,
                    action="final_answer",
                    observation_event_id=str(fa_ev.get("event_id") or ""),
                )
                ms.LedgerStore.sync_ledger_cursor(session)

                final_text = planner_summary or _final_text_from_memory(
                    working_memory=working_memory,
                    thought=thought,
                    fallback="已完成分析。",
                )
                if not final_text:
                    final_text = "已完成分析，但未生成明确答复。"
                emit(
                    {
                        "type": "meta",
                        "flow": "direct_answer",
                        "decision": {
                            "action": ACTION_FINAL_ANSWER,
                            "reason": thought,
                            "direct_reply": final_text,
                        },
                        "run_stamp": run_stamp,
                        "step_index": step_index,
                    }
                )
                emit({"type": "final_answer", "text": final_text})
                return {"final_answer": final_text}

            flow = flow_for_action(to_legacy_action(action))
            emit(
                {
                    "type": "meta",
                    "flow": flow,
                    "decision": {
                        "action": to_legacy_action(action),
                        "reason": thought,
                        "direct_reply": "",
                    },
                    "run_stamp": run_stamp,
                    "step_index": step_index,
                }
            )
            exec_ctx = ToolExecutionContext(
                text=text,
                image_path=effective_image_path,
                api_key=api_key,
                api_base=api_base,
                run_dir=run_dir,
                run_stamp=run_stamp,
                image_paths=[effective_image_path] if effective_image_path else [],
                session_id=str(session.get("session_id") or ""),
                session=session,
            )
            tool_result = self._execute_tool(tool_call=tool_call, ctx=exec_ctx)
            if bool(getattr(tool_result, "requires_clarification", False)):
                observation = tool_result.observation if isinstance(tool_result.observation, dict) else {}
                return clarification_state.activate_pending_clarification(
                    session=session,
                    emit=emit,
                    thought=thought,
                    question=str(observation.get("clarification_question") or "").strip(),
                    source="tool_precondition",
                    step_index=step_index,
                    run_stamp=run_stamp,
                    query_id=active_query_id,
                    thread_id=str(session.get("active_thread_id") or active_thread_id).strip(),
                    normalize_action=normalize_agent_action,
                    task_state=observation.get("task_state") if isinstance(observation.get("task_state"), dict) else {},
                )
            observation = tool_result.observation
            ms.MemoryProjector.persist_step(
                session,
                step_index=step_index,
                text=text,
                step={
                    "action": tool_call.action,
                    "action_input": tool_call.action_input,
                },
                thought=thought,
                observation=observation,
                run_dir=run_dir,
            )
            planner_context = ms.ContextBuilder.build_prompt_context(
                session, text=text, effective_image_path=effective_image_path
            )

            if action == TOOL_RE_QUESTION:
                latest_rewritten_query = str((observation or {}).get("rewritten_query") or "").strip() or latest_rewritten_query
                must_do_requestion = False
                # re_question 执行后同轮立即进入 rag 外层状态机，不再等待下一轮 planner。
                rag_followup_query = (
                    latest_rewritten_query
                    or last_rag_query
                    or str(text or "").strip()
                )
                rag_followup_thought = "re_question 已完成，立即使用改写后的 query 继续 rag_answer 检索。"
                rag_followup_step = {
                    "thought": rag_followup_thought,
                    "decision_type": DECISION_TYPE_TOOL,
                    "action": TOOL_RAG_ANSWER,
                    "action_input": {
                        "query": rag_followup_query,
                        "finish_after_tool": False,
                    },
                    "final_answer": "",
                }
                rag_tool_call = ToolCall.from_step(rag_followup_step)
                emit(
                    {
                        "type": "meta",
                        "flow": flow_for_action(to_legacy_action(TOOL_RAG_ANSWER)),
                        "decision": {
                            "action": to_legacy_action(TOOL_RAG_ANSWER),
                            "reason": rag_followup_thought,
                            "direct_reply": "",
                        },
                        "run_stamp": run_stamp,
                        "step_index": step_index,
                    }
                )
                rag_tool_result = self._execute_tool(tool_call=rag_tool_call, ctx=exec_ctx)
                rag_observation = rag_tool_result.observation
                ms.MemoryProjector.persist_step(
                    session,
                    step_index=step_index,
                    text=text,
                    step={
                        "action": rag_tool_call.action,
                        "action_input": rag_tool_call.action_input,
                    },
                    thought=rag_followup_thought,
                    observation=rag_observation,
                    run_dir=run_dir,
                )
                planner_context = ms.ContextBuilder.build_prompt_context(
                    session, text=text, effective_image_path=effective_image_path
                )
                tool_call = rag_tool_call
                action = TOOL_RAG_ANSWER
                thought = rag_followup_thought
                observation = rag_observation

            if action == TOOL_RAG_ANSWER:
                rag_round_count += 1
                last_rag_query = str((tool_call.action_input or {}).get("query") or text or "").strip()
                observation_chunks = (
                    (observation or {}).get("retrieved_chunks")
                    if isinstance((observation or {}).get("retrieved_chunks"), list)
                    else []
                )
                if not observation_chunks:
                    observation_chunks = _load_rag_retrieved_chunks_from_run_dir(run_dir)
                rag_round_trace.append(
                    {
                        "round": rag_round_count,
                        "query": last_rag_query,
                        "observation": _observation_without_answer(observation),
                        "references": (
                            (observation or {}).get("references")
                            if isinstance((observation or {}).get("references"), list)
                            else []
                        ),
                        "retrieved_chunks": observation_chunks,
                    }
                )
                last_rag_miss = is_rag_miss(observation)
                if last_rag_miss and rag_round_count < 3:
                    must_do_requestion = True
                    continue

                if not last_rag_miss:
                    final_text = (
                        str((observation or {}).get("answer") or "").strip()
                        or str((observation or {}).get("summary") or "").strip()
                        or "任务已执行完成。"
                    )
                    if RAG_RESOLUTION_JUDGE_ENABLED:
                        judged = judge_answer_resolution_with_fallback(
                            user_query=str(text or "").strip(),
                            candidate_answer=final_text,
                            retrieved_chunks=observation_chunks,
                        )
                        if not bool(judged.get("resolved")):
                            return clarification_state.activate_pending_clarification(
                                session=session,
                                emit=emit,
                                thought=str(judged.get("reason") or thought).strip() or "当前回答仍未解决用户问题，需要进一步澄清。",
                                question=str(judged.get("clarification_question") or "").strip(),
                                source="rag_judger",
                                step_index=step_index,
                                run_stamp=run_stamp,
                                query_id=active_query_id,
                                thread_id=str(session.get("active_thread_id") or active_thread_id).strip(),
                                normalize_action=normalize_agent_action,
                                task_state={
                                    "candidate_tool": "",
                                    "known_slots": {},
                                    "missing_slots": [],
                                    "tool_args": {},
                                    "original_user_text": str(text or "").strip(),
                                },
                            )
                    emit({"type": "final_answer", "text": final_text})
                    return {"final_answer": final_text}

                # 第 3 轮仍未命中：按固定闭环，下一步强制进入 answerer。
                offer_text = (
                    "知识库没有找到可直接回答的模型信息。是否基于相似模型和已有能力生成迁移顾问报告？"
                )
                tid = str(session.get("active_thread_id") or active_thread_id).strip()
                session["pending_migration_advisor"] = {
                    "status": "pending",
                    "query_id": active_query_id,
                    "thread_id": tid,
                    "original_user_text": str(text or "").strip(),
                    "rag_round_trace": rag_round_trace,
                }
                emit(
                    {
                        "type": "meta",
                        "flow": "migration_advisor_offer",
                        "decision": {
                            "action": "migration_advisor_offer",
                            "reason": "RAG 已完成 3 轮且均未充分命中，询问用户是否进入迁移顾问分析。",
                            "direct_reply": "",
                        },
                        "run_stamp": run_stamp,
                        "step_index": step_index,
                    }
                )
                emit(
                    {
                        "type": "migration_advisor_offer",
                        "text": offer_text,
                        "query_id": active_query_id,
                        "options": [
                            {"id": "start", "label": "生成迁移顾问报告"},
                            {"id": "fallback_answer", "label": "直接回答"},
                            {"id": "cancel", "label": "取消"},
                        ],
                    }
                )
                return {
                    "final_answer": offer_text,
                    "assistant_text": offer_text,
                    "assistant_event_type": "migration_advisor_offer",
                    "query_completed": False,
                }

            if action == TOOL_PIPELINE_EVAL:
                # 评测报告生成后直接结束，不再进入 LLM 回答流。
                return {"final_answer": ""}

            if action == TOOL_MIGRATION_ADVISOR:
                final_text = (
                    str((observation or {}).get("summary") or "").strip()
                    or "迁移顾问报告已生成。"
                )
                return {"final_answer": final_text}

            if action == TOOL_ADELA_CLI_EVAL:
                final_text = (
                    str((observation or {}).get("summary") or "").strip() or "Adela 任务已执行完成。"
                )
                emit({"type": "final_answer", "text": final_text})
                return {"final_answer": final_text}

            if bool(tool_call.action_input.get("finish_after_tool", False)):
                final_text = (
                    str((observation or {}).get("summary") or "").strip() or "任务已执行完成。"
                )
                emit({"type": "final_answer", "text": final_text})
                return {"final_answer": final_text}

        working_memory = ms.MemoryProjector.derive_working_memory(
            session, ms.SESSION_MEMORY_LIMIT
        )
        final_text = _final_text_from_memory(
            working_memory=working_memory,
            thought="已达到最大循环轮次。",
            fallback="已达到最大循环轮次，但仍未收集到足够信息。",
        )
        emit({"type": "final_answer", "text": final_text})
        return {"final_answer": final_text}
