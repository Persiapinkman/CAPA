from __future__ import annotations

"""
Unified tool execution gateway for the demo.

主要功能：
- 接收标准化 `ToolCall` 与 `ToolExecutionContext`，分发到具体工具执行函数。
- 在执行前进行动作归一化与关键前置条件校验（如图片存在、label 非空）。
- 将执行结果统一收敛为 `ToolResult`，减少跨模块 dict 字段拼写错误。

主要模块：
- `ToolExecutor.__init__`：注入各工具运行回调。
- `ToolExecutor.execute`：统一分发、容错与结果封装。
"""

import re
from pathlib import Path
from typing import Callable

import agent
import clarification as clarification_state
from tools import registry as tool_registry
from tools.contracts import ToolCall, ToolExecutionContext, ToolResult


class ToolExecutor:
    """Unified tool execution gateway."""

    def __init__(
        self,
        *,
        emit: Callable[[dict], None],
        failure_observation: Callable[[str, str], dict],
        run_rag_streaming: Callable[..., dict | None],
        run_flux_only_streaming: Callable[..., dict | None],
        run_detection_only_streaming: Callable[..., dict | None],
        run_rex_detection_only_streaming: Callable[..., dict | None],
        run_pipeline_streaming: Callable[..., dict | None],
        run_migration_advisor_streaming: Callable[..., dict | None],
        run_adela_cli_streaming: Callable[..., dict | None],
        resolve_adela_model_reference: Callable[..., dict],
    ) -> None:
        self._emit = emit
        self._failure_observation = failure_observation
        self._run_rag_streaming = run_rag_streaming
        self._run_flux_only_streaming = run_flux_only_streaming
        self._run_detection_only_streaming = run_detection_only_streaming
        self._run_rex_detection_only_streaming = run_rex_detection_only_streaming
        self._run_pipeline_streaming = run_pipeline_streaming
        self._run_migration_advisor_streaming = run_migration_advisor_streaming
        self._run_adela_cli_streaming = run_adela_cli_streaming
        self._resolve_adela_model_reference = resolve_adela_model_reference

    @staticmethod
    def _parse_adela_eval_type(raw_value) -> int | None:
        text = str(raw_value or "").strip()
        if not text:
            return None
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
        if perf_hit and acc_hit:
            return None
        if text in {"0", "1"}:
            try:
                parsed = int(text)
            except (TypeError, ValueError):
                parsed = -1
            if parsed in (0, 1):
                return parsed
        if low in {
            "0",
            "精度",
            "精度评测",
            "准确率",
            "accuracy",
            "precision",
            "normal_precision",
            "normal precision",
        }:
            return 0
        if low in {
            "1",
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
        if perf_hit:
            return 1
        if acc_hit:
            return 0
        try:
            parsed = int(raw_value)
        except (TypeError, ValueError):
            return None
        if parsed in (0, 1):
            return parsed
        return None

    @staticmethod
    def _clean_model_name(raw_value: str) -> str:
        return re.sub(r"\s+", " ", str(raw_value or "").strip())

    @staticmethod
    def _looks_like_explicit_model_id(rawmodel_id_raw, model_name: str) -> tuple[bool, int]:
        try:
            parsed = int(rawmodel_id_raw)
        except (TypeError, ValueError):
            parsed = 0
        if parsed > 0:
            return True, parsed
        if model_name.isdigit():
            return True, int(model_name)
        return False, 0

    def execute(self, *, tool_call: ToolCall, ctx: ToolExecutionContext) -> ToolResult:
        emit = self._emit
        action = agent.normalize_agent_action(str(tool_call.action or "").strip())
        action_input = tool_call.action_input or {}
        branch = tool_registry.executor_branch_for_action(action)

        if branch == "rag":
            query = str(action_input.get("query") or ctx.text or "").strip()
            obs = self._run_rag_streaming(query, ctx.run_dir, ctx.run_stamp, emit_done=False)
            data = obs or self._failure_observation(action, "RAG 工具未返回有效结果")
            return ToolResult(action=action, observation=data, ok=bool(obs))

        if branch == "re_question":
            source_query = str(action_input.get("query") or ctx.text or "").strip()
            rewrite_reason = str(action_input.get("rewrite_reason") or "rag_miss").strip()
            context_hint = str(action_input.get("context_hint") or "").strip()
            try:
                retrieval_round = int(action_input.get("retrieval_round", 1))
            except (TypeError, ValueError):
                retrieval_round = 1
            retrieval_round = max(1, min(3, retrieval_round))
            rewritten_query = agent.rewrite_query_with_fallback(
                query=source_query,
                rewrite_reason=rewrite_reason,
                context_hint=context_hint,
                retrieval_round=retrieval_round,
            )
            emit(
                {
                    "type": "re_question",
                    "source_query": source_query,
                    "rewritten_query": rewritten_query,
                    "retrieval_round": retrieval_round,
                }
            )
            data = {
                "action": action,
                "summary": f"query 已改写（第{retrieval_round}轮）",
                "source_query": source_query,
                "rewritten_query": rewritten_query,
                "rewrite_reason": rewrite_reason,
                "retrieval_round": retrieval_round,
                "success": bool(rewritten_query),
            }
            return ToolResult(action=action, observation=data, ok=bool(rewritten_query))

        if branch == "flux":
            task_text = str(action_input.get("task_text") or ctx.text or "").strip()
            source_required = bool(action_input.get("source_image_required"))
            try:
                num_images = int(action_input.get("num_images", 1))
            except (TypeError, ValueError):
                num_images = 1
            num_images = max(1, min(5, num_images))
            if source_required and (not ctx.image_path or not Path(ctx.image_path).exists()):
                msg = "图像生成工具要求参考图，但当前没有可用图片"
                emit({"type": "error", "message": msg})
                data = self._failure_observation(action, msg)
                return ToolResult(action=action, observation=data, ok=False, error_message=msg)
            obs = self._run_flux_only_streaming(
                task_text,
                ctx.image_path if ctx.image_path and Path(ctx.image_path).exists() else "",
                "",
                num_images,
                ctx.api_key,
                ctx.api_base,
                ctx.run_dir,
                ctx.run_stamp,
                emit_done=False,
            )
            data = obs or self._failure_observation(action, "Flux 图像生成未返回有效结果")
            return ToolResult(action=action, observation=data, ok=bool(obs))

        if branch == "qwen_detection":
            label = str(action_input.get("label") or "").strip()
            if not ctx.image_path or not Path(ctx.image_path).exists():
                msg = "Qwen 检测工具需要上传待检测图片"
                emit({"type": "error", "message": msg})
                data = self._failure_observation(action, msg)
                return ToolResult(action=action, observation=data, ok=False, error_message=msg)
            if not label:
                msg = "Qwen 检测工具缺少 label"
                emit({"type": "error", "message": msg})
                data = self._failure_observation(action, msg)
                return ToolResult(action=action, observation=data, ok=False, error_message=msg)
            obs = self._run_detection_only_streaming(
                ctx.image_path, label, ctx.run_dir, ctx.run_stamp, emit_done=False
            )
            data = obs or self._failure_observation(action, "Qwen 检测未返回有效结果")
            return ToolResult(action=action, observation=data, ok=bool(obs))

        if branch == "rexomni_detection":
            label = str(action_input.get("label") or "").strip()
            if not ctx.image_path or not Path(ctx.image_path).exists():
                msg = "Rex-Omni 检测工具需要上传待检测图片"
                emit({"type": "error", "message": msg})
                data = self._failure_observation(action, msg)
                return ToolResult(action=action, observation=data, ok=False, error_message=msg)
            if not label:
                msg = "Rex-Omni 检测工具缺少 label"
                emit({"type": "error", "message": msg})
                data = self._failure_observation(action, msg)
                return ToolResult(action=action, observation=data, ok=False, error_message=msg)
            obs = self._run_rex_detection_only_streaming(
                ctx.image_path, label, ctx.run_dir, ctx.run_stamp, emit_done=False
            )
            data = obs or self._failure_observation(action, "Rex-Omni 检测未返回有效结果")
            return ToolResult(action=action, observation=data, ok=bool(obs))

        if branch == "pipeline":
            task_text = str(action_input.get("task_text") or ctx.text or "").strip()
            if not ctx.image_path or not Path(ctx.image_path).exists():
                msg = "评测流水线需要上传参考图片"
                emit({"type": "error", "message": msg})
                data = self._failure_observation(action, msg)
                return ToolResult(action=action, observation=data, ok=False, error_message=msg)
            obs = self._run_pipeline_streaming(
                task_text, ctx.image_path, ctx.api_key, ctx.api_base, ctx.run_dir, ctx.run_stamp, emit_done=False
            )
            data = obs or self._failure_observation(action, "评测流水线未返回有效结果")
            return ToolResult(action=action, observation=data, ok=bool(obs))

        if branch == "migration_advisor":
            user_query = str(action_input.get("user_query") or ctx.text or "").strip()
            use_image = bool(action_input.get("use_image"))
            if use_image and not (ctx.image_path and Path(ctx.image_path).exists()):
                msg = "迁移顾问需要使用样例图，但当前没有可用图片"
                emit({"type": "error", "message": msg})
                data = self._failure_observation(action, msg)
                return ToolResult(action=action, observation=data, ok=False, error_message=msg)
            image_path = ctx.image_path if use_image else ""
            image_paths = ctx.image_paths if use_image else []
            obs = self._run_migration_advisor_streaming(
                text=user_query,
                rag_trace=[],
                run_dir=ctx.run_dir,
                run_stamp=ctx.run_stamp,
                session_id=ctx.session_id,
                image_path=image_path,
                image_paths=image_paths,
                session=ctx.session,
                emit_done=False,
            )
            data = obs or self._failure_observation(action, "迁移顾问未返回有效结果")
            return ToolResult(action=action, observation=data, ok=bool(obs))

        if branch == "adela_cli":
            retrieval_note = ""
            rawmodel_id_raw = action_input.get("rawmodel_id")
            model_name = self._clean_model_name(action_input.get("model_name"))
            platform = str(action_input.get("platform") or "").strip()
            eval_type = self._parse_adela_eval_type(action_input.get("eval_type"))
            if eval_type is None:
                eval_type = clarification_state.adela_eval_type_from_text(str(ctx.text or ""))
            has_explicit_id, rawmodel_id = self._looks_like_explicit_model_id(rawmodel_id_raw, model_name)
            if has_explicit_id and model_name and not model_name.isdigit():
                action_input["rawmodel_id"] = rawmodel_id
                action_input["model_name"] = model_name
            elif has_explicit_id:
                resolved = self._resolve_adela_model_reference(
                    rawmodel_id=rawmodel_id,
                    model_name="",
                    run_dir=ctx.run_dir,
                    run_stamp=ctx.run_stamp,
                )
                if str(resolved.get("status") or "").strip() != "resolved":
                    msg = str(resolved.get("message") or "").strip() or "当前 adela 平台上没有这个模型，如有需要，请到 monolith 平台上训练模型，并上传到 adela 平台。"
                    emit({"type": "error", "message": msg})
                    data = self._failure_observation(action, msg)
                    return ToolResult(action=action, observation=data, ok=False, error_message=msg)
                rawmodel_id = int(resolved.get("rawmodel_id") or rawmodel_id)
                matched_name = self._clean_model_name(resolved.get("matched_name"))
                if matched_name:
                    model_name = matched_name
            elif model_name:
                resolved = self._resolve_adela_model_reference(
                    rawmodel_id=None,
                    model_name=model_name,
                    run_dir=ctx.run_dir,
                    run_stamp=ctx.run_stamp,
                )
                status = str(resolved.get("status") or "").strip()
                if status == "resolved":
                    rawmodel_id = int(resolved.get("rawmodel_id") or 0)
                    matched_name = self._clean_model_name(resolved.get("matched_name"))
                    if matched_name:
                        model_name = matched_name
                    action_input["rawmodel_id"] = rawmodel_id
                    action_input["model_name"] = model_name
                    retrieval_note = f"已通过 RAG 解析模型 ID：{model_name} -> {rawmodel_id}"
                elif status == "ambiguous":
                    candidates = resolved.get("candidate_model_names") if isinstance(resolved.get("candidate_model_names"), list) else []
                    question = (
                        f"没有找到与“{model_name}”完全一致的模型名称。请确认你要的是下面哪个模型：\n"
                        + "\n".join(f"{idx + 1}. {name}" for idx, name in enumerate(candidates))
                        + "\n请直接回复完整模型名称。"
                    ) if candidates else f"没有找到与“{model_name}”完全一致的模型名称，请提供更准确的模型名称。"
                    task_state = clarification_state.build_adela_clarification_task_state(
                        task_state={
                            "candidate_tool": action,
                            "known_slots": {
                                "model_name": model_name,
                                "platform": platform,
                                "eval_type": str(eval_type) if eval_type is not None else "",
                            },
                            "missing_slots": ["model_name"],
                            "tool_args": {
                                "model_name": model_name,
                                "platform": platform,
                                "eval_type": eval_type,
                                "finish_after_tool": bool(action_input.get("finish_after_tool", False)),
                            },
                            "original_user_text": ctx.text,
                        },
                        known_slots={
                            "model_name": model_name,
                            "platform": platform,
                            "eval_type": str(eval_type) if eval_type is not None else "",
                        },
                        missing_slots=["model_name"],
                        tool_args={
                            "model_name": model_name,
                            "platform": platform,
                            "eval_type": eval_type,
                            "finish_after_tool": bool(action_input.get("finish_after_tool", False)),
                        },
                        model_resolution_status="ambiguous",
                        candidate_model_names=candidates,
                    )
                    return ToolResult(
                        action=action,
                        observation={
                            "clarification_question": question,
                            "task_state": task_state,
                        },
                        ok=False,
                        requires_clarification=True,
                    )
                else:
                    msg = str(resolved.get("message") or "").strip() or "当前 adela 平台上没有这个模型，如有需要，请到 monolith 平台上训练模型，并上传到 adela 平台。"
                    emit({"type": "error", "message": msg})
                    data = self._failure_observation(action, msg)
                    return ToolResult(action=action, observation=data, ok=False, error_message=msg)
            else:
                rawmodel_id = -1
            if rawmodel_id <= 0:
                msg = "Adela 评测工具缺少有效的 rawmodel_id。"
                emit({"type": "error", "message": msg})
                data = self._failure_observation(action, msg)
                return ToolResult(action=action, observation=data, ok=False, error_message=msg)
            action_input["rawmodel_id"] = rawmodel_id
            if model_name:
                action_input["model_name"] = model_name
            if not platform:
                msg = "Adela 评测工具缺少目标平台 platform"
                emit({"type": "error", "message": msg})
                data = self._failure_observation(action, msg)
                return ToolResult(action=action, observation=data, ok=False, error_message=msg)
            model_url = (
                "http://scg-adela.sensetime.com/dashboard#/mainpage/project/3/release"
                f"?rid={rawmodel_id}"
            )
            emit(
                {
                    "type": "adela_event",
                    "event": "model_retrieval_result",
                    "message": str(retrieval_note or "").strip()[:500],
                    "status": "SUCCESS",
                    "rawmodel_id": rawmodel_id,
                    "matched_name": model_name,
                    "platform": platform,
                    "model_url": model_url,
                    "deployment_id": "",
                    "benchmark_id": "",
                    "result_preview": "",
                }
            )
            user_q = str(ctx.text or "").strip()
            eval_seq = clarification_state.adela_eval_sequence_from_text(user_q)
            if eval_type is None and len(eval_seq) == 1:
                eval_type = eval_seq[0]
            if eval_type is None:
                msg = "Adela 评测工具缺少有效的 eval_type（0=精度，1=性能）"
                emit({"type": "error", "message": msg})
                data = self._failure_observation(action, msg)
                return ToolResult(action=action, observation=data, ok=False, error_message=msg)
            obs = self._run_adela_cli_streaming(
                rawmodel_id, platform, eval_type, ctx.run_dir, ctx.run_stamp, emit_done=False
            )
            data = obs or self._failure_observation(action, "Adela CLI 未返回有效结果")
            return ToolResult(action=action, observation=data, ok=bool(obs))

        msg = f"未知 Agent 工具 action：{action}"
        emit({"type": "error", "message": msg})
        data = self._failure_observation(action or "unknown", msg)
        return ToolResult(
            action=action or "unknown",
            observation=data,
            ok=False,
            error_message=msg,
        )
