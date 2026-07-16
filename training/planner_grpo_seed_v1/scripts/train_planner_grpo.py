#!/usr/bin/env python3
"""Train the CAPA Planner with TRL GRPO on focused step-level cases."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


ROOT = Path(__file__).resolve().parents[3]
DEMO_DIR = ROOT / "demo"
for path in (ROOT, DEMO_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import agent  # noqa: E402
import memory_system as ms  # noqa: E402
from training.planner_grpo_seed_v1.scripts import reward_planner_grpo as rewardlib  # noqa: E402
from training.planner_grpo_seed_v1.scripts.run_planner_grpo_rollout import (  # noqa: E402
    make_session,
    mock_observation_for_step,
    persist_mock_step,
    resolve_image_path,
)
from util.path_resolver import resolve_model_name_or_path  # noqa: E402

torch.backends.cudnn.enabled = False

DEFAULT_CASES = ROOT / "training" / "planner_grpo_seed_v1" / "cases" / "planner_grpo_focused_4b_cases.jsonl"


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run step-level GRPO for the CAPA Planner.")
    parser.add_argument("--model-name-or-path", default="/raid/zkq/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "planner-grpo-qwen25-7b-focused-lora"))
    parser.add_argument("--max-prompt-length", type=int, default=3072)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", type=parse_bool, default=True)
    parser.add_argument("--bf16", type=parse_bool, default=False)
    parser.add_argument("--gradient-checkpointing", type=parse_bool, default=True)
    parser.add_argument(
        "--gradient-checkpointing-kwargs",
        default='{"use_reentrant": false}',
        help="JSON kwargs passed through TrainingArguments when supported.",
    )
    parser.add_argument("--trust-remote-code", type=parse_bool, default=True)
    parser.add_argument("--use-lora", type=parse_bool, default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--fsdp", default="")
    parser.add_argument(
        "--fsdp-config",
        default="",
        help="JSON object or path to JSON file for TrainingArguments.fsdp_config.",
    )
    parser.add_argument("--ddp-find-unused-parameters", type=parse_bool, default=False)
    parser.add_argument("--report-to", default="tensorboard")
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


def parse_json_arg(value: str) -> dict[str, Any] | None:
    text = str(value or "").strip()
    if not text:
        return None
    maybe_path = Path(text)
    if maybe_path.exists():
        text = maybe_path.read_text(encoding="utf-8")
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("JSON argument must be an object")
    return parsed


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: row must be object")
        rows.append(row)
    return rows


def expected_decision_to_planner_step(expected: dict[str, Any]) -> dict[str, Any]:
    decision_type = str(expected.get("decision_type") or "tool").strip()
    if decision_type == "clarify":
        return {
            "thought": "当前意图存在多种高概率解释，需要先向用户澄清。",
            "decision_type": "clarify",
            "clarification_question": "请明确你是要检测图片、生成图片，还是查询/评估方案？",
        }
    if decision_type == "end":
        required_args = expected.get("required_args") if isinstance(expected.get("required_args"), dict) else {}
        return {
            "thought": "已有上下文结果足以回答当前问题，无需重复调用工具。",
            "decision_type": "end",
            "end_reason": str(required_args.get("end_reason") or "memory_hit"),
            "final_answer": "",
        }
    action_input: dict[str, Any] = {}
    action_input.update(expected.get("required_args") if isinstance(expected.get("required_args"), dict) else {})
    for key, tokens in (expected.get("arg_contains") if isinstance(expected.get("arg_contains"), dict) else {}).items():
        if isinstance(tokens, list):
            action_input.setdefault(key, " ".join(str(token) for token in tokens))
        else:
            action_input.setdefault(key, str(tokens))
    return {
        "thought": "按训练样本期望执行该工具。",
        "decision_type": "tool",
        "action": str(expected.get("action") or "").strip(),
        "action_input": action_input,
        "final_answer": "",
    }


def expected_action_name(expected: dict[str, Any]) -> str:
    decision_type = str(expected.get("decision_type") or "tool").strip()
    if decision_type in {"clarify", "end"}:
        return decision_type
    return str(expected.get("action") or "").strip()


def build_prompt_for_step(case: dict[str, Any], step_index: int, run_root: Path) -> str:
    user_query = str(case.get("user_query") or "").strip()
    image_path = resolve_image_path(case)
    session = make_session(case)
    run_dir = run_root / str(case.get("case_id") or "unknown")
    run_dir.mkdir(parents=True, exist_ok=True)
    expected = case.get("expected_decisions") if isinstance(case.get("expected_decisions"), list) else []
    for prev_idx in range(1, step_index):
        prev_expected = expected[prev_idx - 1]
        prev_decision = expected_decision_to_planner_step(prev_expected)
        observation = mock_observation_for_step(case, prev_idx)
        persist_mock_step(
            session=session,
            run_dir=run_dir,
            step_index=prev_idx,
            user_query=user_query,
            decision=prev_decision,
            observation=observation,
        )
    planner_context = ms.ContextBuilder.build_prompt_context(
        session,
        text=user_query,
        effective_image_path=image_path,
    )
    system_prompt = agent.build_agent_system_prompt(max_steps=agent.AGENT_MAX_STEPS)
    user_prompt = agent.build_agent_user_prompt(
        user_query,
        image_path or None,
        planner_context=planner_context,
        step_index=step_index,
        max_steps=agent.AGENT_MAX_STEPS,
    )
    return (
        "<|system|>\n"
        f"{system_prompt}\n"
        "<|user|>\n"
        f"{user_prompt}\n"
        "<|assistant|>\n"
    )


def build_step_dataset(cases: list[dict[str, Any]]) -> Dataset:
    run_root = ROOT / "training" / "planner_grpo_seed_v1" / "reports" / "grpo_train_prompt_contexts"
    rows: list[dict[str, Any]] = []
    for case in cases:
        expected = case.get("expected_decisions") if isinstance(case.get("expected_decisions"), list) else []
        for idx, expected_step in enumerate(expected, start=1):
            if not isinstance(expected_step, dict):
                continue
            prev_action = ""
            if idx > 1 and isinstance(expected[idx - 2], dict):
                prev_action = str(expected[idx - 2].get("action") or "").strip()
            rows.append(
                {
                    "prompt": build_prompt_for_step(case, idx, run_root),
                    "case_id": str(case.get("case_id") or ""),
                    "category": str(case.get("category") or ""),
                    "step_index": idx,
                    "expected_step": json.dumps(expected_step, ensure_ascii=False),
                    "forbidden_actions": json.dumps(case.get("forbidden_actions") or [], ensure_ascii=False),
                    "reward_spec": json.dumps(case.get("reward_spec") or {}, ensure_ascii=False),
                    "previous_action": prev_action,
                    "entity_id": str(case.get("entity_id") or ""),
                    "group_id": str(case.get("group_id") or case.get("entity_id") or case.get("case_id") or ""),
                    "template_id": str(case.get("template_id") or ""),
                    "scenario_id": str(case.get("scenario_id") or case.get("category") or ""),
                    "full_expected_actions": json.dumps(
                        [
                            expected_action_name(step)
                            for step in expected
                            if isinstance(step, dict)
                        ],
                        ensure_ascii=False,
                    ),
                }
            )
    return Dataset.from_list(rows)


def patch_trl_vllm_import() -> None:
    """Local TRL imports GRPO through vLLM helpers; older vLLM lacks this symbol."""
    try:
        import vllm.sampling_params as sampling_params  # type: ignore

        if not hasattr(sampling_params, "StructuredOutputsParams"):
            class StructuredOutputsParams:  # noqa: D401
                """Compatibility placeholder for TRL import-time type checks."""

                pass

            sampling_params.StructuredOutputsParams = StructuredOutputsParams
    except Exception:
        return


def import_grpo():
    patch_trl_vllm_import()
    from trl import GRPOConfig, GRPOTrainer

    return GRPOConfig, GRPOTrainer


def completion_text(value: Any) -> str:
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            content = value[0].get("content")
        else:
            content = "".join(str(item) for item in value)
    else:
        content = value
    return str(content or "").strip()


def first_json_text(value: Any) -> str:
    text, _, _ = first_json_span_text(value)
    return text


def first_json_span_text(value: Any) -> tuple[str, int, int]:
    text = completion_text(value)
    if not text:
        return "", -1, -1
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return text[start : start + end], start, start + end
    return "", -1, -1


def parse_completion(value: Any, *, first_json_only: bool = True) -> dict[str, Any] | None:
    text = first_json_text(value) if first_json_only else completion_text(value)
    if not text:
        return None
    if not first_json_only:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    try:
        parsed = json.loads(text)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def score_step_completion(
    *,
    completion: Any,
    expected_step: str,
    forbidden_actions: str,
    reward_spec: str,
    previous_action: str,
    full_expected_actions: str,
    step_index: int,
    first_json_only: bool = True,
) -> float:
    actual = parse_completion(completion, first_json_only=first_json_only)
    try:
        expected = json.loads(expected_step)
        forbidden = json.loads(forbidden_actions)
        spec = json.loads(reward_spec)
        full_actions = json.loads(full_expected_actions)
    except Exception:
        return 0.0
    if not isinstance(expected, dict) or actual is None:
        return 0.0
    if not isinstance(spec, dict):
        spec = {}
    reward_spec_data = dict(rewardlib.DEFAULT_REWARD_SPEC)
    reward_spec_data.update(spec)
    strict_argument_types = bool(reward_spec_data.get("strict_argument_types", False))
    base_score, info = rewardlib.score_expected_step(
        expected=expected,
        actual=actual,
        reward_spec=reward_spec_data,
    )
    expected_weight = sum(
        float(reward_spec_data.get(key, rewardlib.DEFAULT_REWARD_SPEC.get(key, 0.0)))
        for key in ("json_valid", "decision_type_valid", "action_match", "argument_match", "finish_after_tool")
    )
    action = rewardlib.normalize_action(str(actual.get("action") or ""))
    actual_decision_type = str(actual.get("decision_type") or "")
    if actual_decision_type in {"clarify", "end"}:
        action = actual_decision_type
    forbidden_set = {rewardlib.normalize_action(str(item)) for item in forbidden if str(item).strip()} if isinstance(forbidden, list) else set()
    forbidden_weight = float(reward_spec_data.get("no_forbidden_action", 0.0))
    forbidden_score = 0.0 if action in forbidden_set else forbidden_weight

    numerator = base_score + forbidden_score
    denominator = expected_weight + forbidden_weight

    current_step = int(step_index)
    total_steps = len(full_actions) if isinstance(full_actions, list) else 0
    if current_step < total_steps:
        weight = float(reward_spec_data.get("no_premature_stop", 0.0))
        if weight > 0:
            continues = (
                str(actual.get("decision_type") or "") == "tool"
                and action not in rewardlib.STOP_ACTIONS
                and rewardlib.value_matches(
                    rewardlib.get_arg(actual, "finish_after_tool"),
                    False,
                    strict_types=strict_argument_types,
                )
            )
            numerator += weight if continues else 0.0
            denominator += weight

    if current_step > 1:
        weight = float(reward_spec_data.get("no_repeated_tool", 0.0))
        if weight > 0:
            prev = rewardlib.normalize_action(previous_action)
            expected_current = rewardlib.normalize_action(str(full_actions[current_step - 1] or ""))
            expected_previous = rewardlib.normalize_action(str(full_actions[current_step - 2] or ""))
            repeated_expected = bool(
                rewardlib._accepted_actions(expected_current).intersection(
                    rewardlib._accepted_actions(expected_previous)
                )
            )
            avoids_unexpected_repeat = repeated_expected or not bool(
                rewardlib._accepted_actions(action).intersection(rewardlib._accepted_actions(prev))
            )
            numerator += weight if avoids_unexpected_repeat else 0.0
            denominator += weight

    if current_step == 1:
        weight = float(reward_spec_data.get("no_skip_required_probe", 0.0))
        requires_probe = (
            total_steps >= 2
            and rewardlib.normalize_action(str(full_actions[0] or "")) in rewardlib.DETECTION_ACTIONS
            and rewardlib.normalize_action(str(full_actions[1] or "")) == "migration_advisor"
        )
        if weight > 0 and requires_probe:
            numerator += weight if action in rewardlib.DETECTION_ACTIONS else 0.0
            denominator += weight

    if current_step == total_steps:
        weight = float(reward_spec_data.get("final_tool_finish", 0.0))
        if weight > 0:
            expected_type = str(expected.get("decision_type") or "tool")
            required_args = (
                expected.get("required_args")
                if isinstance(expected.get("required_args"), dict)
                else {}
            )
            requires_finished_tool = (
                expected_type == "tool"
                and required_args.get("finish_after_tool") is True
            )
            if requires_finished_tool:
                finished = rewardlib.value_matches(
                    rewardlib.get_arg(actual, "finish_after_tool"),
                    True,
                    strict_types=strict_argument_types,
                )
            else:
                # Match the full-trajectory scorer: final_tool_finish is only a
                # constraint when the gold step explicitly requires a finished
                # substantive tool.  Retry steps (finish=false) and end steps
                # therefore treat this process item as not applicable.
                finished = True
            numerator += weight if finished else 0.0
            denominator += weight

    normalized = max(0.0, min(1.0, numerator / denominator if denominator > 0 else 0.0))
    wrong_action_cap = reward_spec_data.get("wrong_action_cap")
    action_matched = float(info.get("detail", {}).get("action_match") or 0.0) >= 1.0
    if wrong_action_cap is not None and not action_matched:
        normalized = min(normalized, max(0.0, min(1.0, float(wrong_action_cap))))
    return normalized


def make_reward_func():
    def reward_func(completions, **kwargs) -> list[float]:
        scores: list[float] = []
        expected_steps = kwargs.get("expected_step", [])
        forbidden_actions = kwargs.get("forbidden_actions", [])
        reward_specs = kwargs.get("reward_spec", [])
        previous_actions = kwargs.get("previous_action", [])
        full_expected_actions = kwargs.get("full_expected_actions", [])
        step_indexes = kwargs.get("step_index", [])
        for idx, completion in enumerate(completions):
            scores.append(
                score_step_completion(
                    completion=completion,
                    expected_step=expected_steps[idx],
                    forbidden_actions=forbidden_actions[idx],
                    reward_spec=reward_specs[idx],
                    previous_action=previous_actions[idx],
                    full_expected_actions=full_expected_actions[idx],
                    step_index=int(step_indexes[idx]),
                )
            )
        return scores

    return reward_func


def main() -> None:
    args = parse_args()
    args.model_name_or_path = resolve_model_name_or_path(args.model_name_or_path, ROOT)
    set_seed(args.seed)
    GRPOConfig, GRPOTrainer = import_grpo()

    cases_path = args.cases if args.cases.is_absolute() else ROOT / args.cases
    dataset = build_step_dataset(load_jsonl(cases_path))
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    if args.gradient_checkpointing:
        model.config.use_cache = False

    peft_config = None
    if args.use_lora:
        target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )

    report_to = [] if str(args.report_to).lower() == "none" else [args.report_to]
    fsdp_config = parse_json_arg(args.fsdp_config)
    gradient_checkpointing_kwargs = parse_json_arg(args.gradient_checkpointing_kwargs)
    config_kwargs = {
        "output_dir": args.output_dir,
        "do_train": True,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "logging_steps": args.logging_steps,
        "logging_first_step": True,
        "report_to": report_to,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_checkpointing_kwargs": gradient_checkpointing_kwargs,
        "remove_unused_columns": False,
        "optim": args.optim,
        "seed": args.seed,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "use_vllm": False,
        "fsdp": args.fsdp or "",
        "fsdp_config": fsdp_config,
        "ddp_find_unused_parameters": args.ddp_find_unused_parameters,
        "average_tokens_across_devices": True,
        "torch_empty_cache_steps": args.logging_steps,
    }
    supported = set(inspect.signature(GRPOConfig.__init__).parameters)
    training_args = GRPOConfig(**{key: value for key, value in config_kwargs.items() if key in supported})
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[make_reward_func()],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
