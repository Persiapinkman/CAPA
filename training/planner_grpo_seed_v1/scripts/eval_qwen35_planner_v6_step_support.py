#!/usr/bin/env python3
"""Generate and audit Qwen3.5 Planner V6 step completions locally."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.planner_grpo_seed_v1.scripts import reward_planner_grpo as rewardlib  # noqa: E402
from training.planner_grpo_seed_v1.scripts.train_planner_grpo import (  # noqa: E402
    first_json_text,
    parse_completion,
    score_step_completion,
)


EXPECTED_EOS_ID = 248046
EXPECTED_PAD_ID = 248044


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", type=Path, default=Path("/raid/zkq/models/Qwen3.5-4B"))
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--step-data", type=Path, required=True)
    parser.add_argument("--samples-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-scenario",
        action="append",
        default=[],
        help="Optional scenario_id filter; repeat to include multiple scenarios.",
    )
    parser.add_argument(
        "--include-detector",
        action="append",
        default=[],
        choices=("qwen", "rex"),
        help="Optional detector-family filter; repeat to include both families.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: row must be an object")
        rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def action_name(decision: dict[str, Any] | None) -> str:
    if not isinstance(decision, dict):
        return ""
    decision_type = str(decision.get("decision_type") or "")
    if decision_type in {"end", "clarify"}:
        return decision_type
    return rewardlib.normalize_action(str(decision.get("action") or ""))


def detector_family(row: dict[str, Any]) -> str:
    explicit = str(row.get("detector_family") or "").strip().lower()
    if explicit in {"qwen", "rex"}:
        return explicit
    actions = json.loads(str(row.get("full_expected_actions") or "[]"))
    first = str(actions[0] if actions else "")
    return "qwen" if first == "qwen_detection" else "rex" if first == "rexomni_detection" else ""


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in samples:
        groups[str(row["case_id"])].append(row)

    group_rows: list[dict[str, Any]] = []
    for case_id, values in groups.items():
        rewards = [float(row["score"]) for row in values]
        valid_actions = {str(row["actual_action"]) for row in values if row["json_valid"] and row["actual_action"]}
        group_rows.append(
            {
                "case_id": case_id,
                "target_action_class": values[0]["target_action_class"],
                "detector_family": values[0]["detector_family"],
                "samples": len(values),
                "gold_action_supported": any(bool(row["action_match"]) for row in values),
                "reward_mean": statistics.fmean(rewards),
                "reward_std": statistics.pstdev(rewards),
                "distinct_valid_actions": len(valid_actions),
            }
        )

    def aggregate(selected: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(selected)
        return {
            "groups": count,
            "gold_action_support_rate": (
                sum(bool(row["gold_action_supported"]) for row in selected) / count if count else 0.0
            ),
            "nonzero_reward_variance_rate": (
                sum(float(row["reward_std"]) > 1e-12 for row in selected) / count if count else 0.0
            ),
            "mean_distinct_valid_actions": (
                statistics.fmean(float(row["distinct_valid_actions"]) for row in selected)
                if count
                else 0.0
            ),
            "mean_reward": (
                statistics.fmean(float(row["reward_mean"]) for row in selected) if count else 0.0
            ),
        }

    by_target = {
        target: aggregate([row for row in group_rows if row["target_action_class"] == target])
        for target in sorted({str(row["target_action_class"]) for row in group_rows})
    }
    by_detector = {
        detector: aggregate([row for row in group_rows if row["detector_family"] == detector])
        for detector in sorted({str(row["detector_family"]) for row in group_rows})
    }
    rewards = [float(row["score"]) for row in samples]
    return {
        "samples": len(samples),
        "prompt_groups": len(group_rows),
        "samples_per_prompt": sorted(Counter(len(values) for values in groups.values()).items()),
        "overall": aggregate(group_rows),
        "by_target_action_class": by_target,
        "by_detector_family": by_detector,
        "sample_metrics": {
            "json_valid_rate": sum(bool(row["json_valid"]) for row in samples) / len(samples) if samples else 0.0,
            "action_match_rate": sum(bool(row["action_match"]) for row in samples) / len(samples) if samples else 0.0,
            "argument_match_rate": sum(bool(row["argument_match"]) for row in samples) / len(samples) if samples else 0.0,
            "finish_match_rate": sum(bool(row["finish_match"]) for row in samples) / len(samples) if samples else 0.0,
            "natural_eos_rate": sum(bool(row["natural_eos"]) for row in samples) / len(samples) if samples else 0.0,
            "clipped_rate": sum(bool(row["clipped"]) for row in samples) / len(samples) if samples else 0.0,
            "mean_reward": statistics.fmean(rewards) if rewards else 0.0,
            "nonfinite_rewards": sum(not math.isfinite(value) for value in rewards),
        },
        "actual_actions": dict(sorted(Counter(str(row["actual_action"]) for row in samples).items())),
        "groups_detail": group_rows,
    }


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        raise ValueError("require num_shards>=1 and 0<=shard_index<num_shards")
    if args.samples_per_prompt < 1:
        raise ValueError("samples_per_prompt must be positive")

    model_path = args.model_name_or_path.resolve()
    adapter_path = args.adapter_path.resolve() if args.adapter_path is not None else None
    step_data = args.step_data if args.step_data.is_absolute() else ROOT / args.step_data
    samples_out = args.samples_out if args.samples_out.is_absolute() else ROOT / args.samples_out
    summary_out = args.summary_out if args.summary_out.is_absolute() else ROOT / args.summary_out
    all_rows = load_jsonl(step_data)
    if args.include_scenario:
        included_scenarios = set(args.include_scenario)
        all_rows = [
            row for row in all_rows if str(row.get("scenario_id") or "") in included_scenarios
        ]
    if args.include_detector:
        included_detectors = set(args.include_detector)
        all_rows = [row for row in all_rows if detector_family(row) in included_detectors]
    if not all_rows:
        raise ValueError("step-data filters selected zero rows")
    indexed_rows = [(index, row) for index, row in enumerate(all_rows) if index % args.num_shards == args.shard_index]

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=False,
        use_fast=True,
        padding_side="left",
    )
    if tokenizer.eos_token_id != EXPECTED_EOS_ID or tokenizer.pad_token_id != EXPECTED_PAD_ID:
        raise RuntimeError("Qwen3.5 tokenizer stop contract changed")
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).to("cuda")
    model: Any = base
    if adapter_path is not None:
        model = PeftModel.from_pretrained(base, adapter_path).to("cuda")
    model.eval()

    samples_out.parent.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    with samples_out.open("w", encoding="utf-8") as handle:
        for global_index, row in indexed_rows:
            inputs = tokenizer(str(row["prompt"]), return_tensors="pt", add_special_tokens=False).to("cuda")
            for sample_index in range(args.samples_per_prompt):
                sample_seed = args.seed + global_index * 1009 + sample_index * 9176
                torch.manual_seed(sample_seed)
                torch.cuda.manual_seed_all(sample_seed)
                generation_kwargs: dict[str, Any] = {
                    "max_new_tokens": args.max_new_tokens,
                    "pad_token_id": EXPECTED_PAD_ID,
                    "eos_token_id": EXPECTED_EOS_ID,
                    "remove_invalid_values": True,
                    "renormalize_logits": True,
                    "use_cache": True,
                }
                if args.temperature > 0:
                    generation_kwargs.update(
                        {
                            "do_sample": True,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                        }
                    )
                else:
                    generation_kwargs["do_sample"] = False
                with torch.inference_mode():
                    output = model.generate(**inputs, **generation_kwargs)
                ids = output[0, inputs.input_ids.shape[1] :].detach().cpu().tolist()
                completion = tokenizer.decode(ids, skip_special_tokens=False)
                scored = first_json_text(completion)
                actual = parse_completion(scored, first_json_only=True)
                expected = json.loads(str(row["expected_step"]))
                spec = {**rewardlib.DEFAULT_REWARD_SPEC, **json.loads(str(row["reward_spec"]))}
                _, info = rewardlib.score_expected_step(expected=expected, actual=actual, reward_spec=spec)
                detail = info.get("detail", {})
                score = score_step_completion(
                    completion=scored,
                    expected_step=str(row["expected_step"]),
                    forbidden_actions=str(row["forbidden_actions"]),
                    reward_spec=str(row["reward_spec"]),
                    previous_action=str(row.get("previous_action") or ""),
                    full_expected_actions=str(row["full_expected_actions"]),
                    step_index=int(row["step_index"]),
                    first_json_only=True,
                )
                sample = {
                    "case_id": str(row["case_id"]),
                    "global_row_index": global_index,
                    "sample_index": sample_index,
                    "target_action_class": str(row.get("target_action_class") or ""),
                    "detector_family": detector_family(row),
                    "score": score,
                    "json_valid": actual is not None,
                    "action_match": float(detail.get("action_match") or 0.0) >= 1.0,
                    "argument_match": float(detail.get("argument_match") or 0.0) >= 1.0,
                    "finish_match": float(detail.get("finish_after_tool") or 0.0) >= 1.0,
                    "actual_action": action_name(actual),
                    "completion_tokens": len(ids),
                    "natural_eos": EXPECTED_EOS_ID in ids,
                    "clipped": len(ids) >= args.max_new_tokens and EXPECTED_EOS_ID not in ids,
                    "completion": completion,
                    "scored_completion": scored,
                }
                samples.append(sample)
                handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()

    summary = {
        "model_name_or_path": str(model_path),
        "adapter_path": str(adapter_path) if adapter_path is not None else "",
        "step_data": str(step_data),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "generation": {
            "samples_per_prompt": args.samples_per_prompt,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
        },
        "filters": {
            "scenario_ids": list(args.include_scenario),
            "detector_families": list(args.include_detector),
        },
        **summarize_samples(samples),
    }
    write_json(summary_out, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
