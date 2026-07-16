#!/usr/bin/env python3
"""Audit GRPO sampled reward support before starting an expensive training run."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src", ROOT, ROOT / "demo"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from training.planner_grpo_seed_v1.scripts.train_planner_grpo import (  # noqa: E402
    parse_completion,
)
from training.planner_grpo_seed_v1.scripts.train_planner_grpo_trl import (  # noqa: E402
    make_reward_func,
)
from util.path_resolver import resolve_model_name_or_path  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--generation-chunk-size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--step", action="append", type=int, default=[])
    parser.add_argument("--task-reward-weight", type=float, default=0.85)
    parser.add_argument("--format-reward-weight", type=float, default=0.15)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Opt in to custom Hub model code. Local Qwen3.5 audits leave this disabled.",
    )
    return parser.parse_args()


def action_name(completion: str) -> str:
    parsed = parse_completion(completion)
    if not isinstance(parsed, dict):
        return "invalid"
    decision_type = str(parsed.get("decision_type") or "")
    if decision_type in {"clarify", "end"}:
        return decision_type
    return str(parsed.get("action") or "missing_action")


def aggregate(groups: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_category_step: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        category = str(group["category"])
        step_index = int(group["step_index"])
        by_category[category].append(group)
        by_step[step_index].append(group)
        by_category_step[(category, step_index)].append(group)

    def stats(items: list[dict[str, Any]]) -> dict[str, Any]:
        result = {
            "groups": len(items),
            "mean_reward": statistics.mean(float(item["mean_reward"]) for item in items),
            "mean_group_std": statistics.mean(float(item["reward_std"]) for item in items),
            "nonzero_std_rate": statistics.mean(float(item["reward_std"] > 1e-6) for item in items),
            "positive_support_rate": statistics.mean(float(item["max_reward"] >= 0.95) for item in items),
            "usable_support_rate": statistics.mean(
                float(item["reward_std"] > 1e-6 and item["max_reward"] >= 0.75)
                for item in items
            ),
            "fully_saturated_rate": statistics.mean(
                float(item["min_reward"] >= 0.95) for item in items
            ),
            "mean_distinct_actions": statistics.mean(
                float(item["distinct_actions"]) for item in items
            ),
        }
        if all("max_task_reward" in item for item in items):
            result.update(
                {
                    "mean_task_reward": statistics.mean(
                        float(item["mean_task_reward"]) for item in items
                    ),
                    "near_exact_task_support_rate": statistics.mean(
                        float(item["max_task_reward"] >= 0.95) for item in items
                    ),
                    "exact_task_support_rate": statistics.mean(
                        float(item["max_task_reward"] >= 1.0 - 1e-9) for item in items
                    ),
                }
            )
        if all("exact_action_support" in item for item in items):
            result["exact_action_support_rate"] = statistics.mean(
                float(bool(item["exact_action_support"])) for item in items
            )
        if all("distinct_valid_actions" in item for item in items):
            result["mean_distinct_valid_actions"] = statistics.mean(
                float(item["distinct_valid_actions"]) for item in items
            )
        return result

    return {
        "overall": stats(groups),
        "categories": {key: stats(value) for key, value in sorted(by_category.items())},
        "steps": {str(key): stats(value) for key, value in sorted(by_step.items())},
        "category_steps": {
            f"{category}::step{step_index}": stats(value)
            for (category, step_index), value in sorted(by_category_step.items())
        },
    }


def main() -> None:
    args = parse_args()
    if args.samples_per_prompt < 2:
        raise ValueError("--samples-per-prompt must be at least 2")
    if args.generation_chunk_size < 1:
        raise ValueError("--generation-chunk-size must be positive")
    torch.manual_seed(args.seed)
    data_path = args.data if args.data.is_absolute() else ROOT / args.data
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    rows = [
        {**row, "_audit_source_index": source_index}
        for source_index, row in enumerate(load_jsonl(data_path))
    ]
    rows = [
        row
        for row in rows
        if (not args.category or str(row["category"]) in args.category)
        and (not args.step or int(row["step_index"]) in args.step)
    ]
    if args.offset < 0:
        raise ValueError("--offset must be non-negative")
    rows = rows[args.offset :]
    if args.limit > 0:
        rows = rows[: args.limit]

    model_path = resolve_model_name_or_path(args.model_name_or_path, ROOT)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    ).to("cuda")
    adapter_path = str((ROOT / args.adapter_path).resolve()) if args.adapter_path else ""
    model = PeftModel.from_pretrained(base, adapter_path).to("cuda") if adapter_path else base
    model.eval()

    reward_func = make_reward_func(
        tokenizer=tokenizer,
        score_first_json_only=True,
        max_completion_length=args.max_new_tokens,
        task_reward_weight=args.task_reward_weight,
        format_reward_weight=args.format_reward_weight,
        tail_penalty_tokens=64,
        prefix_penalty_tokens=16,
        penalize_truncated_completions=True,
    )
    sample_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for row in rows:
        global_row_index = int(row["_audit_source_index"])
        inputs = tokenizer([row["prompt"]], return_tensors="pt").to("cuda")
        completions: list[str] = []
        while len(completions) < args.samples_per_prompt:
            chunk_size = min(
                args.generation_chunk_size, args.samples_per_prompt - len(completions)
            )
            torch.manual_seed(
                args.seed + (global_row_index * args.samples_per_prompt) + len(completions)
            )
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_return_sequences=chunk_size,
                    remove_invalid_values=True,
                    renormalize_logits=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            completions.extend(
                tokenizer.decode(output[inputs.input_ids.shape[1] :], skip_special_tokens=False)
                for output in outputs
            )
        repeated = [row] * len(completions)
        rewards = reward_func(
            completions,
            expected_step=[item["expected_step"] for item in repeated],
            forbidden_actions=[item.get("forbidden_actions", "[]") for item in repeated],
            reward_spec=[item.get("reward_spec", "{}") for item in repeated],
            previous_action=[item.get("previous_action", "") for item in repeated],
            full_expected_actions=[item.get("full_expected_actions", "[]") for item in repeated],
            step_index=[int(item["step_index"]) for item in repeated],
        )
        actions = [action_name(completion) for completion in completions]
        for sample_index, (completion, reward, action) in enumerate(
            zip(completions, rewards, actions), start=1
        ):
            sample_rows.append(
                {
                    "case_id": row["case_id"],
                    "category": row["category"],
                    "step_index": int(row["step_index"]),
                    "entity_id": str(row.get("entity_id") or ""),
                    "sample_index": sample_index,
                    "reward": reward,
                    "action": action,
                    "completion": completion,
                }
            )
        group_rows.append(
            {
                "case_id": row["case_id"],
                "category": row["category"],
                "step_index": int(row["step_index"]),
                "entity_id": str(row.get("entity_id") or ""),
                "mean_reward": statistics.mean(rewards),
                "reward_std": statistics.pstdev(rewards),
                "min_reward": min(rewards),
                "max_reward": max(rewards),
                "distinct_actions": len(set(actions)),
                "action_counts": dict(Counter(actions)),
            }
        )

    summary = {
        "schema_version": "1.0",
        "model": model_path,
        "adapter_path": adapter_path,
        "data": str(data_path),
        "rows": len(rows),
        "offset": args.offset,
        "filters": {
            "categories": args.category,
            "steps": args.step,
        },
        "sampling": {
            "samples_per_prompt": args.samples_per_prompt,
            "generation_chunk_size": args.generation_chunk_size,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
        },
        "reward_weights": {
            "task": args.task_reward_weight,
            "format": args.format_reward_weight,
        },
        "support": aggregate(group_rows),
    }
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "groups.jsonl", group_rows)
    write_jsonl(output_dir / "samples.jsonl", sample_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
