#!/usr/bin/env python3
"""Generate and score Planner SFT outputs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "demo"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from training.planner_grpo_seed_v1.scripts.train_planner_grpo import first_json_text, score_step_completion  # noqa: E402
from training.planner_grpo_seed_v1.scripts import reward_planner_grpo as rewardlib  # noqa: E402
from util.path_resolver import resolve_model_name_or_path  # noqa: E402


DEFAULT_EVAL_FILE = ROOT / "training" / "planner_grpo_seed_v1" / "sft_data" / "val.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Planner SFT outputs with rule reward.")
    parser.add_argument("--model-name-or-path", default="/raid/zkq/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=ROOT / "training" / "planner_grpo_seed_v1" / "reports" / "sft_eval.json")
    parser.add_argument("--predictions-out", type=Path, default=ROOT / "training" / "planner_grpo_seed_v1" / "reports" / "sft_eval_predictions.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--score-first-json-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Score the first complete JSON object while preserving raw completion stats.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def first_json_span(text: str) -> tuple[dict[str, Any] | None, int, int]:
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed, start, start + end
    return None, -1, -1


def completion_stats(text: str) -> dict[str, Any]:
    parsed, start, end = first_json_span(text)
    extra = text[end:].strip() if end >= 0 else text.strip()
    for stop_text in ("<|im_end|>", "<|endoftext|>"):
        while extra.startswith(stop_text):
            extra = extra[len(stop_text) :].strip()
    return {
        "json_valid": parsed is not None,
        "first_json_start": start,
        "first_json_end": end,
        "extra_text_after_json": bool(extra),
        "extra_text_chars": len(extra),
    }


def summarize(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_category[str(row["category"])].append(row)

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows)
        metrics = {
            "count": n,
            "mean_score": sum(float(row["score"]) for row in rows) / n if n else 0.0,
            "json_valid_rate": sum(bool(row["json_valid"]) for row in rows) / n if n else 0.0,
            "extra_text_after_json_rate": sum(bool(row["extra_text_after_json"]) for row in rows) / n if n else 0.0,
            "mean_extra_text_chars": sum(int(row["extra_text_chars"]) for row in rows) / n if n else 0.0,
            "effective_json_valid_rate": sum(bool(row["effective_json_valid"]) for row in rows) / n if n else 0.0,
            "effective_extra_text_after_json_rate": sum(bool(row["effective_extra_text_after_json"]) for row in rows) / n if n else 0.0,
            "effective_mean_extra_text_chars": sum(int(row["effective_extra_text_chars"]) for row in rows) / n if n else 0.0,
        }
        if rows and all("action_match" in row for row in rows):
            metrics["action_match_rate"] = sum(bool(row["action_match"]) for row in rows) / n
            metrics["argument_match_rate"] = sum(bool(row["argument_match"]) for row in rows) / n
            metrics["finish_match_rate"] = sum(bool(row["finish_match"]) for row in rows) / n
        return metrics

    return {
        "overall": aggregate(predictions),
        "categories": {category: aggregate(rows) for category, rows in sorted(by_category.items())},
        "score_distribution": dict(sorted(Counter(round(float(row["score"]), 3) for row in predictions).items())),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    model_path = resolve_model_name_or_path(args.model_name_or_path, ROOT)
    eval_path = args.eval_file if args.eval_file.is_absolute() else ROOT / args.eval_file
    rows = load_jsonl(eval_path)
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        raise ValueError("require num_shards>=1 and 0<=shard_index<num_shards")
    rows = rows[args.shard_index :: args.num_shards]
    if args.limit > 0:
        rows = rows[: args.limit]

    pred_path = args.predictions_out if args.predictions_out.is_absolute() else ROOT / args.predictions_out
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.write_text("", encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False, use_fast=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).to("cuda")
    if args.adapter_path:
        model = PeftModel.from_pretrained(base, args.adapter_path).to("cuda")
    else:
        model = base
    model.eval()

    predictions: list[dict[str, Any]] = []
    for row in rows:
        inputs = tokenizer([row["prompt"]], return_tensors="pt").to("cuda")
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": args.max_new_tokens,
            "remove_invalid_values": True,
            "renormalize_logits": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if args.temperature > 0:
            generation_kwargs.update({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p})
        else:
            generation_kwargs["do_sample"] = False
        with torch.no_grad():
            output = model.generate(**inputs, **generation_kwargs)
        completion = tokenizer.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=False)
        stats = completion_stats(completion)
        scored_completion = first_json_text(completion) if args.score_first_json_only else completion
        effective_stats = completion_stats(scored_completion)
        expected_step = row["expected_step"]
        score = score_step_completion(
            completion=scored_completion,
            expected_step=expected_step,
            forbidden_actions=row.get("forbidden_actions", "[]"),
            reward_spec=row.get("reward_spec", "{}"),
            previous_action=row.get("previous_action", ""),
            full_expected_actions=row.get("full_expected_actions", "[]"),
            step_index=int(row["step_index"]),
            first_json_only=args.score_first_json_only,
        )
        actual = None
        if effective_stats["json_valid"]:
            parsed_decisions, _ = rewardlib.as_decision_list(json.loads(scored_completion))
            actual = parsed_decisions[0] if parsed_decisions else None
        expected_object = json.loads(expected_step)
        _, detail_info = rewardlib.score_expected_step(
            expected=expected_object,
            actual=actual,
            reward_spec={
                **rewardlib.DEFAULT_REWARD_SPEC,
                **json.loads(row.get("reward_spec", "{}")),
            },
        )
        detail = detail_info.get("detail", {})
        prediction = {
            "case_id": row["case_id"],
            "category": row["category"],
            "step_index": int(row["step_index"]),
            "score": score,
            "action_match": float(detail.get("action_match") or 0.0) >= 1.0,
            "argument_match": float(detail.get("argument_match") or 0.0) >= 1.0,
            "finish_match": float(detail.get("finish_after_tool") or 0.0) >= 1.0,
            "detector_family": row.get("detector_family", ""),
            "target_action_class": row.get("target_action_class", ""),
            "scenario_id": row.get("scenario_id", ""),
            "group_id": row.get("group_id", ""),
            **stats,
            "effective_json_valid": effective_stats["json_valid"],
            "effective_extra_text_after_json": effective_stats["extra_text_after_json"],
            "effective_extra_text_chars": effective_stats["extra_text_chars"],
            "scored_completion": scored_completion,
            "completion": completion,
        }
        predictions.append(prediction)
        with pred_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        if args.progress_every > 0 and (
            len(predictions) % args.progress_every == 0 or len(predictions) == len(rows)
        ):
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "completed": len(predictions),
                        "total": len(rows),
                        "shard_index": args.shard_index,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    report = {
        "model_name_or_path": model_path,
        "adapter_path": args.adapter_path,
        "eval_file": str(eval_path),
        "limit": args.limit,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "score_first_json_only": args.score_first_json_only,
        **summarize(predictions),
    }
    out_path = args.out if args.out.is_absolute() else ROOT / args.out
    write_json(out_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
