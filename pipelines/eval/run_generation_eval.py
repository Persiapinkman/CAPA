#!/usr/bin/env python3
"""Run a repeated local generation evaluation and emit a complete run record."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (SRC, ROOT, ROOT / "demo"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from capa.experiments.registry import current_git_commit, sha256_file  # noqa: E402
from training.planner_grpo_seed_v1.scripts.eval_planner_sft_outputs import (  # noqa: E402
    completion_stats,
    load_jsonl,
    summarize,
)
from training.planner_grpo_seed_v1.scripts.train_planner_grpo import (  # noqa: E402
    first_json_text,
    score_step_completion,
)
from util.path_resolver import resolve_model_name_or_path  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def decision_action(value: str | dict[str, Any]) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return ""
    if not isinstance(value, dict):
        return ""
    decision_type = str(value.get("decision_type") or "")
    if decision_type in {"clarify", "end"}:
        return decision_type
    return str(value.get("action") or "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--parent-run-id", default="")
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--prompt-format", choices=["pseudo", "qwen_chatml"], required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--score-first-json-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def git_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    return bool(result.stdout.strip())


def environment_snapshot() -> dict[str, Any]:
    import accelerate
    import datasets
    import peft
    import transformers
    import trl

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "trl": trl.__version__,
        "peft": peft.__version__,
        "datasets": datasets.__version__,
        "accelerate": accelerate.__version__,
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu": gpu_name,
        "dtype": "float16",
        "attention": "sdpa",
    }


def aggregate_repeats(summaries: list[dict[str, Any]], predictions: list[list[dict[str, Any]]]) -> dict[str, Any]:
    metric_names = (
        "mean_score",
        "action_match_rate",
        "json_valid_rate",
        "extra_text_after_json_rate",
        "mean_extra_text_chars",
        "effective_json_valid_rate",
        "effective_extra_text_after_json_rate",
        "effective_mean_extra_text_chars",
    )

    def aggregate_values(values: list[float]) -> dict[str, float]:
        return {
            "mean": statistics.mean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }

    overall = {
        name: aggregate_values([float(summary["overall"][name]) for summary in summaries])
        for name in metric_names
    }
    categories = sorted(
        {category for summary in summaries for category in summary.get("categories", {})}
    )
    category_metrics: dict[str, Any] = {}
    for category in categories:
        category_metrics[category] = {
            name: aggregate_values(
                [float(summary["categories"][category][name]) for summary in summaries]
            )
            for name in metric_names
        }

    keyed_runs: list[dict[tuple[str, int], dict[str, Any]]] = []
    for run in predictions:
        keyed_runs.append({(str(row["case_id"]), int(row["step_index"])): row for row in run})
    keys = sorted(set.intersection(*(set(run) for run in keyed_runs))) if keyed_runs else []
    stable = sum(
        len({round(float(run[key]["score"]), 12) for run in keyed_runs}) == 1
        and len({str(run[key]["scored_completion"]) for run in keyed_runs}) == 1
        for key in keys
    )
    return {
        "repeat_count": len(summaries),
        "overall": overall,
        "categories": category_metrics,
        "determinism": {
            "rows_compared": len(keys),
            "identical_score_and_completion_rows": stable,
            "agreement_rate": stable / len(keys) if keys else 0.0,
        },
    }


def evaluate_once(
    *,
    rows: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    repeat_index: int,
) -> list[dict[str, Any]]:
    torch.manual_seed(args.seed)
    predictions: list[dict[str, Any]] = []
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    for batch_start in range(0, len(rows), args.batch_size):
        batch_rows = rows[batch_start : batch_start + args.batch_size]
        inputs = tokenizer(
            [row["prompt"] for row in batch_rows], padding=True, return_tensors="pt"
        ).to("cuda")
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "remove_invalid_values": True,
            "renormalize_logits": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        with torch.no_grad():
            output = model.generate(**inputs, **generation_kwargs)
        for row_index, row in enumerate(batch_rows):
            completion = tokenizer.decode(
                output[row_index, inputs.input_ids.shape[1] :], skip_special_tokens=False
            )
            stats = completion_stats(completion)
            scored_completion = (
                first_json_text(completion) if args.score_first_json_only else completion
            )
            effective_stats = completion_stats(scored_completion)
            expected_action = decision_action(row["expected_step"])
            actual_action = decision_action(scored_completion)
            score = score_step_completion(
                completion=scored_completion,
                expected_step=row["expected_step"],
                forbidden_actions=row.get("forbidden_actions", "[]"),
                reward_spec=row.get("reward_spec", "{}"),
                previous_action=row.get("previous_action", ""),
                full_expected_actions=row.get("full_expected_actions", "[]"),
                step_index=int(row["step_index"]),
                first_json_only=args.score_first_json_only,
            )
            predictions.append(
                {
                    "case_id": row["case_id"],
                    "category": row["category"],
                    "step_index": int(row["step_index"]),
                    "entity_id": str(row.get("entity_id") or ""),
                    "group_id": str(row.get("group_id") or row["case_id"]),
                    "template_id": str(row.get("template_id") or ""),
                    "scenario_id": str(row.get("scenario_id") or row["category"]),
                    "repeat": repeat_index,
                    "score": score,
                    "expected_action": expected_action,
                    "actual_action": actual_action,
                    "action_match": actual_action == expected_action,
                    **stats,
                    "effective_json_valid": effective_stats["json_valid"],
                    "effective_extra_text_after_json": effective_stats["extra_text_after_json"],
                    "effective_extra_text_chars": effective_stats["extra_text_chars"],
                    "scored_completion": scored_completion,
                    "completion": completion,
                }
            )
    return predictions


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if args.offset < 0:
        raise ValueError("--offset must be non-negative")
    started_at = utc_now()
    started_perf = time.perf_counter()
    run_dir = resolve(args.run_dir)
    artifact_dir = resolve(args.artifact_dir)
    eval_path = resolve(args.eval_file)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolve_model_name_or_path(args.model_name_or_path, ROOT)
    adapter_path = str(resolve(Path(args.adapter_path))) if args.adapter_path else ""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda")
    model = PeftModel.from_pretrained(base, adapter_path).to("cuda") if adapter_path else base
    model.eval()
    torch.cuda.reset_peak_memory_stats()

    source_rows = load_jsonl(eval_path)
    rows = source_rows[args.offset :]
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("evaluation slice is empty")
    all_predictions: list[list[dict[str, Any]]] = []
    summaries: list[dict[str, Any]] = []
    prediction_paths: list[str] = []
    for repeat_index in range(1, args.repeats + 1):
        predictions = evaluate_once(
            rows=rows, model=model, tokenizer=tokenizer, args=args, repeat_index=repeat_index
        )
        summary = summarize(predictions)
        prediction_path = artifact_dir / f"predictions_run{repeat_index}.jsonl"
        write_jsonl(prediction_path, predictions)
        write_json(run_dir / f"metrics_run{repeat_index}.json", summary)
        all_predictions.append(predictions)
        summaries.append(summary)
        prediction_paths.append(str(prediction_path))

    aggregate = aggregate_repeats(summaries, all_predictions)
    aggregate["runtime_seconds"] = time.perf_counter() - started_perf
    aggregate["peak_gpu_memory_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
    write_json(run_dir / "metrics.json", aggregate)

    command = " ".join(shlex.quote(value) for value in [sys.executable, *sys.argv])
    environment = environment_snapshot()
    record = {
        "schema_version": "2.0",
        "run_id": args.run_id,
        "study_id": args.study_id,
        "date": started_at[:10],
        "kind": "eval_generation_repeated",
        "status": "completed",
        "purpose": args.purpose,
        "hypothesis": args.hypothesis,
        "parent_run_id": args.parent_run_id or None,
        "provenance": {
            "git_commit": current_git_commit(ROOT),
            "git_dirty": git_dirty(),
            "command": command,
            "seed": args.seed,
            "started_at": started_at,
            "finished_at": utc_now(),
            "environment": environment,
        },
        "data": {
            "dataset_id": args.dataset_id,
            "split": args.split,
            "files": {"eval": str(eval_path)},
            "sha256": {"eval": sha256_file(eval_path)},
            "rows": len(rows),
            "offset": args.offset,
            "total_source_rows": len(source_rows),
        },
        "method": {
            "model_label": args.model_label,
            "model": model_path,
            "adapter_path": adapter_path,
            "prompt_format": args.prompt_format,
            "generation": {
                "temperature": 0.0,
                "top_p": 1.0,
                "do_sample": False,
                "max_new_tokens": args.max_new_tokens,
                "batch_size": args.batch_size,
                "repeats": args.repeats,
                "score_first_json_only": args.score_first_json_only,
            },
        },
        "metrics": {
            "primary": {
                "name": "step_mean_verifier_score",
                "value": aggregate["overall"]["mean_score"]["mean"],
                "higher_is_better": True,
            },
            "aggregate": aggregate,
        },
        "artifacts": {
            "run_dir": str(run_dir),
            "metrics": str(run_dir / "metrics.json"),
            "predictions": prediction_paths,
        },
        "decision": {
            "outcome": "pending_comparison",
            "rationale": "Run completed; paired study comparison has not yet been applied.",
        },
    }
    write_json(run_dir / "run_record.json", record)
    write_json(
        run_dir / "config.json",
        {
            "run_id": args.run_id,
            "study_id": args.study_id,
            "model_label": args.model_label,
            "model_name_or_path": model_path,
            "adapter_path": adapter_path,
            "prompt_format": args.prompt_format,
            "eval_file": str(eval_path),
            "repeats": args.repeats,
            "seed": args.seed,
            "offset": args.offset,
            "limit": args.limit,
        },
    )
    print(json.dumps({"status": "completed", "run_record": str(run_dir / 'run_record.json')}))


if __name__ == "__main__":
    main()
