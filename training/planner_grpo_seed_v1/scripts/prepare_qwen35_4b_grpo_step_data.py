#!/usr/bin/env python3
"""Freeze Qwen3.5 non-thinking step-2 prompts for the V5 GRPO train pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.planner_grpo_seed_v1.scripts.train_planner_grpo import (  # noqa: E402
    build_prompt_for_step,
    expected_action_name,
    load_jsonl,
)


DEFAULT_MODEL = Path("/raid/zkq/models/Qwen3.5-4B")
DEFAULT_CASES = (
    ROOT
    / "training/planner_grpo_seed_v1/cases/"
    "planner_multistep_grpo_value_v5_train_v1_train_cases.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT
    / "training/planner_grpo_seed_v1/step_data/"
    "planner_multistep_grpo_value_v5_train_v1_qwen35_4b_nothinking_step2.jsonl"
)
EXPECTED_DATASET_ID = "planner_multistep_grpo_value_v5_train_v1"
EXPECTED_EOS_ID = 248046
EXPECTED_PAD_ID = 248044
NONTHINKING_SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-prompt-tokens", type=int, default=4608)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pseudo_prompt_to_messages(prompt: str) -> list[dict[str, str]]:
    prefix = "<|system|>\n"
    separator = "\n<|user|>\n"
    suffix = "\n<|assistant|>\n"
    if not prompt.startswith(prefix) or separator not in prompt or not prompt.endswith(suffix):
        raise ValueError("prompt does not match the CAPA pseudo-chat contract")
    body = prompt[len(prefix) : -len(suffix)]
    system, user = body.split(separator, 1)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def render_nonthinking_prompt(tokenizer: Any, prompt: str) -> str:
    rendered = tokenizer.apply_chat_template(
        pseudo_prompt_to_messages(prompt),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not rendered.endswith(NONTHINKING_SUFFIX):
        raise ValueError(f"unexpected Qwen3.5 non-thinking template tail: {rendered[-96:]!r}")
    return rendered


def percentile(sorted_values: list[int], quantile: float) -> int:
    index = min(len(sorted_values) - 1, int((len(sorted_values) - 1) * quantile))
    return sorted_values[index]


def build_row(case: dict[str, Any], tokenizer: Any, prompt_root: Path, max_prompt_tokens: int) -> dict[str, Any]:
    if case.get("dataset_id") != EXPECTED_DATASET_ID:
        raise ValueError(f"unexpected dataset_id for {case.get('case_id')}: {case.get('dataset_id')}")
    if case.get("training_only") is not True or case.get("evaluation_only") is not False:
        raise ValueError(f"invalid train/eval role flags for {case.get('case_id')}")
    target_step = int(case.get("grpo_target_step") or 0)
    expected = case.get("expected_decisions")
    if target_step != 2 or not isinstance(expected, list) or len(expected) < target_step:
        raise ValueError(f"{case.get('case_id')} must define grpo_target_step=2")

    pseudo = build_prompt_for_step(case, target_step, prompt_root)
    prompt = render_nonthinking_prompt(tokenizer, pseudo)
    token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(token_ids) > max_prompt_tokens:
        raise ValueError(
            f"{case.get('case_id')} has {len(token_ids)} prompt tokens, above hard gate {max_prompt_tokens}"
        )

    expected_step = expected[target_step - 1]
    previous = expected[target_step - 2]
    if not isinstance(expected_step, dict) or not isinstance(previous, dict):
        raise ValueError(f"invalid expected decision shape for {case.get('case_id')}")
    return {
        "prompt": prompt,
        "case_id": str(case.get("case_id") or ""),
        "dataset_id": EXPECTED_DATASET_ID,
        "category": str(case.get("category") or ""),
        "step_index": target_step,
        "expected_step": json.dumps(expected_step, ensure_ascii=False, sort_keys=True),
        "forbidden_actions": json.dumps(case.get("forbidden_actions") or [], ensure_ascii=False, sort_keys=True),
        "reward_spec": json.dumps(case.get("reward_spec") or {}, ensure_ascii=False, sort_keys=True),
        "previous_action": expected_action_name(previous),
        "entity_id": str(case.get("entity_id") or ""),
        "group_id": str(case.get("group_id") or case.get("entity_id") or case.get("case_id") or ""),
        "template_id": str(case.get("template_id") or ""),
        "scenario_id": str(case.get("scenario_id") or case.get("category") or ""),
        "target_action_class": str(case.get("target_action_class") or ""),
        "full_expected_actions": json.dumps(
            [expected_action_name(step) for step in expected if isinstance(step, dict)],
            ensure_ascii=False,
            sort_keys=True,
        ),
        "prompt_token_count": len(token_ids),
        "prompt_sha256": sha256_text(prompt),
    }


def main() -> None:
    args = parse_args()
    model_path = args.model_name_or_path.resolve()
    cases_path = args.cases if args.cases.is_absolute() else ROOT / args.cases
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=False,
        use_fast=True,
        padding_side="left",
    )
    if tokenizer.eos_token_id != EXPECTED_EOS_ID or tokenizer.pad_token_id != EXPECTED_PAD_ID:
        raise ValueError(
            "Qwen3.5 tokenizer stop contract mismatch: "
            f"eos={tokenizer.eos_token_id}, pad={tokenizer.pad_token_id}"
        )

    cases = load_jsonl(cases_path)
    if len(cases) != 480:
        raise ValueError(f"expected 480 cases, found {len(cases)}")
    prompt_root = output_path.parent / ".prompt_contexts"
    rows = [build_row(case, tokenizer, prompt_root, args.max_prompt_tokens) for case in cases]
    prompt_hashes = [row["prompt_sha256"] for row in rows]
    if len(set(prompt_hashes)) != len(rows):
        raise ValueError("rendered prompts are not unique")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    lengths = sorted(int(row["prompt_token_count"]) for row in rows)
    action_counts = Counter(json.loads(row["expected_step"])["action"] for row in rows)
    scenario_counts = Counter(row["scenario_id"] for row in rows)
    template_path = model_path / "chat_template.jinja"
    if not template_path.exists():
        raise FileNotFoundError(template_path)
    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": EXPECTED_DATASET_ID,
        "role": "optimization_only_qwen35_step_data",
        "model_name_or_path": str(model_path),
        "cases": str(cases_path),
        "output": str(output_path),
        "rows": len(rows),
        "target_step": 2,
        "prompt_contract": {
            "chat_template": "native_qwen35",
            "enable_thinking": False,
            "suffix": NONTHINKING_SUFFIX,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "max_prompt_tokens_hard_gate": args.max_prompt_tokens,
            "token_stats": {
                "min": min(lengths),
                "mean": statistics.fmean(lengths),
                "p50": percentile(lengths, 0.50),
                "p95": percentile(lengths, 0.95),
                "p99": percentile(lengths, 0.99),
                "max": max(lengths),
                "over_limit": sum(length > args.max_prompt_tokens for length in lengths),
            },
        },
        "distribution": {
            "expected_actions": dict(sorted(action_counts.items())),
            "scenarios": dict(sorted(scenario_counts.items())),
        },
        "sha256": {
            "cases": sha256_file(cases_path),
            "step_data": sha256_file(output_path),
            "tokenizer_config": sha256_file(model_path / "tokenizer_config.json"),
            "config": sha256_file(model_path / "config.json"),
            "chat_template": sha256_file(template_path),
        },
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
