#!/usr/bin/env python3
"""Freeze SFT and GRPO step-2 stage data for planner_retry_migrate_v7_longobs.

This is the v7 equivalent of the SFT / GRPO stage tables that v6 emits inside
its own builder. We keep it as a separate script to avoid touching the frozen
v6 builder while still reusing its low-level helpers.

Outputs:

  training/planner_grpo_seed_v1/sft_data_planner_retry_migrate_v7_longobs_qwen35_nothinking/
    train.jsonl   dev.jsonl   metadata.json

  training/planner_grpo_seed_v1/step_data/
    planner_retry_migrate_v7_longobs_grpo_train_qwen35_4b_nothinking_step2.jsonl
    planner_retry_migrate_v7_longobs_grpo_train_qwen35_4b_nothinking_step2.manifest.json
    planner_retry_migrate_v7_longobs_grpo_dev_qwen35_4b_nothinking_step2.jsonl
    planner_retry_migrate_v7_longobs_grpo_dev_qwen35_4b_nothinking_step2.manifest.json

v7 uses long observations (~2.4k tokens each), so the total prompt length after
rendering the Qwen3-4B chat template lands in the 6-10k range. We raise the
MAX_PROMPT_TOKENS gate to 12288 and expect training scripts to be launched
with --max-length 12288 or higher.

Determinism: same case JSONL + same tokenizer -> byte-identical output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
for import_root in (ROOT / "src", ROOT, ROOT / "demo"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

# Reuse the frozen v6 pipeline helpers -- their behaviour is orthogonal to
# case content so they work for v7 as long as the case schema is compatible.
from training.planner_grpo_seed_v1.scripts import build_planner_retry_migrate_v6 as v6  # noqa: E402


# Which case family to stage. The 3-step retry variant lives under its own
# dataset_id so v7 artifacts are never overwritten; select it with
# CAPA_STAGE_DATASET_ID=planner_retry_migrate_v8_retry3.
DATASET_ID = os.environ.get(
    "CAPA_STAGE_DATASET_ID", "planner_retry_migrate_v7_longobs"
).strip()
SCHEMA_VERSION = "1.0"
CREATED_AT = "2026-08-01T00:00:00+00:00"

CASE_DIR = ROOT / "training/planner_grpo_seed_v1/cases"
SFT_DIR = ROOT / f"training/planner_grpo_seed_v1/sft_data_{DATASET_ID}_qwen35_nothinking"
STEP_DIR = ROOT / "training/planner_grpo_seed_v1/step_data"
DEFAULT_MODEL = Path(os.environ.get(
    "CAPA_QWEN35_TOKENIZER_DIR",
    "/apdcephfs_hzlf/share_1227201/zkq/capa_h20/models/Qwen3.5-4B",
))

# v7 long-observation gate. The v6 builder used 4608; we need headroom for the
# 1500-4000 token observation payloads.
V7_MAX_PROMPT_TOKENS = 12288

# ---------------------------------------------------------------------------
# Routing-hint stripping
# ---------------------------------------------------------------------------
#``build_planner_retry_migrate_v7_longobs.py`` injects one explicit routing
# sentence into ``observation.summary`` so that a *zero-shot* planner can pass
# the 0.85 dataset-health gate by reading the summary alone.  Its own docstring
# states that SFT/GRPO training "may strip it via CAPA_STRIP_ROUTING_HINT=1 in
# prepare_v7_longobs_stage_data.py" -- but that switch was never implemented
# here, so every SFT and GRPO row shipped with the hint intact.
#
# Consequences observed on 2026-08-03:
#   * SFT ckpt-100 learns to follow the literal hint instead of the
#     observation content (the v6 "rule field leakage" defect in new clothes).
#   * The hint names the *action* but never the arguments, so the residual
#     concentrates on hint-uncovered fields: 42/43 P1+P6 dev failures are
#     `migration_advisor` with `finish_after_tool=false`.
#   * The GRPO optimizer pool saturates (frac_reward_zero_std -> 0.994,
#     grad_norm 0 on 21/22 steps), i.e. no learnable signal remains.
#
# Setting ``CAPA_STRIP_ROUTING_HINT=1`` removes the hint sentence from the
# rendered prompt and replaces it with the neutral fallback wording that the
# builder uses when no hint exists.  Evaluation data should normally KEEP the
# hint (that is what the 35B >= 0.85 health gate was measured against), so the
# switch is per-invocation rather than global.
STRIP_ROUTING_HINT = os.environ.get("CAPA_STRIP_ROUTING_HINT", "0").strip() in {"1", "true", "yes"}

# The builder emits "决策提示：<hint>" as the tail of observation.summary.
_HINT_PREFIX = "决策提示："
_HINT_FALLBACK = "请根据以上信息判断下一步。"

# Populated by ``_render_row`` so the manifest can record what was stripped.
_HINT_STRIP_STATS: dict[str, int] = {"rows": 0, "rows_with_hint": 0, "replacements": 0}


def strip_routing_hint(text: str) -> tuple[str, int]:
    """Replace every``决策提示：...`` tail with the neutral fallback wording.

    The hint always terminates the summary string, and the summary is embedded
    in a JSON payload, so the hint runs until the closing quote of the JSON
    string value.  Returns ``(new_text, replacements)``.
    """
    if _HINT_PREFIX not in text:
        return text, 0

    out: list[str] = []
    cursor = 0
    replacements = 0
    while True:
        start = text.find(_HINT_PREFIX, cursor)
        if start < 0:
            out.append(text[cursor:])
            break
        # The summary is a JSON string value; the hint ends at the next
        # unescaped double quote.
        end = start
        while end < len(text):
            if text[end] == '"' and text[end - 1] != "\\":
                break
            end += 1
        out.append(text[cursor:start])
        out.append(_HINT_FALLBACK)
        replacements += 1
        cursor = end
    return "".join(out), replacements


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_cases(split: str) -> list[dict[str, Any]]:
    path = CASE_DIR / f"{DATASET_ID}_{split}_cases.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _render_row(case: dict[str, Any], step_index: int, tokenizer: Any) -> tuple[str, int]:
    """Return (rendered_prompt, prompt_token_count) using v6 helpers."""
    pseudo = v6.build_sanitized_pseudo_prompt(case, step_index)
    rendered = v6.render_nonthinking_prompt(tokenizer, pseudo)
    if STRIP_ROUTING_HINT:
        rendered, replaced = strip_routing_hint(rendered)
        _HINT_STRIP_STATS["rows"] += 1
        _HINT_STRIP_STATS["replacements"] += replaced
        if replaced:
            _HINT_STRIP_STATS["rows_with_hint"] += 1
    ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    if len(ids) > V7_MAX_PROMPT_TOKENS:
        raise ValueError(
            f"{case.get('case_id')} step {step_index}: "
            f"{len(ids)} prompt tokens exceeds hard gate {V7_MAX_PROMPT_TOKENS}"
        )
    return rendered, len(ids)


def build_sft_rows(cases: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    """SFT rows: one row per (case, step) up to the number of expected steps.

    Deduplicate identical (prompt, completion) pairs (v6 pattern).
    """
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for case in cases:
        expected = case.get("expected_decisions") or []
        for step_index, exp in enumerate(expected, start=1):
            if not isinstance(exp, dict):
                continue
            rendered, prompt_tokens = _render_row(case, step_index, tokenizer)
            decision = v6.canonical_decision(case, exp, step_index=step_index)
            completion_json = json.dumps(decision, ensure_ascii=False, separators=(",", ":"))
            completion = completion_json + "<|im_end|>"
            key = (sha256_text(rendered), sha256_text(completion))
            if key in seen:
                continue
            seen.add(key)
            completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
            row = {
                "prompt": rendered,
                "completion": completion,
                "prompt_sha256": key[0],
                "completion_sha256": key[1],
                "prompt_token_count": prompt_tokens,
                "completion_token_count": len(completion_ids),
                "case_id": case.get("case_id"),
                "entity_id": case.get("entity_id"),
                "counterfactual_bundle_id": case.get("counterfactual_bundle_id"),
                "scenario_id": case.get("scenario_id"),
                "category": case.get("category"),
                "target_action_class": case.get("target_action_class"),
                "expected_step": json.dumps(exp, ensure_ascii=False, sort_keys=True),
                "forbidden_actions": json.dumps(case.get("forbidden_actions") or [], ensure_ascii=False, sort_keys=True),
                "full_expected_actions": json.dumps(
                    [v6.expected_action_name(step) for step in expected if isinstance(step, dict)],
                    ensure_ascii=False, sort_keys=True,
                ),
                "previous_action": v6.expected_action_name(expected[step_index - 2]) if step_index >= 2 else "",
                "reward_spec": json.dumps(case.get("reward_spec") or {}, ensure_ascii=False, sort_keys=True),
                "group_id": case.get("group_id") or case.get("entity_id"),
                "template_id": case.get("template_id"),
                "split": case.get("split"),
                "data_stage": "sft",
                "dataset_id": DATASET_ID,
                "step_index": step_index,
                "source_case_count": 1,
                "source_case_ids": json.dumps([case.get("case_id")], ensure_ascii=False),
            }
            out.append(row)
    return out


def _forbidden_for_step(case: dict[str, Any], step_index: int) -> list[Any]:
    """Per-step forbidden set, falling back to the case-level list.

    The 3-step retry variant (planner_retry_migrate_v8_retry3) ships
    ``forbidden_actions_by_step`` because the illegal moves differ by state:
    at step 2 the retry budget is fresh so ``migration_advisor`` is premature,
    while at step 3 the budget is spent so both detectors are illegal.  Using
    the case-level union for every row would erase exactly the signal this
    variant was built to create.
    """
    by_step = case.get("forbidden_actions_by_step")
    if isinstance(by_step, dict):
        val = by_step.get(str(step_index))
        if isinstance(val, list):
            return val
    return case.get("forbidden_actions") or []


def build_grpo_step_rows(cases: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    """GRPO optimizer rows: one row per (case, target step).

    v7 emitted step-2 only, because every v7 trajectory was 2 steps long and
    the step-2 gold action was always ``migration_advisor`` or ``end`` -- a
    2-action, 75/25 decision space that GRPO exhausts in ~30 optimizer steps.

    Cases may now declare ``grpo_target_steps`` (list) in addition to the
    legacy scalar ``grpo_target_step``.  For the 3-step retry variant that list
    is ``[2, 3]``, which yields two rows per retry case:

        step 2 -- retry the same detector (migrating now is premature)
        step 3 -- migrate (a third probe is illegal)

    The two rows share an ``entity_id``/``group_id`` so entity-level
    independence is preserved for CI purposes, but they are different decisions
    under different states.
    """
    out: list[dict[str, Any]] = []
    for case in cases:
        steps = case.get("grpo_target_steps")
        if not isinstance(steps, list) or not steps:
            steps = [int(case.get("grpo_target_step") or 2)]
        expected = case.get("expected_decisions") or []
        for target_step in (int(s) for s in steps):
            if len(expected) < target_step or target_step < 2:
                continue
            exp = expected[target_step - 1]
            rendered, prompt_tokens = _render_row(case, target_step, tokenizer)
            row = {
                "prompt": rendered,
                "case_id": case.get("case_id"),
                "dataset_id": DATASET_ID,
                "category": case.get("category"),
                "step_index": target_step,
                "expected_step": json.dumps(exp, ensure_ascii=False, sort_keys=True),
                "forbidden_actions": json.dumps(
                    _forbidden_for_step(case, target_step), ensure_ascii=False, sort_keys=True
                ),
                "reward_spec": json.dumps(case.get("reward_spec") or {}, ensure_ascii=False, sort_keys=True),
                "previous_action": v6.expected_action_name(expected[target_step - 2]),
                "entity_id": case.get("entity_id"),
                "group_id": case.get("group_id") or case.get("entity_id"),
                "template_id": case.get("template_id"),
                "scenario_id": case.get("scenario_id"),
                "target_action_class": case.get("target_action_class"),
                "full_expected_actions": json.dumps(
                    [v6.expected_action_name(step) for step in expected if isinstance(step, dict)],
                    ensure_ascii=False, sort_keys=True,
                ),
                "prompt_token_count": prompt_tokens,
                "prompt_sha256": sha256_text(rendered),
            }
            out.append(row)
    return out


# Backwards-compatible alias: v7 callers import the step2-only name.
def build_grpo_step2_rows(cases: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    return build_grpo_step_rows(cases, tokenizer)


def audit_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tokens = sorted(int(r["prompt_token_count"]) for r in rows)
    prompt_hashes = [r["prompt_sha256"] for r in rows]
    audit = {
        "rows": len(rows),
        "duplicate_prompt_hashes": len(prompt_hashes) - len(set(prompt_hashes)),
        "prompt_tokens": {
            "min": tokens[0],
            "p50": tokens[len(tokens) // 2],
            "p95": tokens[int(len(tokens) * 0.95)],
            "max": tokens[-1],
            "mean": statistics.mean(tokens),
        },
        "forbidden_fragment_hits": {},
        "status": "pass",
    }
    # Same forbidden fragments as the v6 audit -- must NOT appear in prompts
    # or completions.
    forbidden = ('"_thought"', '"external_ref"', "/raid/", "/tmp/", "按训练样本期望")
    dump = "".join(r.get("prompt", "") + r.get("completion", "") for r in rows)
    for token in forbidden:
        n = dump.count(token)
        audit["forbidden_fragment_hits"][token] = n
        if n > 0:
            audit["status"] = "fail"
    if audit["duplicate_prompt_hashes"] > 0:
        audit["status"] = "fail"
    return audit


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-prompt-tokens", type=int, default=V7_MAX_PROMPT_TOKENS)
    parser.add_argument(
        "--variant-suffix",
        default=None,
        help=(
            "Optional filename suffix so a variant build never overwrites the "
            "already-audited default artifacts. Defaults to '_nohint' when "
            "CAPA_STRIP_ROUTING_HINT=1, else empty."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    global V7_MAX_PROMPT_TOKENS
    V7_MAX_PROMPT_TOKENS = args.max_prompt_tokens

    suffix = args.variant_suffix
    if suffix is None:
        suffix = "_nohint" if STRIP_ROUTING_HINT else ""
    sft_dir = Path(str(SFT_DIR) + suffix)

    print(f"[v7-stage] loading tokenizer from {args.model_dir}")
    print(f"[v7-stage] strip_routing_hint={STRIP_ROUTING_HINT} variant_suffix={suffix!r}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, trust_remote_code=False, use_fast=True, padding_side="right",
    )

    audits: dict[str, Any] = {}
    files: dict[str, str] = {}
    sha256_map: dict[str, str] = {}

    # --- SFT ---
    for split in ("sft_train", "sft_dev"):
        cases = load_cases(split)
        rows = build_sft_rows(cases, tokenizer)
        a = audit_rows(rows)
        if a["status"] != "pass":
            raise RuntimeError(f"{split} audit failed: {a}")
        target = sft_dir / f"{split.split('_')[1]}.jsonl"  # train.jsonl / dev.jsonl
        write_jsonl(target, rows)
        audits[f"sft_{split.split('_')[1]}"] = {**a, "source_cases": len(cases)}
        files[f"sft_{split.split('_')[1]}"] = str(target.relative_to(ROOT))
        sha256_map[f"sft_{split.split('_')[1]}"] = sha256_file(target)
        print(f"[v7-stage] wrote {target} rows={len(rows)}")

    # --- GRPO step2 ---
    for split in ("grpo_train", "grpo_dev"):
        cases = load_cases(split)
        rows = build_grpo_step_rows(cases, tokenizer)
        a = audit_rows(rows)
        if a["status"] != "pass":
            raise RuntimeError(f"{split} audit failed: {a}")
        # Name the file after the actual target steps: v7 is step2-only, the
        # 3-step retry variant mixes step2 and step3 in one optimizer pool.
        step_tag = "step" + "".join(
            str(x) for x in sorted({int(r["step_index"]) for r in rows})
        )
        target = STEP_DIR / (
            f"{DATASET_ID}_{split}"
            f"_qwen35_4b_nothinking_{step_tag}{suffix}.jsonl"
        )
        write_jsonl(target, rows)
        audits[f"grpo_{split.split('_')[1]}"] = {**a, "source_cases": len(cases)}
        files[f"grpo_{split.split('_')[1]}"] = str(target.relative_to(ROOT))
        sha256_map[f"grpo_{split.split('_')[1]}"] = sha256_file(target)
        # Manifest sidecar (upstream trainer requires this file to exist)
        action_counts: Counter[str] = Counter()
        scenario_counts: Counter[str] = Counter()
        for row in rows:
            action_counts[v6.expected_action_name(json.loads(row["expected_step"]))] += 1
            scenario_counts[row.get("scenario_id") or ""] += 1
        tokenizer_dir = args.model_dir
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_id": DATASET_ID,
            "role": "optimization_only_qwen35_step_data",
            "model_name_or_path": str(tokenizer_dir),
            "cases": str((CASE_DIR / f"planner_retry_migrate_v7_longobs_{split}_cases.jsonl").relative_to(ROOT)),
            "output": str(target.relative_to(ROOT)),
            "rows": len(rows),
            "target_step": 2,
            "prompt_contract": {
                "chat_template": "native_qwen35",
                "enable_thinking": False,
                "suffix": v6.NONTHINKING_SUFFIX,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "max_prompt_tokens_hard_gate": V7_MAX_PROMPT_TOKENS,
                "token_stats": {
                    "min": a["prompt_tokens"]["min"],
                    "mean": a["prompt_tokens"]["mean"],
                    "p50": a["prompt_tokens"]["p50"],
                    "p95": a["prompt_tokens"]["p95"],
                    "max": a["prompt_tokens"]["max"],
                    "over_limit": 0,
                },
            },
            "distribution": {
                "expected_actions": dict(sorted(action_counts.items())),
                "scenarios": dict(sorted(scenario_counts.items())),
            },
            "sha256": {
                "cases": sha256_file(CASE_DIR / f"planner_retry_migrate_v7_longobs_{split}_cases.jsonl"),
                "step_data": sha256_file(target),
                "tokenizer_config": sha256_file(tokenizer_dir / "tokenizer_config.json"),
                "config": sha256_file(tokenizer_dir / "config.json"),
            },
            "routing_hint": {
                "stripped": STRIP_ROUTING_HINT,
                "env": "CAPA_STRIP_ROUTING_HINT",
                "variant_suffix": suffix,
                "hint_prefix": _HINT_PREFIX,
                "fallback": _HINT_FALLBACK,
            },
        }
        write_json(target.with_suffix(".manifest.json"), manifest)
        print(f"[v7-stage] wrote {target} rows={len(rows)} + manifest sidecar")

    # --- SFT metadata (mirrors v6 format so downstream tooling is happy) ---
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "created_at": CREATED_AT,
        "model_name_or_path": str(args.model_dir),
        "prompt_contract": {
            "chat_template": "native_qwen35",
            "enable_thinking": False,
            "assistant_suffix": v6.NONTHINKING_SUFFIX,
            "completion_suffix_sft": "<|im_end|>",
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "max_prompt_tokens": V7_MAX_PROMPT_TOKENS,
            "max_steps": 3,
        },
        "audits": audits,
        "files": files,
        "sha256": sha256_map,
        "routing_hint": {
            "stripped": STRIP_ROUTING_HINT,
            "env": "CAPA_STRIP_ROUTING_HINT",
            "variant_suffix": suffix,
            "hint_prefix": _HINT_PREFIX,
            "fallback": _HINT_FALLBACK,
            "stats": dict(_HINT_STRIP_STATS),
        },
        "deduplication": {
            "rule": "deduplicate identical (prompt_sha256, completion_sha256) pairs across steps",
            "sft_train_rows": audits.get("sft_train", {}).get("rows"),
            "sft_dev_rows": audits.get("sft_dev", {}).get("rows"),
        },
        "rows": {k: v.get("rows") for k, v in audits.items()},
    }
    write_json(sft_dir / "metadata.json", metadata)
    print(f"[v7-stage] wrote {sft_dir / 'metadata.json'}")
    print(json.dumps({k: v["rows"] for k, v in audits.items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
