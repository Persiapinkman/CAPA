#!/usr/bin/env python3
"""Trainer-faithful GRPO support audit for v7 step-2 pools.

Answers exactly one question before any optimizer step is allowed:

    Does the current initializer, sampled with the *trainer's* decoding
    settings, still produce (a) gold-action support and (b) non-degenerate
    within-group reward variance on this optimizer pool?

Why this exists
---------------
On 2026-08-03 three concurrent `seed43` GRPO runs were observed with
`frac_reward_zero_std = 0.994` and `grad_norm = 0` on 21 of 22 steps: the SFT
initializer had fully saturated the optimizer pool, so GRPO was performing
zero-update busywork.  The project playbook requires that such a support
failure force `optimizer_steps = 0` rather than "train anyway".

This script makes that gate cheap and mechanical, and can compare two pool
variants side by side (e.g. hint-leaking vs hint-stripped prompts) so the
decision "which pool is actually learnable" is data-driven.

Sampling contract
-----------------
Defaults mirror `train_qwen35_4b_grpo.py`: `temperature=0.7`, `top_p=0.9`,
`num_generations=4`, `max_completion_length=320`, first-JSON-only parsing and
the per-row frozen `reward_spec`.  Prompts are sent as raw completions
(`/v1/completions`) because the step data already contains the fully rendered
ChatML prompt including the non-thinking assistant prefix.

Usage
-----
    python pipelines/eval/audit_grpo_support.py \
        --pool hint=training/.../..._step2.jsonl \
        --pool nohint=training/.../..._step2_nohint.jsonl \
        --api-base http://127.0.0.1:8001/v1 --model sft_ckpt100 \
        --prompts 64 --generations 4 --seed 42 \
        --out reports/grpo_support_audit.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.planner_grpo_seed_v1.scripts import reward_planner_grpo as rewardlib  # noqa: E402

# Hard gates. Sourced from the project playbook (V9/V10/V12 lineage).
GATE_JSON_VALID_MIN = 0.99
GATE_CLIPPED_MAX = 0.01
GATE_GOLD_SUPPORT_MIN = 0.80
GATE_NONZERO_VARIANCE_MIN = 0.25


def load_pool(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
                    raise ValueError(f"{path}: empty pool")
    return rows


def post_json(url: str, payload: dict[str, Any], timeout: float, retries: int = 3) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    last: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:  # noqa: PERF203
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"POST {url} failed after {retries} attempts: {last}")


def parse_decision(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a completion."""
    stripped = text.strip()
    start = stripped.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(stripped[start : index + 1])
                except json.JSONDecodeError:
                    return None
                return value if isinstance(value, dict) else None
    return None


def score_completion(row: dict[str, Any], text: str) -> dict[str, Any]:
    """Score one completion against the row's frozen expected_step/reward_spec."""
    expected = row["expected_step"]
    expected = json.loads(expected) if isinstance(expected, str) else expected
    spec_raw = row.get("reward_spec") or {}
    spec_raw = json.loads(spec_raw) if isinstance(spec_raw, str) else spec_raw
    spec = dict(rewardlib.DEFAULT_REWARD_SPEC)
    spec.update(spec_raw)
    forbidden_raw = row.get("forbidden_actions") or []
    forbidden_raw = json.loads(forbidden_raw) if isinstance(forbidden_raw, str) else forbidden_raw
    forbidden = {rewardlib.normalize_action(str(x)) for x in forbidden_raw if str(x).strip()}

    actual = parse_decision(text)
    if actual is None:
        return {
            "json_valid": False,
            "reward": 0.0,
            "gold_action": False,
            "forbidden": False,
            "action": "",
            "detail": {},
        }

    base, info = rewardlib.score_expected_step(expected=expected, actual=actual, reward_spec=spec)
    detail = info.get("detail", {})

    decision_type = str(actual.get("decision_type") or "").strip()
    action = (
        decision_type
        if decision_type in {"clarify", "end"}
        else rewardlib.normalize_action(str(actual.get("action") or ""))
    )
    hit_forbidden = action in forbidden

    step_weight = sum(
        float(spec.get(key, rewardlib.DEFAULT_REWARD_SPEC.get(key, 0.0)))
        for key in ("json_valid", "decision_type_valid", "action_match", "argument_match", "finish_after_tool")
    )
    forbidden_weight = float(spec.get("no_forbidden_action", 0.0))
    total = step_weight + forbidden_weight
    reward = (base + (0.0 if hit_forbidden else forbidden_weight)) / total if total > 0 else 0.0

    return {
        "json_valid": True,
        "reward": round(reward, 6),
        "gold_action": float(detail.get("action_match") or 0.0) >= 1.0,
        "forbidden": hit_forbidden,
        "action": action,
        "detail": {k: float(v) for k, v in detail.items()},
    }


def audit_pool(
    name: str,
    path: Path,
    *,
    api_base: str,
    model: str,
    n_prompts: int,
    generations: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
    timeout: float,
) -> dict[str, Any]:
    rows = load_pool(path)

    # Deterministic, stratified prompt selection: round-robin over categories so
    # a single saturated category cannot dominate the verdict.
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row.get("category") or "unknown")].append(row)
    rng = random.Random(seed)
    for bucket in by_category.values():
        rng.shuffle(bucket)
    selected: list[dict[str, Any]] = []
    categories = sorted(by_category)
    index = 0
    while len(selected) < min(n_prompts, len(rows)):
        bucket = by_category[categories[index % len(categories)]]
        if bucket:
            selected.append(bucket.pop())
        index += 1
        if all(not bucket for bucket in by_category.values()):
            break

    url = api_base.rstrip("/") + "/completions"
    groups: list[dict[str, Any]] = []
    json_valid = 0
    clipped = 0
    total_completions = 0
    action_counter: Counter[str] = Counter()
    component_fail: Counter[str] = Counter()

    for position, row in enumerate(selected, start=1):
        payload = {
            "model": model,
            "prompt": row["prompt"],
            "n": generations,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed + position,
        }
        response = post_json(url, payload, timeout=timeout)
        texts = [choice.get("text") or "" for choice in response.get("choices", [])]
        finish = [choice.get("finish_reason") for choice in response.get("choices", [])]
        if len(texts) != generations:
            raise RuntimeError(
                f"{name}: expected {generations} completions, got {len(texts)} "
                f"for {row.get('case_id')}"
            )

        scored = [score_completion(row, text) for text in texts]
        rewards = [item["reward"] for item in scored]
        for item, reason in zip(scored, finish):
            total_completions += 1
            json_valid += int(item["json_valid"])
            clipped += int(reason == "length")
            action_counter[item["action"] or "<unparsed>"] += 1
            for component, value in item["detail"].items():
                if value < 1.0:
                    component_fail[component] += 1

        std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
        groups.append(
            {
                "case_id": row.get("case_id"),
                "category": row.get("category"),
                "step_index": int(row.get("step_index") or 0),
                "rewards": rewards,
                "reward_std": round(std, 6),
                "reward_mean": round(sum(rewards) / len(rewards), 6),
                "gold_support": any(item["gold_action"] for item in scored),
                "gold_support_rate": sum(item["gold_action"] for item in scored) / len(scored),
                "forbidden_any": any(item["forbidden"] for item in scored),
                "nonzero_variance": std > 1e-9,
            }
        )

    n_groups = len(groups)
    nonzero = sum(group["nonzero_variance"] for group in groups)
    gold_support_groups = sum(group["gold_support"] for group in groups)
    gold_support_completion_rate = (
        sum(group["gold_support_rate"] for group in groups) / max(1, n_groups)
    )

    # Per-step breakdown. Mixed step pools (e.g. the 3-step retry variant emits
    # both step-2 and step-3 rows) can hide a saturated step behind a healthy
    # one: the pooled average says "there is variance" while one of the two
    # decisions is already solved. GRPO suitability has to be judged per
    # decision, so report both.
    per_step: dict[str, dict[str, Any]] = {}
    step_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        step_groups[int(group.get("step_index") or 0)].append(group)
    for step, items in sorted(step_groups.items()):
        per_step[str(step)] = {
            "groups": len(items),
            "nonzero_variance_rate": round(
                sum(item["nonzero_variance"] for item in items) / len(items), 4
            ),
            "gold_support_rate": round(
                sum(item["gold_support_rate"] for item in items) / len(items), 4
            ),
            "reward_mean": round(sum(item["reward_mean"] for item in items) / len(items), 4),
            "forbidden_group_rate": round(
                sum(item["forbidden_any"] for item in items) / len(items), 4
            ),
        }

    per_category: dict[str, dict[str, Any]] = {}
    cat_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        cat_groups[str(group["category"])].append(group)
    for category, items in sorted(cat_groups.items()):
        per_category[category] = {
            "groups": len(items),
            "nonzero_variance_rate": round(
                sum(item["nonzero_variance"] for item in items) / len(items), 4
            ),
            "gold_support_rate": round(
                sum(item["gold_support_rate"] for item in items) / len(items), 4
            ),
            "reward_mean": round(sum(item["reward_mean"] for item in items) / len(items), 4),
        }

    metrics = {
        "groups": n_groups,
        "completions": total_completions,
        "json_valid_rate": round(json_valid / max(1, total_completions), 6),
        "clipped_rate": round(clipped / max(1, total_completions), 6),
        "gold_support_rate": round(gold_support_completion_rate, 6),
        "gold_support_group_rate": round(gold_support_groups / max(1, n_groups), 6),
        "nonzero_variance_rate": round(nonzero / max(1, n_groups), 6),
        "frac_reward_zero_std": round(1.0 - nonzero / max(1, n_groups), 6),
        "reward_mean": round(
            sum(group["reward_mean"] for group in groups) / max(1, n_groups), 6
        ),
        "forbidden_group_rate": round(
            sum(group["forbidden_any"] for group in groups) / max(1, n_groups), 6
        ),
    }

    gates = {
        "json_valid": metrics["json_valid_rate"] >= GATE_JSON_VALID_MIN,
        "clipping": metrics["clipped_rate"] <= GATE_CLIPPED_MAX,
        "gold_support": metrics["gold_support_rate"] >= GATE_GOLD_SUPPORT_MIN,
        "nonzero_variance": metrics["nonzero_variance_rate"] >= GATE_NONZERO_VARIANCE_MIN,
    }

    return {
        "pool": name,
        "path": str(path),
        "sampling": {
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "num_generations": generations,
            "max_tokens": max_tokens,
            "seed": seed,
            "prompt_selection": "category round-robin, seeded shuffle",
        },
        "metrics": metrics,
        "gates": gates,
        "passed": all(gates.values()),
        "thresholds": {
            "json_valid_min": GATE_JSON_VALID_MIN,
            "clipped_max": GATE_CLIPPED_MAX,
            "gold_support_min": GATE_GOLD_SUPPORT_MIN,
            "nonzero_variance_min": GATE_NONZERO_VARIANCE_MIN,
        },
        "action_distribution": dict(action_counter.most_common()),
        "component_failures": dict(component_fail.most_common()),
        "by_category": per_category,
        "by_step": per_step,
        "groups": groups,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--api-base", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=int, default=64)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    results = []
    for spec in args.pool:
        if "=" not in spec:
            raise SystemExit(f"--pool expects NAME=PATH, got {spec!r}")
        name, _, raw = spec.partition("=")
        results.append(
            audit_pool(
                name.strip(),
                Path(raw.strip()),
                api_base=args.api_base,
                model=args.model,
                n_prompts=args.prompts,
                generations=args.generations,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                seed=args.seed,
                timeout=args.timeout,
            )
        )

    report = {
        "tool": "audit_grpo_support",
        "decision_rule": (
            "all hard gates must pass before any optimizer step is authorised; "
            "on failure record optimizer_steps=0 and redesign the pool"
        ),
        "pools": results,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    for pool in results:
        metrics = pool["metrics"]
        print(f"\n===== pool={pool['pool']}({metrics['groups']} groups x "
              f"{pool['sampling']['num_generations']} gen)")
        print(f"  json_valid        = {metrics['json_valid_rate']:.4f}")
        print(f"  clipped           = {metrics['clipped_rate']:.4f}")
        print(f"  gold_support      = {metrics['gold_support_rate']:.4f}")
        print(f"  nonzero_variance  = {metrics['nonzero_variance_rate']:.4f}"
              f"   (frac_reward_zero_std = {metrics['frac_reward_zero_std']:.4f})")
        print(f"  reward_mean       = {metrics['reward_mean']:.4f}")
        print(f"  forbidden_groups  = {metrics['forbidden_group_rate']:.4f}")
        print(f"  GATES: {pool['gates']}  -> passed={pool['passed']}")
        print("  component failures:", pool["component_failures"])
        if len(pool.get("by_step") or {}) > 1:
            print("  by step (groups / nonzero_var / gold_support / reward_mean / forbidden):")
            for step, st in pool["by_step"].items():
                print(
                    f"    step{step:<24s} {st['groups']:3d}  "
                    f"{st['nonzero_variance_rate']:.3f}  {st['gold_support_rate']:.3f}  "
                    f"{st['reward_mean']:.3f}  {st['forbidden_group_rate']:.3f}"
                )
        print("  by category (groups / nonzero_var / gold_support / reward_mean):")
        for category, stats in pool["by_category"].items():
            print(f"    {category:28s} {stats['groups']:3d}  "
                  f"{stats['nonzero_variance_rate']:.3f}  "
                  f"{stats['gold_support_rate']:.3f}  {stats['reward_mean']:.3f}")

    if args.out:
        print(f"\nwrote {args.out}")
    return 0 if all(pool["passed"] for pool in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
