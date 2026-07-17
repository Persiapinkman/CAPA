#!/usr/bin/env python3
"""Apply the preregistered V12 optimizer-matched task and safety gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.planner_grpo_seed_v1.scripts.gate_planner_retry_safe_end_residual_v8_support import (  # noqa: E402
    check,
    load_json,
    load_jsonl,
)
from training.planner_grpo_seed_v1.scripts.gate_planner_retry_safety_balanced_v11_support import (  # noqa: E402
    apply_gate as apply_v11_gate,
)


def apply_gate(
    *,
    preregistration: dict[str, Any],
    data_rows: list[dict[str, Any]],
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = apply_v11_gate(
        preregistration=preregistration,
        data_rows=data_rows,
        samples=samples,
    )
    expected_primary_groups = int(
        preregistration["support_audit"]["expected_primary_groups"]
    )
    observed_primary_groups = int(payload["observed"]["primary"]["groups"])
    if observed_primary_groups != expected_primary_groups:
        raise ValueError(
            "V12 optimizer-matched support has the wrong primary group count: "
            f"expected {expected_primary_groups}, found {observed_primary_groups}"
        )
    primary_safety_groups = sum(
        int(value)
        for value in payload["observed"]["safety"][
            "primary_safety_variance_groups"
        ].values()
    )
    primary_safety_rate = primary_safety_groups / expected_primary_groups
    rate_check = check(
        "primary_safety_variance_rate",
        primary_safety_rate,
        ">=",
        float(
            preregistration["support_audit"]["hard_gates"][
                "minimum_primary_safety_variance_rate"
            ]
        ),
    )
    payload["hard_checks"].append(rate_check)
    payload["observed"]["safety"]["primary_groups"] = expected_primary_groups
    payload["observed"]["safety"]["primary_safety_variance_groups_total"] = (
        primary_safety_groups
    )
    payload["observed"]["safety"]["primary_safety_variance_rate"] = (
        primary_safety_rate
    )
    passed = all(item["passed"] for item in payload["hard_checks"])
    payload["status"] = "pass" if passed else "fail"
    payload["optimizer_authorized"] = passed
    payload["optimizer_scenarios"] = (
        list(preregistration["design"]["primary_scenarios"])
        + list(preregistration["design"]["stability_controls"])
        if passed
        else []
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--step-data", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accepted-scenarios-out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = apply_gate(
        preregistration=load_json(args.preregistration),
        data_rows=load_jsonl(args.step_data),
        samples=load_jsonl(args.samples),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.accepted_scenarios_out.parent.mkdir(parents=True, exist_ok=True)
    args.accepted_scenarios_out.write_text(
        "".join(f"{scenario}\n" for scenario in payload["optimizer_scenarios"]),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
