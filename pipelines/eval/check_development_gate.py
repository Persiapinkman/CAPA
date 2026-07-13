#!/usr/bin/env python3
"""Apply the preregistered development promotion gate to a paired comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    study = load(args.study)
    comparison = load(args.comparison)
    gate = study["development_gate"]
    pair_key = f"{args.baseline_label}_to_{args.candidate_label}"
    paired = comparison["comparisons"][pair_key]
    baseline = comparison["runs"][args.baseline_label]
    candidate = comparison["runs"][args.candidate_label]
    guardrails = study["support_gate"]["guardrail_categories"]
    category_deltas = {
        category: candidate["categories"][category] - baseline["categories"][category]
        for category in guardrails
    }
    primary_passed = paired["case_macro_delta"] >= gate["minimum_case_macro_delta"]
    guardrails_passed = all(
        delta >= -gate["maximum_category_regression"] for delta in category_deltas.values()
    )
    payload = {
        "schema_version": "1.0",
        "study_id": study["study_id"],
        "comparison": str(args.comparison),
        "baseline": args.baseline_label,
        "candidate": args.candidate_label,
        "passed": primary_passed and guardrails_passed,
        "primary": {
            "observed_delta": paired["case_macro_delta"],
            "threshold": gate["minimum_case_macro_delta"],
            "passed": primary_passed,
        },
        "guardrails": {
            "maximum_regression": gate["maximum_category_regression"],
            "category_deltas": category_deltas,
            "passed": guardrails_passed,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
