#!/usr/bin/env python3
"""Freeze the V12 1:1 non-migration/migration optimizer dataset."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.planner_grpo_seed_v1.scripts.freeze_planner_retry_migrate_residual_v7_optimizer_data import (  # noqa: E402
    freeze_optimizer_data,
)


PRIMARY = {
    "current_success_step2",
    "fresh_retry_step2",
    "post_retry_success_step3",
}
CONTROLS = {
    "post_retry_error_step3",
    "post_retry_metric_veto_step3",
    "conflicting_state_step2",
    "nonretryable_step2",
    "budget_exhausted_step2",
    "missing_required_state_step2",
}
EXPECTED_SCENARIOS = PRIMARY | CONTROLS
EXPECTED_ACTIONS = Counter({"end": 192, "retry": 96, "migrate": 288})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--support-decision", type=Path, required=True)
    parser.add_argument("--accepted-scenarios", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    accepted = {
        line.strip()
        for line in args.accepted_scenarios.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if accepted != EXPECTED_SCENARIOS:
        raise ValueError("V12 optimizer scope is not the preregistered all-or-none set")
    manifest = freeze_optimizer_data(
        source_path=args.source,
        support_decision_path=args.support_decision,
        accepted_scenarios_path=args.accepted_scenarios,
        output_path=args.output,
        manifest_path=args.manifest,
        allowed_optimization_scopes={"primary_residual", "stability_control"},
        scenario_multipliers={scenario: 2 for scenario in PRIMARY},
    )
    rows = [
        json.loads(line)
        for line in args.output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    actions = Counter(str(row["target_action_class"]) for row in rows)
    if (
        manifest["rows"] != 576
        or set(manifest["accepted_scenarios"]) != EXPECTED_SCENARIOS
        or actions != EXPECTED_ACTIONS
    ):
        raise ValueError(
            f"V12 optimizer manifest violates action balance: rows={manifest['rows']} "
            f"actions={actions}"
        )
    manifest["distribution"]["target_actions"] = dict(sorted(actions.items()))
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
