#!/usr/bin/env python3
"""Freeze the V9 all-or-none optimizer dataset after a passing support gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.planner_grpo_seed_v1.scripts.freeze_planner_retry_migrate_residual_v7_optimizer_data import (
    freeze_optimizer_data,
)


EXPECTED_SCENARIOS = {
    "current_success_step2",
    "fresh_retry_step2",
    "post_retry_success_step3",
}


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
        raise ValueError("V9 optimizer scope is not the preregistered all-or-none set")
    manifest = freeze_optimizer_data(
        source_path=args.source,
        support_decision_path=args.support_decision,
        accepted_scenarios_path=args.accepted_scenarios,
        output_path=args.output,
        manifest_path=args.manifest,
    )
    if set(manifest["accepted_scenarios"]) != EXPECTED_SCENARIOS or manifest["rows"] != 144:
        raise ValueError("V9 optimizer manifest violates the preregistered scope")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
