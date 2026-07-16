from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from training.planner_grpo_seed_v1.scripts import (
    gate_planner_retry_migrate_residual_v7_support as gate,
)


ROOT = Path(__file__).resolve().parents[1]
PREREG = (
    ROOT
    / "experiments/studies/planner_retry_migrate_residual_v7_qwen35_4b_v1/preregistration.json"
)
STEP_DATA = (
    ROOT
    / "training/planner_grpo_seed_v1/step_data/"
    "planner_retry_migrate_residual_v7_grpo_dev_qwen35_4b_nothinking_mixed_steps.jsonl"
)


def write_synthetic_samples(path: Path, *, disable_fresh_variance: bool) -> None:
    rows = gate.load_jsonl(STEP_DATA)
    samples = []
    for row in rows:
        scenario = str(row["scenario_id"])
        for sample_index in range(4):
            variable = row["optimization_scope"] == "primary_residual"
            if disable_fresh_variance and scenario == "fresh_retry_step2":
                variable = False
            score = 1.0 if not variable or sample_index else 0.2
            samples.append(
                {
                    "case_id": row["case_id"],
                    "sample_index": sample_index,
                    "score": score,
                    "action_match": sample_index > 0 or not variable,
                    "json_valid": True,
                    "clipped": False,
                }
            )
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in samples),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("disable_fresh_variance", "expected_status"),
    [(False, "pass"), (True, "fail")],
)
def test_preregistered_gate_requires_fresh_retry_support(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    disable_fresh_variance: bool,
    expected_status: str,
) -> None:
    samples = tmp_path / "samples.jsonl"
    output = tmp_path / "decision.json"
    accepted = tmp_path / "accepted.txt"
    write_synthetic_samples(samples, disable_fresh_variance=disable_fresh_variance)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gate",
            "--preregistration",
            str(PREREG),
            "--step-data",
            str(STEP_DATA),
            "--samples",
            str(samples),
            "--output",
            str(output),
            "--accepted-scenarios-out",
            str(accepted),
        ],
    )
    gate.main()
    decision = json.loads(output.read_text(encoding="utf-8"))
    assert decision["status"] == expected_status
    if expected_status == "pass":
        assert decision["optimizer_scenarios"] == list(gate.PRIMARY_ORDER)
        assert accepted.read_text(encoding="utf-8").splitlines() == list(gate.PRIMARY_ORDER)
    else:
        assert decision["optimizer_scenarios"] == []
        assert accepted.read_text(encoding="utf-8") == ""
