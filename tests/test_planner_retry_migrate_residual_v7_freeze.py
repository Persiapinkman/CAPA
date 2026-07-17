import json

import pytest

from training.planner_grpo_seed_v1.scripts.freeze_planner_retry_migrate_residual_v7_optimizer_data import (
    freeze_optimizer_data,
)


def _write_json(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _fixture(tmp_path):
    source = tmp_path / "source.jsonl"
    rows = []
    for entity in ("entity-a", "entity-b"):
        for detector in ("qwen", "rex"):
            rows.append(
                {
                    "case_id": f"fresh-{entity}-{detector}",
                    "dataset_id": "v7-test",
                    "entity_id": entity,
                    "detector_family": detector,
                    "scenario_id": "fresh_retry_step2",
                    "optimization_scope": "primary_residual",
                    "step_index": 2,
                    "prompt_sha256": f"fresh-{entity}-{detector}-hash",
                }
            )
            rows.append(
                {
                    "case_id": f"control-{entity}-{detector}",
                    "dataset_id": "v7-test",
                    "entity_id": entity,
                    "detector_family": detector,
                    "scenario_id": "nonretryable_step2",
                    "optimization_scope": "stability_control",
                    "step_index": 2,
                    "prompt_sha256": f"control-{entity}-{detector}-hash",
                }
            )
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    decision = tmp_path / "decision.json"
    _write_json(
        decision,
        {
            "status": "pass",
            "optimizer_authorized": True,
            "optimizer_scenarios": ["fresh_retry_step2"],
        },
    )
    accepted = tmp_path / "accepted.txt"
    accepted.write_text("fresh_retry_step2\n", encoding="utf-8")
    return source, decision, accepted


def test_freeze_uses_only_gate_selected_balanced_rows(tmp_path):
    source, decision, accepted = _fixture(tmp_path)
    output = tmp_path / "optimizer.jsonl"
    manifest_path = tmp_path / "optimizer.manifest.json"
    manifest = freeze_optimizer_data(
        source_path=source,
        support_decision_path=decision,
        accepted_scenarios_path=accepted,
        output_path=output,
        manifest_path=manifest_path,
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 4
    assert {row["scenario_id"] for row in rows} == {"fresh_retry_step2"}
    assert manifest["rows"] == 4
    assert manifest["distribution"]["detectors"] == {"qwen": 2, "rex": 2}
    assert manifest["sha256"]["step_data"]


def test_freeze_rejects_decision_file_disagreement(tmp_path):
    source, decision, accepted = _fixture(tmp_path)
    accepted.write_text("fresh_retry_step2\npost_retry_error_step3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        freeze_optimizer_data(
            source_path=source,
            support_decision_path=decision,
            accepted_scenarios_path=accepted,
            output_path=tmp_path / "optimizer.jsonl",
            manifest_path=tmp_path / "optimizer.manifest.json",
        )


def test_freeze_can_explicitly_include_stability_control_replay(tmp_path):
    source, decision, accepted = _fixture(tmp_path)
    scenarios = ["fresh_retry_step2", "nonretryable_step2"]
    _write_json(
        decision,
        {
            "status": "pass",
            "optimizer_authorized": True,
            "optimizer_scenarios": scenarios,
        },
    )
    accepted.write_text("".join(f"{scenario}\n" for scenario in scenarios), encoding="utf-8")
    output = tmp_path / "optimizer.jsonl"
    manifest = freeze_optimizer_data(
        source_path=source,
        support_decision_path=decision,
        accepted_scenarios_path=accepted,
        output_path=output,
        manifest_path=tmp_path / "optimizer.manifest.json",
        allowed_optimization_scopes={"primary_residual", "stability_control"},
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 8
    assert set(manifest["allowed_optimization_scopes"]) == {
        "primary_residual",
        "stability_control",
    }


def test_freeze_accepts_preregistered_scenario_multiplier(tmp_path):
    source, decision, accepted = _fixture(tmp_path)
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    replay = []
    for row in rows:
        if row["scenario_id"] != "fresh_retry_step2":
            continue
        variant = dict(row)
        variant["case_id"] += "-replay"
        variant["prompt_sha256"] += "-replay"
        replay.append(variant)
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows + replay),
        encoding="utf-8",
    )
    output = tmp_path / "optimizer.jsonl"
    manifest = freeze_optimizer_data(
        source_path=source,
        support_decision_path=decision,
        accepted_scenarios_path=accepted,
        output_path=output,
        manifest_path=tmp_path / "optimizer.manifest.json",
        scenario_multipliers={"fresh_retry_step2": 2},
    )
    assert manifest["rows"] == 8
    assert manifest["scenario_multipliers"] == {"fresh_retry_step2": 2}
