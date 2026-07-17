from pathlib import Path

import pytest

from training.planner_grpo_seed_v1.scripts.select_planner_retry_safe_end_hard_v9_checkpoint import (
    operational_metrics,
    select_checkpoint,
)


def _decision(action, *, finish_after_tool=False):
    if action == "end":
        return {"decision_type": "end", "end_reason": "memory_hit", "_planner_metrics": {}}
    return {
        "decision_type": "tool",
        "action": action,
        "action_input": {"finish_after_tool": finish_after_tool},
        "_planner_metrics": {"first_finish_reason": "stop"},
    }


def _fixture():
    cases = []
    sft = {}
    candidates = {10: {}, 20: {}, 40: {}}
    for entity in range(10):
        for primary, scenario in ((True, "current_success_step2"), (False, "control")):
            case_id = f"e{entity}-{scenario}"
            expected = [_decision("qwen_detection"), _decision("end")]
            cases.append(
                {
                    "case_id": case_id,
                    "entity_id": f"e{entity}",
                    "scenario_id": scenario,
                    "detector_family": "qwen",
                    "expected_decisions": expected,
                    "forbidden_actions": [],
                    "reward_spec": {},
                }
            )
            bad = [_decision("qwen_detection"), _decision("migration_advisor", finish_after_tool=True)]
            sft[case_id] = {"case_id": case_id, "decisions": bad if primary and entity < 5 else expected}
            for checkpoint in candidates:
                improve = checkpoint == 20 and primary and entity < 5
                regress = checkpoint == 40 and not primary and entity < 5
                candidates[checkpoint][case_id] = {
                    "case_id": case_id,
                    "decisions": bad if regress or (primary and entity < 5 and not improve) else expected,
                }
    return cases, sft, candidates


def _preregistration():
    return {
        "study_id": "test-v9",
        "design": {"primary_scenarios": ["current_success_step2"]},
        "selection_dev": {
            "models": ["SFT initializer", "GRPO checkpoint-10", "GRPO checkpoint-20", "GRPO checkpoint-40"],
            "promotion_gates": {
                "minimum_primary_complete_trajectory_gain_over_sft": 0.05,
                "maximum_control_complete_trajectory_regression": 0.02,
                "maximum_added_wrong_side_effecting_actions": 0,
                "minimum_json_valid_rate": 0.99,
                "maximum_clipped_rate": 0.01,
            },
            "on_fail": "do not open test",
        },
    }


def test_selection_promotes_only_checkpoint_meeting_primary_and_control_gates():
    cases, sft, predictions = _fixture()
    result = select_checkpoint(
        preregistration=_preregistration(),
        cases=cases,
        sft_predictions=sft,
        candidates=[
            (f"checkpoint-{checkpoint}", Path(f"{checkpoint}.jsonl"), rows)
            for checkpoint, rows in predictions.items()
        ],
        bootstrap_replicates=100,
        seed=42,
    )
    assert result["status"] == "promote"
    assert result["selected"]["checkpoint"] == 20
    assert result["larger_reference_used_for_selection"] is False


def test_operational_metrics_count_length_and_empty_failures():
    predictions = {
        "a": {"decisions": []},
        "b": {"decisions": [{"decision_type": "tool", "_planner_metrics": {"first_finish_reason": "length"}}]},
    }
    result = operational_metrics(predictions)
    assert result == {
        "planner_decisions": 2,
        "empty_cases": 1,
        "runtime_errors": 0,
        "json_valid_rate": 0.5,
        "clipped_rate": 0.5,
    }


def test_selection_rejects_planner_runtime_fallbacks():
    cases, sft, predictions = _fixture()
    predictions[10][cases[0]["case_id"]] = {
        "case_id": cases[0]["case_id"],
        "decisions": [
            {
                "decision_type": "tool",
                "action": "answerer",
                "action_input": {},
                "_planner_metrics": {
                    "error_type": "RuntimeError",
                    "error": "cuDNN error: CUDNN_STATUS_NOT_INITIALIZED",
                },
            }
        ],
    }
    with pytest.raises(ValueError, match="checkpoint-10.*runtime errors"):
        select_checkpoint(
            preregistration=_preregistration(),
            cases=cases,
            sft_predictions=sft,
            candidates=[
                (f"checkpoint-{checkpoint}", Path(f"{checkpoint}.jsonl"), rows)
                for checkpoint, rows in predictions.items()
            ],
            bootstrap_replicates=10,
            seed=42,
        )


def test_selection_rejects_incomplete_prediction_coverage():
    cases, sft, predictions = _fixture()
    predictions[20].pop(cases[0]["case_id"])
    with pytest.raises(ValueError, match="checkpoint-20 prediction coverage mismatch"):
        select_checkpoint(
            preregistration=_preregistration(),
            cases=cases,
            sft_predictions=sft,
            candidates=[
                (f"checkpoint-{checkpoint}", Path(f"{checkpoint}.jsonl"), rows)
                for checkpoint, rows in predictions.items()
            ],
            bootstrap_replicates=10,
            seed=42,
        )
