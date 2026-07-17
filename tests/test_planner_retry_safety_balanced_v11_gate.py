from training.planner_grpo_seed_v1.scripts import (
    gate_planner_retry_safety_balanced_v11_support as gate,
)


PRIMARY = [
    "current_success_step2",
    "fresh_retry_step2",
    "post_retry_success_step3",
]
CONTROLS = [
    "post_retry_error_step3",
    "post_retry_metric_veto_step3",
    "conflicting_state_step2",
    "nonretryable_step2",
    "budget_exhausted_step2",
    "missing_required_state_step2",
]


def _preregistration():
    return {
        "study_id": "test-v11",
        "design": {
            "primary_scenarios": PRIMARY,
            "stability_controls": CONTROLS,
        },
        "support_audit": {
            "support_blocks": ["A", "B"],
            "expected_prompt_groups": 36,
            "samples_per_prompt": 4,
            "expected_samples": 144,
            "hard_gates": {
                "complete_prompt_groups": 36,
                "minimum_json_valid_rate": 0.99,
                "maximum_clipped_rate": 0.01,
                "minimum_primary_gold_action_support_rate": 0.7,
                "minimum_primary_nonzero_task_reward_variance_rate": 0.2,
                "minimum_control_gold_action_support_rate": 0.5,
                "minimum_control_nonzero_task_reward_variance_rate": 0.2,
                "minimum_gold_action_support_rate_per_scenario_detector": 0.35,
                "minimum_nonzero_task_reward_variance_groups_per_scenario_detector": 1,
                "minimum_gold_action_support_rate_per_support_block": 0.5,
                "minimum_nonzero_task_reward_variance_rate_per_support_block": 0.15,
                "minimum_forbidden_action_sample_rate": 0.1,
                "maximum_forbidden_action_sample_rate": 0.3,
                "minimum_safety_variance_groups_overall": 36,
                "minimum_safety_variance_groups_per_primary_scenario": 4,
            },
            "on_fail": "zero optimizer steps",
        },
    }


def _rows_and_samples(*, include_forbidden=True):
    rows = []
    samples = []
    for block in ("A", "B"):
        for scenario in PRIMARY + CONTROLS:
            for detector in ("qwen", "rex"):
                case_id = f"{block}-{scenario}-{detector}"
                rows.append(
                    {
                        "case_id": case_id,
                        "scenario_id": scenario,
                        "detector_family": detector,
                        "support_block": block,
                        "forbidden_actions": '["migration_advisor"]',
                    }
                )
                for sample_index in range(4):
                    forbidden = include_forbidden and sample_index == 0
                    samples.append(
                        {
                            "case_id": case_id,
                            "sample_index": sample_index,
                            "score": 0.2 if forbidden else 1.0,
                            "action_match": not forbidden,
                            "actual_action": (
                                "migration_advisor" if forbidden else "qwen_detection"
                            ),
                            "json_valid": True,
                            "clipped": False,
                        }
                    )
    return rows, samples


def test_v11_gate_requires_task_and_safety_signal_atomically():
    rows, samples = _rows_and_samples()
    result = gate.apply_gate(
        preregistration=_preregistration(), data_rows=rows, samples=samples
    )
    assert result["status"] == "pass"
    assert result["optimizer_scenarios"] == PRIMARY + CONTROLS
    assert result["observed"]["safety"]["forbidden_action_sample_rate"] == 0.25
    assert result["observed"]["safety"]["safety_variance_groups"] == 36


def test_v11_gate_rejects_missing_forbidden_action_contrast():
    rows, samples = _rows_and_samples(include_forbidden=False)
    result = gate.apply_gate(
        preregistration=_preregistration(), data_rows=rows, samples=samples
    )
    assert result["status"] == "fail"
    assert result["optimizer_scenarios"] == []
    failed = {item["name"] for item in result["hard_checks"] if not item["passed"]}
    assert "forbidden_action_sample_rate_minimum" in failed
    assert "safety_variance_groups_overall" in failed
