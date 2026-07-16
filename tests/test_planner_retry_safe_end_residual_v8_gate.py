import json

from training.planner_grpo_seed_v1.scripts import (
    gate_planner_retry_safe_end_residual_v8_support as gate,
)


def _preregistration():
    return {
        "study_id": "test-v8",
        "design": {
            "primary_scenarios": [
                "current_success_step2",
                "fresh_retry_step2",
                "post_retry_success_step3",
            ]
        },
        "support_audit": {
            "expected_prompt_groups": 6,
            "samples_per_prompt": 4,
            "expected_samples": 24,
            "hard_gates": {
                "complete_prompt_groups": 6,
                "minimum_json_valid_rate": 0.99,
                "maximum_clipped_rate": 0.01,
                "minimum_primary_gold_action_support_rate": 0.7,
                "minimum_primary_nonzero_reward_variance_rate": 0.15,
                "minimum_gold_action_support_rate_per_scenario_detector": 0.5,
                "minimum_nonzero_reward_variance_groups_per_scenario_detector": 1,
            },
            "on_fail": "zero optimizer steps",
        },
    }


def _rows_and_samples(*, disable_rex_post_success_variance=False):
    rows = []
    samples = []
    scenarios = _preregistration()["design"]["primary_scenarios"]
    for scenario in scenarios:
        for detector in ("qwen", "rex"):
            case_id = f"{scenario}-{detector}"
            rows.append(
                {
                    "case_id": case_id,
                    "scenario_id": scenario,
                    "detector_family": detector,
                }
            )
            variable = not (
                disable_rex_post_success_variance
                and scenario == "post_retry_success_step3"
                and detector == "rex"
            )
            for sample_index in range(4):
                samples.append(
                    {
                        "case_id": case_id,
                        "sample_index": sample_index,
                        "score": 1.0 if not variable or sample_index else 0.2,
                        "action_match": sample_index > 0 or not variable,
                        "json_valid": True,
                        "clipped": False,
                    }
                )
    return rows, samples


def test_v8_support_gate_passes_only_when_every_stratum_has_variance():
    rows, samples = _rows_and_samples()
    result = gate.apply_gate(
        preregistration=_preregistration(), data_rows=rows, samples=samples
    )
    assert result["status"] == "pass"
    assert result["optimizer_scenarios"] == _preregistration()["design"]["primary_scenarios"]


def test_v8_support_gate_fails_one_detector_stratum_atomically():
    rows, samples = _rows_and_samples(disable_rex_post_success_variance=True)
    result = gate.apply_gate(
        preregistration=_preregistration(), data_rows=rows, samples=samples
    )
    assert result["status"] == "fail"
    assert result["optimizer_scenarios"] == []
    failed = {item["name"] for item in result["hard_checks"] if not item["passed"]}
    assert "post_retry_success_step3_rex_nonzero_variance_groups" in failed
