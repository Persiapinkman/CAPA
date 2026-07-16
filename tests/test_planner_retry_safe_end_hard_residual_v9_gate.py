from training.planner_grpo_seed_v1.scripts import (
    gate_planner_retry_safe_end_hard_residual_v9_support as gate,
)


def _preregistration():
    return {
        "study_id": "test-v9",
        "design": {
            "primary_scenarios": [
                "current_success_step2",
                "fresh_retry_step2",
                "post_retry_success_step3",
            ]
        },
        "support_audit": {
            "support_blocks": ["A", "B"],
            "expected_prompt_groups": 12,
            "samples_per_prompt": 4,
            "expected_samples": 48,
            "hard_gates": {
                "complete_prompt_groups": 12,
                "minimum_json_valid_rate": 0.99,
                "maximum_clipped_rate": 0.01,
                "minimum_primary_gold_action_support_rate": 0.7,
                "minimum_primary_nonzero_reward_variance_rate": 0.2,
                "minimum_gold_action_support_rate_per_scenario_detector": 0.5,
                "minimum_nonzero_reward_variance_groups_per_scenario_detector": 1,
                "minimum_gold_action_support_rate_per_support_block": 0.6,
                "minimum_nonzero_reward_variance_rate_per_support_block": 0.15,
            },
            "on_fail": "zero optimizer steps",
        },
    }


def _rows_and_samples(*, saturate_block_b=False):
    rows = []
    samples = []
    for block in ("A", "B"):
        for scenario in _preregistration()["design"]["primary_scenarios"]:
            for detector in ("qwen", "rex"):
                case_id = f"{block}-{scenario}-{detector}"
                rows.append(
                    {
                        "case_id": case_id,
                        "scenario_id": scenario,
                        "detector_family": detector,
                        "support_block": block,
                    }
                )
                for sample_index in range(4):
                    variable = not (saturate_block_b and block == "B")
                    samples.append(
                        {
                            "case_id": case_id,
                            "sample_index": sample_index,
                            "score": 1.0 if sample_index == 0 or not variable else 0.2,
                            "action_match": True,
                            "json_valid": True,
                            "clipped": False,
                        }
                    )
    return rows, samples


def test_v9_support_gate_passes_both_blocks_atomically():
    rows, samples = _rows_and_samples()
    result = gate.apply_gate(
        preregistration=_preregistration(), data_rows=rows, samples=samples
    )
    assert result["status"] == "pass"
    assert set(result["observed"]["by_support_block"]) == {"A", "B"}


def test_v9_support_gate_rejects_one_saturated_block():
    rows, samples = _rows_and_samples(saturate_block_b=True)
    result = gate.apply_gate(
        preregistration=_preregistration(), data_rows=rows, samples=samples
    )
    assert result["status"] == "fail"
    assert result["optimizer_scenarios"] == []
    failed = {item["name"] for item in result["hard_checks"] if not item["passed"]}
    assert "support_block_B_nonzero_variance_rate" in failed
