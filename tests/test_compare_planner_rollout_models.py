from training.planner_grpo_seed_v1.scripts.compare_planner_rollout_models import (
    compare_rollouts,
)


def _case(case_id, entity_id, scenario_id):
    return {
        "case_id": case_id,
        "entity_id": entity_id,
        "scenario_id": scenario_id,
        "category": scenario_id,
        "detector_family": "qwen",
        "expected_decisions": [
            {
                "decision_type": "end",
                "required_args": {"end_reason": "memory_hit"},
            }
        ],
        "forbidden_actions": ["migration_advisor"],
        "reward_spec": {},
    }


def _prediction(case_id, *, correct):
    decision = (
        {
            "decision_type": "end",
            "end_reason": "memory_hit",
            "final_answer": "",
        }
        if correct
        else {
            "decision_type": "tool",
            "action": "migration_advisor",
            "action_input": {},
        }
    )
    return {case_id: {"case_id": case_id, "decisions": [decision]}}


def test_compare_rollouts_uses_entity_clustered_primary_delta():
    cases = [
        _case("a-primary", "a", "primary"),
        _case("a-control", "a", "control"),
        _case("b-primary", "b", "primary"),
        _case("b-control", "b", "control"),
    ]
    candidate = {}
    reference = {}
    for case in cases:
        candidate.update(_prediction(case["case_id"], correct=True))
        reference.update(
            _prediction(case["case_id"], correct=case["scenario_id"] == "control")
        )
    result = compare_rollouts(
        cases=cases,
        candidate_predictions=candidate,
        reference_predictions=reference,
        primary_scenarios={"primary"},
        candidate_label="small",
        reference_label="large",
        bootstrap_replicates=100,
        seed=42,
    )
    assert result["primary"]["pass_rate_delta"] == 1.0
    assert result["controls"]["pass_rate_delta"] == 0.0
    assert result["entity_clustered_primary_delta_bootstrap"]["point_estimate"] == 1.0
    assert result["primary"]["paired_case_counts"]["candidate_only"] == 2
