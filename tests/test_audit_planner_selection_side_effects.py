import pytest

from training.planner_grpo_seed_v1.scripts.audit_planner_selection_side_effects import (
    audit_side_effects,
)


def _prediction(case_id, action):
    return {
        "case_id": case_id,
        "decisions": [
            {
                "decision_type": "tool",
                "action": action,
                "action_input": {"finish_after_tool": False},
            }
        ],
    }


def test_side_effect_audit_counts_introduced_removed_and_net():
    cases = [
        {
            "case_id": "a",
            "entity_id": "e1",
            "scenario_id": "primary",
            "detector_family": "qwen",
            "expected_decisions": [_prediction("a", "qwen_detection")["decisions"][0]],
            "forbidden_actions": ["migration_advisor"],
            "reward_spec": {},
        },
        {
            "case_id": "b",
            "entity_id": "e2",
            "scenario_id": "control",
            "detector_family": "rex",
            "expected_decisions": [_prediction("b", "rexomni_detection")["decisions"][0]],
            "forbidden_actions": ["migration_advisor"],
            "reward_spec": {},
        },
    ]
    reference = {
        "a": _prediction("a", "qwen_detection"),
        "b": _prediction("b", "migration_advisor"),
    }
    candidate = {
        "a": _prediction("a", "migration_advisor"),
        "b": _prediction("b", "rexomni_detection"),
    }
    result = audit_side_effects(
        cases=cases,
        candidate=candidate,
        reference=reference,
        candidate_label="grpo",
        reference_label="sft",
    )
    assert result["changed_cases"] == 2
    assert result["totals"] == {
        "grpo": 1,
        "sft": 1,
        "introduced": 1,
        "removed": 1,
        "net_added": 0,
    }
    assert result["by_scenario"]["primary"]["introduced"] == 1
    assert result["by_scenario"]["control"]["removed"] == 1


def test_side_effect_audit_requires_complete_coverage():
    with pytest.raises(ValueError, match="candidate prediction coverage mismatch"):
        audit_side_effects(
            cases=[{"case_id": "a"}],
            candidate={},
            reference={"a": _prediction("a", "qwen_detection")},
            candidate_label="candidate",
            reference_label="reference",
        )
