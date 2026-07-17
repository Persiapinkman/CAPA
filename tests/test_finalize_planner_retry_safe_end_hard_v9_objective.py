import pytest

from training.planner_grpo_seed_v1.scripts.finalize_planner_retry_safe_end_hard_v9_objective import (
    finalize_objective,
)


def _decision(action):
    if action == "end":
        return {"decision_type": "end", "end_reason": "memory_hit"}
    return {"decision_type": "tool", "action": action, "action_input": {"finish_after_tool": False}}


def test_finalize_objective_requires_both_wins_and_control_guardrail():
    cases = []
    sft = {}
    grpo = {}
    larger = {}
    for entity in range(20):
        for scenario in ("primary", "control"):
            cid = f"{entity}-{scenario}"
            expected = [_decision("qwen_detection"), _decision("end")]
            wrong = [_decision("qwen_detection"), _decision("qwen_detection")]
            cases.append(
                {
                    "case_id": cid,
                    "entity_id": str(entity),
                    "scenario_id": scenario,
                    "detector_family": "qwen",
                    "expected_decisions": expected,
                    "forbidden_actions": [],
                    "reward_spec": {},
                }
            )
            sft[cid] = {"case_id": cid, "decisions": wrong if scenario == "primary" and entity < 10 else expected}
            larger[cid] = {"case_id": cid, "decisions": wrong if scenario == "primary" and entity < 15 else expected}
            grpo[cid] = {"case_id": cid, "decisions": wrong if scenario == "primary" and entity < 2 else expected}
    result = finalize_objective(
        contract={"study_id": "test", "units_and_metrics": {"primary_scope": ["primary"]}},
        cases=cases,
        sft=sft,
        grpo=grpo,
        larger=larger,
        bootstrap_replicates=1000,
        seed=42,
    )
    assert result["objective_met"] is True
    assert result["checks"] == {
        "grpo_above_sft_primary": True,
        "grpo_above_larger_primary": True,
        "control_guardrail": True,
        "wrong_side_effecting_action_guardrail": True,
    }


def test_finalize_objective_rejects_larger_reference_runtime_fallback():
    cases = [
        {
            "case_id": "case-1",
            "entity_id": "entity-1",
            "scenario_id": "primary",
            "detector_family": "qwen",
            "expected_decisions": [_decision("qwen_detection"), _decision("end")],
            "forbidden_actions": [],
            "reward_spec": {},
        }
    ]
    valid = {
        "case-1": {
            "case_id": "case-1",
            "decisions": [_decision("qwen_detection"), _decision("end")],
        }
    }
    larger = {
        "case-1": {
            "case_id": "case-1",
            "decisions": [
                {
                    "decision_type": "tool",
                    "action": "answerer",
                    "action_input": {},
                    "_planner_metrics": {
                        "error_type": "RuntimeError",
                        "error": "gateway unavailable",
                    },
                }
            ],
        }
    }
    with pytest.raises(ValueError, match="larger.*runtime errors"):
        finalize_objective(
            contract={
                "study_id": "test",
                "units_and_metrics": {"primary_scope": ["primary"]},
            },
            cases=cases,
            sft=valid,
            grpo=valid,
            larger=larger,
            bootstrap_replicates=10,
            seed=42,
        )
