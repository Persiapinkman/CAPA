import json
from collections import Counter
from pathlib import Path

from training.planner_grpo_seed_v1.scripts import (
    build_planner_retry_optimizer_matched_v12 as v12,
)
from training.planner_grpo_seed_v1.scripts import (
    build_planner_retry_safe_end_hard_residual_v9 as v9,
)
from training.planner_grpo_seed_v1.scripts import (
    build_planner_retry_safety_balanced_v11 as v11,
)


def test_v12_support_and_optimizer_have_the_same_action_distribution():
    cases = v12.build_cases_in_memory()
    assert {split: len(rows) for split, rows in cases.items()} == {
        "grpo_train": 576,
        "support_dev_a": 288,
        "support_dev_b": 288,
        "selection_dev": 216,
        "sealed_test_a": 216,
        "sealed_test_b": 216,
    }
    expected_actions = Counter({"end": 192, "retry": 96, "migrate": 288})
    assert Counter(row["target_action_class"] for row in cases["grpo_train"]) == expected_actions
    support = cases["support_dev_a"] + cases["support_dev_b"]
    assert Counter(row["target_action_class"] for row in support) == expected_actions
    assert sum(row.get("primary_replay_variant") == "B" for row in support) == 144


def test_v12_custom_validation_passes_and_replay_prompts_are_independent():
    with v12.configured_base():
        with v9.configured_v7():
            cases = v9.v7.build_all_cases()
            validation = v9.v7.validate_cases(cases)
    assert validation["status"] == "pass", validation["errors"]
    for split in v12.REPLAY_SPLITS:
        parents = {
            row["case_id"]: row
            for row in cases[split]
            if row.get("primary_replay_variant") != "B"
        }
        variants = [
            row for row in cases[split] if row.get("primary_replay_variant") == "B"
        ]
        for variant in variants:
            parent = parents[variant["case_id"].removesuffix("-PRB")]
            assert variant["user_query"] != parent["user_query"]
            assert variant["entity_id"] == parent["entity_id"]
            assert variant["expected_decisions"] == parent["expected_decisions"]


def test_v12_protected_values_are_disjoint_from_v11():
    current = v12.build_cases_in_memory()
    previous = v11.build_cases_in_memory()
    current_values = v9.v7.protected_values(
        [row for rows in current.values() for row in rows]
    )
    previous_values = v9.v7.protected_values(
        [row for rows in previous.values() for row in rows]
    )
    assert {
        field: current_values[field] & previous_values[field]
        for field in current_values
    } == {field: set() for field in current_values}


def test_v12_preregistration_keeps_the_v11_count_gate_and_matches_support():
    prereg = json.loads(
        Path(
            "experiments/studies/planner_retry_optimizer_matched_v12_qwen35_4b_v1/"
            "preregistration.json"
        ).read_text(encoding="utf-8")
    )
    support = prereg["support_audit"]
    assert support["expected_prompt_groups"] == 576
    assert support["expected_primary_groups"] == 288
    assert support["expected_control_groups"] == 288
    assert support["hard_gates"]["minimum_safety_variance_groups_overall"] == 43
    assert support["hard_gates"]["minimum_primary_safety_variance_rate"] == 0.15
    assert prereg["training"]["reward_weights"]["no_forbidden_action"] == 0.2


def test_v12_builder_restores_v9_module_settings():
    original = (v9.DATASET_ID, v9.SEED, v9.OPTIMIZER_SCENARIOS, v9.LEXICONS)
    v12.build_cases_in_memory()
    assert (v9.DATASET_ID, v9.SEED, v9.OPTIMIZER_SCENARIOS, v9.LEXICONS) == original
