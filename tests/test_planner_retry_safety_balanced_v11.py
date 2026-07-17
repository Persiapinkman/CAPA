import json
from collections import Counter
from pathlib import Path

from training.planner_grpo_seed_v1.scripts import (
    build_planner_retry_anti_forgetting_v10 as v10,
)
from training.planner_grpo_seed_v1.scripts import (
    build_planner_retry_safe_end_hard_residual_v9 as v9,
)
from training.planner_grpo_seed_v1.scripts import (
    build_planner_retry_safety_balanced_v11 as v11,
)


def test_v11_train_replay_balances_non_migration_and_migration_actions():
    cases = v11.build_cases_in_memory()
    assert {split: len(rows) for split, rows in cases.items()} == {
        "grpo_train": 576,
        "support_dev_a": 216,
        "support_dev_b": 216,
        "selection_dev": 216,
        "sealed_test_a": 216,
        "sealed_test_b": 216,
    }
    train = cases["grpo_train"]
    variants = [row for row in train if row.get("primary_replay_variant") == "B"]
    assert len(variants) == 144
    assert Counter(row["scenario_id"] for row in variants) == Counter(
        {scenario: 48 for scenario in v9.PRIMARY_SCENARIOS}
    )
    assert Counter(row["target_action_class"] for row in train) == {
        "end": 192,
        "retry": 96,
        "migrate": 288,
    }
    assert len({row["case_id"] for row in train}) == len(train)


def test_v11_custom_validation_passes_and_prompts_are_distinct_by_query():
    with v11.configured_base():
        with v9.configured_v7():
            cases = v9.v7.build_all_cases()
            validation = v9.v7.validate_cases(cases)
    assert validation["status"] == "pass", validation["errors"]
    variants = [
        row
        for row in cases["grpo_train"]
        if row.get("primary_replay_variant") == "B"
    ]
    parents = {
        row["case_id"]: row
        for row in cases["grpo_train"]
        if row.get("primary_replay_variant") != "B"
    }
    for variant in variants:
        parent = parents[variant["case_id"].removesuffix("-PRB")]
        assert variant["user_query"] != parent["user_query"]
        assert variant["entity_id"] == parent["entity_id"]
        assert variant["expected_decisions"] == parent["expected_decisions"]


def test_v11_protected_values_are_disjoint_from_v10():
    current = v11.build_cases_in_memory()
    previous = v10.build_cases_in_memory()
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


def test_v11_preregisters_safety_weight_fresh_selection_and_same_gates():
    prereg = json.loads(
        Path(
            "experiments/studies/planner_retry_safety_balanced_v11_qwen35_4b_v1/"
            "preregistration.json"
        ).read_text(encoding="utf-8")
    )
    assert prereg["optimizer_dataset"]["expected_rows"] == 576
    assert prereg["training"]["reward_weights"] == {
        "task_reward": 0.75,
        "format_reward": 0.05,
        "no_forbidden_action": 0.2,
    }
    assert prereg["training"]["screen"]["candidate_checkpoints"] == [2, 5, 8]
    gates = prereg["selection_dev"]["promotion_gates"]
    assert gates["minimum_primary_complete_trajectory_gain_over_sft"] == 0.05
    assert gates["maximum_control_complete_trajectory_regression"] == 0.02
    assert gates["maximum_added_wrong_side_effecting_actions"] == 0


def test_v11_builder_restores_v9_module_settings():
    original = (v9.DATASET_ID, v9.SEED, v9.OPTIMIZER_SCENARIOS, v9.LEXICONS)
    v11.build_cases_in_memory()
    assert (v9.DATASET_ID, v9.SEED, v9.OPTIMIZER_SCENARIOS, v9.LEXICONS) == original
