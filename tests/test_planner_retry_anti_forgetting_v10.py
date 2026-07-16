import json
from pathlib import Path

from training.planner_grpo_seed_v1.scripts import (
    build_planner_retry_anti_forgetting_v10 as v10,
)
from training.planner_grpo_seed_v1.scripts import (
    build_planner_retry_safe_end_hard_residual_v9 as v9,
)


def test_v10_split_sizes_and_all_scenario_optimizer_replay():
    cases = v10.build_cases_in_memory()
    assert {split: len(rows) for split, rows in cases.items()} == {
        "grpo_train": 432,
        "support_dev_a": 216,
        "support_dev_b": 216,
        "selection_dev": 216,
        "sealed_test_a": 216,
        "sealed_test_b": 216,
    }
    assert all(row["grpo_eligible"] for row in cases["grpo_train"])
    assert {
        row["scenario_id"] for row in cases["grpo_train"]
    } == set(v9.ALL_SCENARIOS)


def test_v10_entities_and_protected_values_are_disjoint_from_v9():
    v10_cases = v10.build_cases_in_memory()
    v9_cases = v9.build_cases_in_memory()
    v10_entities = {
        split: {row["entity_id"] for row in rows}
        for split, rows in v10_cases.items()
    }
    names = list(v10_entities)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            assert v10_entities[left].isdisjoint(v10_entities[right])
    assert not set().union(*v10_entities.values()) & {
        row["entity_id"] for rows in v9_cases.values() for row in rows
    }


def test_v10_counterfactual_bundles_and_provenance():
    cases = v10.build_cases_in_memory()
    for rows in cases.values():
        bundles = {}
        for row in rows:
            bundles.setdefault(row["counterfactual_bundle_id"], []).append(row)
            assert row["case_id"].startswith("PRAFV10-")
            assert row["difficulty_family"] == (
                "v7_anchored_hard_nuisance_control_replay"
            )
        assert all(
            {row["scenario_id"] for row in bundle} == set(v9.ALL_SCENARIOS)
            for bundle in bundles.values()
        )


def test_v10_preregisters_early_checkpoints_and_unchanged_promotion_gates():
    path = Path(
        "experiments/studies/planner_retry_anti_forgetting_v10_qwen35_4b_v1/"
        "preregistration.json"
    )
    prereg = json.loads(path.read_text(encoding="utf-8"))
    assert prereg["training"]["screen"]["candidate_checkpoints"] == [2, 5, 10]
    assert prereg["optimizer_dataset"]["expected_rows"] == 432
    assert set(prereg["optimizer_dataset"]["scenario_distribution"]) == set(
        v9.ALL_SCENARIOS
    )
    gates = prereg["selection_dev"]["promotion_gates"]
    assert gates["minimum_primary_complete_trajectory_gain_over_sft"] == 0.05
    assert gates["maximum_control_complete_trajectory_regression"] == 0.02


def test_v10_builder_restores_v9_module_settings():
    original = (v9.DATASET_ID, v9.SEED, v9.OPTIMIZER_SCENARIOS, v9.LEXICONS)
    v10.build_cases_in_memory()
    assert (v9.DATASET_ID, v9.SEED, v9.OPTIMIZER_SCENARIOS, v9.LEXICONS) == original
