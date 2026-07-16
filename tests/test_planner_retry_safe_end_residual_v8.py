from training.planner_grpo_seed_v1.scripts import (
    build_planner_retry_migrate_residual_v7 as v7,
)
from training.planner_grpo_seed_v1.scripts import (
    build_planner_retry_safe_end_residual_v8 as v8,
)


def test_v8_split_sizes_and_primary_counts():
    cases = v8.build_cases_in_memory()
    assert {split: len(rows) for split, rows in cases.items()} == {
        "grpo_train": 432,
        "support_dev": 216,
        "selection_dev": 216,
        "sealed_test_a": 216,
        "sealed_test_b": 216,
    }
    assert sum(row["scenario_id"] in v8.PRIMARY_SCENARIOS for row in cases["grpo_train"]) == 144
    assert sum(row["scenario_id"] in v8.PRIMARY_SCENARIOS for row in cases["support_dev"]) == 72


def test_v8_entities_and_protected_values_are_disjoint():
    cases = v8.build_cases_in_memory()
    split_entities = {
        split: {row["entity_id"] for row in rows} for split, rows in cases.items()
    }
    names = list(split_entities)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            assert split_entities[left].isdisjoint(split_entities[right])
    assert all(row["dataset_id"] == v8.DATASET_ID for rows in cases.values() for row in rows)
    assert all(row["case_id"].startswith("PRSV8-") for rows in cases.values() for row in rows)


def test_v8_every_detector_bundle_has_all_counterfactuals():
    cases = v8.build_cases_in_memory()
    for rows in cases.values():
        bundles = {}
        for row in rows:
            bundles.setdefault(row["counterfactual_bundle_id"], []).append(row)
        for bundle in bundles.values():
            assert {row["scenario_id"] for row in bundle} == set(v8.ALL_SCENARIOS)


def test_v8_builder_does_not_mutate_v7_module_contract():
    original = (v7.DATASET_ID, v7.SEED, tuple(v7.PRIMARY_SCENARIOS))
    v8.build_cases_in_memory()
    assert (v7.DATASET_ID, v7.SEED, tuple(v7.PRIMARY_SCENARIOS)) == original
