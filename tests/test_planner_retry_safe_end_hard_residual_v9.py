from collections import Counter

from training.planner_grpo_seed_v1.scripts import (
    build_planner_retry_migrate_residual_v7 as v7,
)
from training.planner_grpo_seed_v1.scripts import (
    build_planner_retry_safe_end_hard_residual_v9 as v9,
)


def test_v9_split_sizes_and_primary_counts():
    cases = v9.build_cases_in_memory()
    assert {split: len(rows) for split, rows in cases.items()} == {
        "grpo_train": 432,
        "support_dev_a": 216,
        "support_dev_b": 216,
        "selection_dev": 216,
        "sealed_test_a": 216,
        "sealed_test_b": 216,
    }
    assert sum(row["scenario_id"] in v9.PRIMARY_SCENARIOS for row in cases["grpo_train"]) == 144
    assert sum(
        row["scenario_id"] in v9.PRIMARY_SCENARIOS
        for split in ("support_dev_a", "support_dev_b")
        for row in cases[split]
    ) == 144


def test_v9_support_blocks_are_disjoint_and_factor_balanced():
    cases = v9.build_cases_in_memory()
    entities = {
        split: {row["entity_id"] for row in rows} for split, rows in cases.items()
    }
    names = list(entities)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            assert entities[left].isdisjoint(entities[right])
    for split, block in (("support_dev_a", "A"), ("support_dev_b", "B")):
        entity_rows = {row["entity_id"]: row for row in cases[split]}
        assert {row["support_block"] for row in entity_rows.values()} == {block}
        style_badge = Counter(
            (row["query_style_index"], row["badge_condition"])
            for row in entity_rows.values()
        )
        assert len(style_badge) == 12
        assert set(style_badge.values()) == {1}


def test_v9_every_bundle_has_all_counterfactuals_and_hard_provenance():
    cases = v9.build_cases_in_memory()
    for rows in cases.values():
        bundles = {}
        for row in rows:
            bundles.setdefault(row["counterfactual_bundle_id"], []).append(row)
            assert row["case_id"].startswith("PRHV9-")
            assert row["difficulty_family"] == "v7_anchored_hard_nuisance"
        assert all({row["scenario_id"] for row in bundle} == set(v9.ALL_SCENARIOS) for bundle in bundles.values())


def test_v9_builder_does_not_mutate_v7_module_contract():
    original = (v7.DATASET_ID, v7.SEED, tuple(v7.PRIMARY_SCENARIOS))
    v9.build_cases_in_memory()
    assert (v7.DATASET_ID, v7.SEED, tuple(v7.PRIMARY_SCENARIOS)) == original
