from __future__ import annotations

from collections import Counter, defaultdict

from training.planner_grpo_seed_v1.scripts import (
    build_planner_retry_migrate_residual_v7 as v7,
)


def test_v7_split_sizes_and_entity_isolation() -> None:
    cases = v7.build_all_cases()
    assert {split: len(rows) for split, rows in cases.items()} == {
        "grpo_train": 432,
        "grpo_dev": 216,
        "test": 216,
    }
    assert {
        split: len({row["entity_id"] for row in rows})
        for split, rows in cases.items()
    } == {"grpo_train": 24, "grpo_dev": 12, "test": 12}
    assert v7.validate_cases(cases)["status"] == "pass"


def test_each_entity_detector_bundle_has_all_residuals_and_controls() -> None:
    for split, rows in v7.build_all_cases().items():
        bundles: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            bundles[row["counterfactual_bundle_id"]].append(row)
        for bundle in bundles.values():
            assert {row["scenario_id"] for row in bundle} == set(v7.ALL_SCENARIOS)
            assert len(bundle) == 9
            assert len({row["user_query"] for row in bundle}) == 1
            assert len({row["badge_condition"] for row in bundle}) == 1
            assert len({row["image_fixture_family"] for row in bundle}) == 1
            assert len({row["bundle_error_alias"] for row in bundle}) == 1
            assert Counter(row["optimization_scope"] for row in bundle) == Counter(
                {"primary_residual": 6, "stability_control": 3}
            )


def test_factor_layout_is_balanced_by_construction() -> None:
    cases = v7.build_all_cases()
    train_entities = {
        row["entity_id"]: row for row in cases["grpo_train"]
    }
    train_combinations = Counter(
        (
            row["query_style_index"],
            row["badge_condition"],
            row["image_fixture_family"],
        )
        for row in train_entities.values()
    )
    assert len(train_combinations) == 24
    assert set(train_combinations.values()) == {1}

    for split in ("grpo_dev", "test"):
        entities = {row["entity_id"]: row for row in cases[split]}
        style_badge = Counter(
            (row["query_style_index"], row["badge_condition"])
            for row in entities.values()
        )
        fixtures = Counter(row["image_fixture_family"] for row in entities.values())
        assert len(style_badge) == 12
        assert set(style_badge.values()) == {1}
        assert sorted(fixtures.values()) == [6, 6]


def test_primary_targets_cover_step2_and_step3_boundaries() -> None:
    rows = v7.build_all_cases()["grpo_dev"]
    by_scenario = {row["scenario_id"]: row for row in rows}
    assert by_scenario["fresh_retry_step2"]["grpo_target_step"] == 2
    assert by_scenario["fresh_retry_step2"]["target_action_class"] == "retry"
    assert by_scenario["post_retry_success_step3"]["grpo_target_step"] == 3
    assert by_scenario["post_retry_success_step3"]["target_action_class"] == "end"
    assert by_scenario["post_retry_error_step3"]["target_action_class"] == "migrate"
    assert by_scenario["post_retry_metric_veto_step3"]["target_action_class"] == "migrate"
    assert by_scenario["current_success_step2"]["target_action_class"] == "end"
    assert by_scenario["conflicting_state_step2"]["target_action_class"] == "migrate"


def test_test_commitment_is_deterministic_without_materialization() -> None:
    first = v7.jsonl_text(v7.build_all_cases()["test"])
    second = v7.jsonl_text(v7.build_all_cases()["test"])
    assert v7.sha256_text(first) == v7.sha256_text(second)
