from __future__ import annotations

import unittest
from collections import Counter

from training.planner_grpo_seed_v1.scripts import (
    build_planner_multistep_grpo_value_v5_train_v1 as train_v1,
)
from training.planner_grpo_seed_v1.scripts.reward_planner_grpo import score_case


class PlannerMultistepGrpoValueV5TrainV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        train_v1.write_fixture_images()
        cls.rows = train_v1.build_cases()

    def test_size_and_v5_distribution_match(self) -> None:
        self.assertEqual(len(self.rows), 480)
        self.assertEqual(len({row["entity_id"] for row in self.rows}), 60)
        report = train_v1.distribution_report(self.rows)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["case_scale_factor"], 2.0)
        self.assertEqual(report["total_variation_distance"]["families"], 0.0)
        self.assertEqual(
            report["total_variation_distance"]["target_action_classes"], 0.0
        )
        self.assertEqual(train_v1.text_shape_diagnostic(self.rows)["status"], "pass")

    def test_each_family_has_balanced_retry_anchor(self) -> None:
        for scenario in train_v1.SUPPORTED_SCENARIOS:
            selected = [row for row in self.rows if row["scenario_id"] == scenario]
            self.assertEqual(len(selected), 60)
            self.assertEqual(
                Counter(row["target_action_class"] for row in selected),
                Counter({"migrate": 36, "retry": 24}),
            )
            aliases = train_v1.PACK_SPECS[scenario.split("_", 1)[1]]["aliases"]
            for alias in aliases:
                alias_rows = [row for row in selected if row["error_alias"] == alias]
                self.assertEqual(
                    Counter(row["target_action_class"] for row in alias_rows),
                    Counter({"migrate": 18, "retry": 12}),
                )

    def test_entity_blocks_and_queries_are_matched(self) -> None:
        for entity_id in {row["entity_id"] for row in self.rows}:
            block = [row for row in self.rows if row["entity_id"] == entity_id]
            self.assertEqual(len(block), 8)
            self.assertEqual(len({row["target_action_class"] for row in block}), 1)
            for detector in ("qwen", "rex"):
                self.assertEqual(
                    len(
                        {
                            row["user_query"]
                            for row in block
                            if row["detector_family"] == detector
                        }
                    ),
                    1,
                )

    def test_v5_entities_aliases_queries_templates_and_fixtures_are_disjoint(self) -> None:
        report = train_v1.isolation_report(self.rows)
        self.assertEqual(report["status"], "pass")
        v5 = report["v5_evaluation"]
        for key in train_v1.PROTECTED_OVERLAP_KEYS:
            self.assertEqual(v5[f"{key}_overlap"], 0, key)
        self.assertEqual(v5["shared_taxonomy"]["scenario_id_overlap"], 8)

    def test_training_flags_and_strict_gold(self) -> None:
        self.assertFalse(train_v1.validate_rows(self.rows))
        for row in self.rows:
            self.assertTrue(row["training_only"])
            self.assertFalse(row["evaluation_only"])
            self.assertFalse(row["exclude_from_training"])
            self.assertEqual(row["grpo_target_step"], 2)
            self.assertTrue(score_case(row)["passed"])

    def test_badge_is_not_an_action_shortcut(self) -> None:
        contingency: dict[str, set[str]] = {"migrate": set(), "retry": set()}
        for row in self.rows:
            summary = row["mock_observations"][0]["observation"]["summary"]
            for badge in ("amber", "red"):
                if f"overall_badge={badge}" in summary:
                    contingency[row["target_action_class"]].add(badge)
        self.assertEqual(contingency, {"migrate": {"amber", "red"}, "retry": {"amber", "red"}})


if __name__ == "__main__":
    unittest.main()
