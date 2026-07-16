from __future__ import annotations

import unittest
from collections import Counter

from training.planner_grpo_seed_v1.scripts import (
    build_planner_multistep_grpo_value_v4 as v4,
)
from training.planner_grpo_seed_v1.scripts.reward_planner_grpo import score_case


class PlannerMultistepGrpoValueV4Tests(unittest.TestCase):
    def test_calibration_is_blocked_and_action_balanced(self) -> None:
        rows = v4.build_split(
            split="calibration", accepted_scenarios=[], confirmation_cases=240
        )
        self.assertEqual(len(rows), 240)
        self.assertEqual(len({row["entity_id"] for row in rows}), 20)
        self.assertEqual({row["scenario_id"] for row in rows}, set(v4.SCENARIOS))
        self.assertFalse(v4.validate_rows(rows, split="calibration"))
        for scenario in v4.SCENARIOS:
            selected = [row for row in rows if row["scenario_id"] == scenario]
            self.assertEqual(
                Counter(row["target_action_class"] for row in selected),
                Counter({"migrate": 12, "retry": 6, "end": 2}),
            )

    def test_policy_query_is_identical_within_detector_entity_block(self) -> None:
        rows = v4.build_split(
            split="calibration", accepted_scenarios=[], confirmation_cases=240
        )
        for entity_id in {row["entity_id"] for row in rows}:
            block = [row for row in rows if row["entity_id"] == entity_id]
            for detector in ("qwen", "rex"):
                queries = {
                    row["user_query"]
                    for row in block
                    if row["detector_family"] == detector
                }
                self.assertEqual(len(queries), 1)

    def test_current_observation_contains_no_action_hint(self) -> None:
        rows = v4.build_split(
            split="calibration", accepted_scenarios=[], confirmation_cases=240
        )
        for row in rows:
            summary = row["mock_observations"][0]["observation"]["summary"]
            for hint in v4.CURRENT_OBSERVATION_FORBIDDEN_HINTS:
                self.assertNotIn(hint, summary)

    def test_canonical_gold_passes_strict_reward(self) -> None:
        rows = v4.build_split(
            split="calibration", accepted_scenarios=[], confirmation_cases=240
        )
        self.assertTrue(all(score_case(row)["passed"] for row in rows))

    def test_confirmation_is_frozen_balanced_and_split_isolated(self) -> None:
        accepted = list(v4.SCENARIOS[:8])
        calibration = v4.build_split(
            split="calibration", accepted_scenarios=[], confirmation_cases=240
        )
        confirmation = v4.build_split(
            split="confirmation",
            accepted_scenarios=accepted,
            confirmation_cases=v4.DEFAULT_CONFIRMATION_CASES,
        )
        self.assertEqual(len(confirmation), 240)
        self.assertEqual(len({row["entity_id"] for row in confirmation}), 30)
        self.assertEqual(
            Counter(row["target_action_class"] for row in confirmation),
            Counter({"migrate": 144, "retry": 72, "end": 24}),
        )
        self.assertFalse(v4.validate_rows(confirmation, split="confirmation"))
        self.assertEqual(
            v4.v1.split_integrity(
                {"calibration": calibration, "confirmation": confirmation}
            )["status"],
            "pass",
        )

    def test_calibration_and_confirmation_use_distinct_aliases(self) -> None:
        for pack in v4.PACKS:
            calibration = v4.PACK_ERROR_ALIASES[pack]["calibration"]
            confirmation = v4.PACK_ERROR_ALIASES[pack]["confirmation"]
            self.assertFalse(
                set(calibration["migrate"]) & set(confirmation["migrate"])
            )
            self.assertFalse(set(calibration["retry"]) & set(confirmation["retry"]))


if __name__ == "__main__":
    unittest.main()
