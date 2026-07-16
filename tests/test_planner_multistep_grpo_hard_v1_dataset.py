from __future__ import annotations

import unittest
from collections import Counter

from training.planner_grpo_seed_v1.scripts.build_planner_multistep_grpo_hard_v1 import (
    SCENARIOS,
    STRICT_REWARD,
    build_split,
    split_integrity,
    validate_rows,
)
from training.planner_grpo_seed_v1.scripts import build_planner_multistep_grpo_hard_v2 as v2


class PlannerMultistepGrpoHardV1DatasetTests(unittest.TestCase):
    def test_calibration_is_24_complete_counterfactual_bundles(self) -> None:
        rows = build_split(
            split="calibration",
            accepted_scenarios=[],
            confirmation_cases=600,
        )
        self.assertEqual(len(rows), 288)
        self.assertEqual(len({row["entity_id"] for row in rows}), 24)
        self.assertEqual(Counter(row["scenario_id"] for row in rows), Counter({s: 24 for s in SCENARIOS}))
        self.assertFalse(validate_rows(rows))

        by_entity: dict[str, list[dict]] = {}
        for row in rows:
            by_entity.setdefault(row["entity_id"], []).append(row)
        for bundle in by_entity.values():
            qwen = [row for row in bundle if row["scenario_id"].startswith("qwen_")]
            rex = [row for row in bundle if row["scenario_id"].startswith("rex_")]
            self.assertEqual(len({row["user_query"] for row in qwen}), 1)
            self.assertEqual(len({row["user_query"] for row in rex}), 1)
            self.assertEqual({row["mock_observations"][0]["observation"]["status"] for row in qwen}, {"uncertain", "confident", "retryable_failure"})
            self.assertEqual({row["mock_observations"][0]["observation"]["status"] for row in rex}, {"uncertain", "confident", "retryable_failure"})

    def test_confirmation_is_family_balanced_and_not_case_filtered(self) -> None:
        admitted = list(SCENARIOS[:8])
        rows = build_split(
            split="confirmation",
            accepted_scenarios=admitted,
            confirmation_cases=600,
        )
        counts = Counter(row["scenario_id"] for row in rows)
        self.assertEqual(len(rows), 600)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertEqual(set(counts), set(admitted))
        self.assertTrue(all(row["selection_role"] == "frozen_family_confirmation_unfiltered" for row in rows))
        self.assertFalse(validate_rows(rows))

    def test_split_resources_and_templates_are_isolated(self) -> None:
        calibration = build_split(
            split="calibration",
            accepted_scenarios=[],
            confirmation_cases=600,
        )
        confirmation = build_split(
            split="confirmation",
            accepted_scenarios=list(SCENARIOS[:8]),
            confirmation_cases=600,
        )
        integrity = split_integrity(
            {"calibration": calibration, "confirmation": confirmation}
        )
        self.assertEqual(integrity["status"], "pass")

    def test_reward_contract_is_strict_and_grpo_suitable(self) -> None:
        self.assertTrue(STRICT_REWARD["strict_action_match"])
        self.assertTrue(STRICT_REWARD["strict_argument_types"])
        self.assertEqual(STRICT_REWARD["wrong_action_cap"], 0.20)
        self.assertGreater(STRICT_REWARD["no_premature_stop"], 0)
        self.assertGreater(STRICT_REWARD["no_skip_required_probe"], 0)
        self.assertGreater(STRICT_REWARD["final_tool_finish"], 0)


class PlannerMultistepGrpoHardV2DatasetTests(unittest.TestCase):
    def test_v2_calibration_has_new_entities_and_observation_contrasts(self) -> None:
        v1_rows = build_split(
            split="calibration",
            accepted_scenarios=[],
            confirmation_cases=600,
        )
        rows = v2.build_split(
            split="calibration",
            accepted_scenarios=[],
            confirmation_cases=600,
        )
        self.assertEqual(len(rows), 384)
        self.assertEqual(len({row["entity_id"] for row in rows}), 32)
        self.assertFalse(validate_rows(rows))
        self.assertFalse(
            {row["user_query"] for row in v1_rows}
            & {row["user_query"] for row in rows}
        )
        self.assertFalse(
            {row["entity_id"] for row in v1_rows}
            & {row["entity_id"] for row in rows}
        )
        qwen = [row for row in rows if row["entity_id"].endswith("001") and row["scenario_id"].startswith("qwen_")]
        self.assertEqual(len(qwen), 4)
        self.assertEqual(len({row["user_query"] for row in qwen}), 1)
        self.assertEqual(
            {row["mock_observations"][0]["observation"]["status"] for row in qwen},
            {"box_variance", "empty_result", "domain_shift", "confident"},
        )

    def test_v2_confirmation_is_exactly_600_primary_cases(self) -> None:
        admitted = [scenario for scenario in v2.SCENARIOS if scenario in v2.PRIMARY_SCENARIOS]
        rows = v2.build_split(
            split="confirmation",
            accepted_scenarios=admitted,
            confirmation_cases=600,
        )
        self.assertEqual(len(rows), 600)
        self.assertEqual({row["scenario_id"] for row in rows}, set(admitted))
        self.assertTrue(all(row["scenario_tier"] == "primary_challenge" for row in rows))
        self.assertFalse(validate_rows(rows))


if __name__ == "__main__":
    unittest.main()
