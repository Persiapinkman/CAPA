from __future__ import annotations

import unittest
from collections import Counter

from training.planner_grpo_seed_v1.scripts import (
    build_planner_multistep_grpo_value_v5 as v5,
)
from training.planner_grpo_seed_v1.scripts.reward_planner_grpo import score_case


class PlannerMultistepGrpoValueV5Tests(unittest.TestCase):
    def test_calibration_is_blocked_and_balanced(self) -> None:
        rows = v5.build_split(
            split="calibration", accepted_scenarios=[], confirmation_cases=240
        )
        self.assertEqual(len(rows), 240)
        self.assertEqual(len({row["entity_id"] for row in rows}), 20)
        self.assertFalse(v5.validate_rows(rows, split="calibration"))
        for scenario in v5.SCENARIOS:
            selected = [row for row in rows if row["scenario_id"] == scenario]
            self.assertEqual(
                Counter(row["target_action_class"] for row in selected),
                Counter({"migrate": 12, "retry": 8}),
            )

    def test_matched_alias_changes_only_retry_state_within_entity_pack(self) -> None:
        rows = v5.build_split(
            split="calibration", accepted_scenarios=[], confirmation_cases=240
        )
        for row in rows:
            summary = row["mock_observations"][0]["observation"]["summary"]
            pack = row["state_contract_pack"]
            entity_index = int(row["case_id"].split("-")[2]) - 1
            aliases = v5.PACK_SPECS[pack]["calibration"]
            self.assertIn(f"gateway_error={aliases[entity_index % len(aliases)]}", summary)

    def test_current_observation_has_no_action_hint(self) -> None:
        rows = v5.build_split(
            split="calibration", accepted_scenarios=[], confirmation_cases=240
        )
        for row in rows:
            summary = row["mock_observations"][0]["observation"]["summary"]
            for hint in v5.CURRENT_OBSERVATION_FORBIDDEN_HINTS:
                self.assertNotIn(hint, summary)

    def test_canonical_gold_passes(self) -> None:
        rows = v5.build_split(
            split="calibration", accepted_scenarios=[], confirmation_cases=240
        )
        self.assertTrue(all(score_case(row)["passed"] for row in rows))

    def test_confirmation_is_frozen_and_split_isolated(self) -> None:
        accepted = list(v5.SCENARIOS[:8])
        calibration = v5.build_split(
            split="calibration", accepted_scenarios=[], confirmation_cases=240
        )
        confirmation = v5.build_split(
            split="confirmation",
            accepted_scenarios=accepted,
            confirmation_cases=v5.DEFAULT_CONFIRMATION_CASES,
        )
        self.assertEqual(len(confirmation), 240)
        self.assertEqual(
            Counter(row["target_action_class"] for row in confirmation),
            Counter({"migrate": 144, "retry": 96}),
        )
        self.assertFalse(v5.validate_rows(confirmation, split="confirmation"))
        self.assertEqual(
            v5.v1.split_integrity(
                {"calibration": calibration, "confirmation": confirmation}
            )["status"],
            "pass",
        )

    def test_confirmation_aliases_are_unseen_in_calibration(self) -> None:
        for spec in v5.PACK_SPECS.values():
            self.assertFalse(set(spec["calibration"]) & set(spec["confirmation"]))


if __name__ == "__main__":
    unittest.main()
