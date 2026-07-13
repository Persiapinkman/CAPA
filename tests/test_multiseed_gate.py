from __future__ import annotations

import copy
import unittest

from pipelines.eval.check_multiseed_replication_gate import evaluate_gate


class MultiSeedGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.arm = {
            "study_id": "study",
            "arm_id": "arm",
            "development_replication_gate": {
                "primary_category": "target",
                "minimum_three_seed_mean_action_delta": 0.05,
                "minimum_positive_seed_count": 2,
                "require_entity_clustered_action_ci_lower_above_zero": True,
                "minimum_three_seed_mean_score_delta": 0.03,
                "guardrail_categories": ["guard"],
                "maximum_mean_guardrail_regression": 0.03,
                "anti_shortcut_category": "anti",
                "maximum_mean_anti_shortcut_regression": 0.05,
            },
        }
        self.primary = {
            "mean_policy": {
                "comparison": {
                    "action_match_delta": 0.075,
                    "action_match_ci95": [0.02, 0.13],
                    "case_macro_delta": 0.06,
                }
            },
            "seed_comparisons": {
                "seed42": {"action_match_delta": 0.075},
                "seed43": {"action_match_delta": 0.05},
                "seed44": {"action_match_delta": 0.10},
            },
        }
        self.full = {
            "mean_policy": {"category_deltas": {"guard": 0.0, "anti": -0.01}}
        }

    def test_all_preregistered_conditions_pass(self) -> None:
        result = evaluate_gate(self.arm, self.full, self.primary)
        self.assertTrue(result["passed"])
        self.assertEqual(result["primary"]["positive_seed_count"], 3)

    def test_positive_mean_does_not_override_zero_crossing_interval(self) -> None:
        primary = copy.deepcopy(self.primary)
        primary["mean_policy"]["comparison"]["action_match_ci95"] = [0.0, 0.13]
        result = evaluate_gate(self.arm, self.full, primary)
        self.assertFalse(result["passed"])
        self.assertFalse(result["primary"]["action_ci_lower_above_zero_passed"])

    def test_guardrail_regression_blocks_test_opening(self) -> None:
        full = copy.deepcopy(self.full)
        full["mean_policy"]["category_deltas"]["guard"] = -0.04
        result = evaluate_gate(self.arm, full, self.primary)
        self.assertFalse(result["passed"])
        self.assertFalse(result["guardrails"]["passed"])


if __name__ == "__main__":
    unittest.main()
