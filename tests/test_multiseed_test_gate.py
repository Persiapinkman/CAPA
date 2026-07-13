from __future__ import annotations

import copy
import unittest

from pipelines.eval.check_multiseed_test_gate import evaluate_test_gate


class MultiSeedTestGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.arm = {
            "study_id": "study",
            "arm_id": "arm",
            "test_gate": {
                "primary_category": "target",
                "require_three_seed_mean_action_delta_positive": True,
                "require_entity_clustered_action_ci_lower_above_zero": True,
                "require_three_seed_mean_score_delta_positive": True,
                "guardrail_categories": ["guard"],
                "maximum_mean_guardrail_regression": 0.03,
                "anti_shortcut_category": "anti",
                "maximum_mean_anti_shortcut_regression": 0.05,
            },
        }
        self.primary = {
            "mean_policy": {
                "comparison": {
                    "action_match_delta": 0.05,
                    "action_match_ci95": [0.01, 0.09],
                    "case_macro_delta": 0.04,
                }
            }
        }
        self.full = {
            "mean_policy": {"category_deltas": {"guard": -0.01, "anti": 0.0}}
        }

    def test_all_test_conditions_pass(self) -> None:
        result = evaluate_test_gate(self.arm, self.full, self.primary)
        self.assertTrue(result["passed"])
        self.assertEqual(result["split"], "test")

    def test_zero_action_delta_is_not_positive(self) -> None:
        primary = copy.deepcopy(self.primary)
        primary["mean_policy"]["comparison"]["action_match_delta"] = 0.0
        result = evaluate_test_gate(self.arm, self.full, primary)
        self.assertFalse(result["passed"])
        self.assertFalse(result["primary"]["mean_action_positive_passed"])

    def test_anti_shortcut_regression_blocks_confirmation(self) -> None:
        full = copy.deepcopy(self.full)
        full["mean_policy"]["category_deltas"]["anti"] = -0.051
        result = evaluate_test_gate(self.arm, full, self.primary)
        self.assertFalse(result["passed"])
        self.assertFalse(result["anti_shortcut"]["passed"])


if __name__ == "__main__":
    unittest.main()
