from __future__ import annotations

import copy
import unittest

from pipelines.eval.check_runtime_routing_multiseed_test_gate import (
    evaluate_test_gate,
)


def category(action: float, score: float, strict: float) -> dict:
    return {
        "action_match_rate": action,
        "mean_score": score,
        "strict_case_action_rate": strict,
        "wrong_side_effect_count": 0,
    }


class RuntimeRoutingMultiSeedTestGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.study = {
            "study_id": "study",
            "arm_id": "arm",
            "test_confirmation": {
                "models": ["baseline", "seed42", "seed43", "seed44"]
            },
            "test_confirmation_gate": {
                "primary_category": "primary",
                "require_three_seed_mean_strict_case_action_delta_positive": True,
                "require_three_seed_mean_step_action_delta_positive": True,
                "minimum_positive_seed_count": 2,
                "require_entity_clustered_action_ci_lower_above_zero": True,
                "require_three_seed_mean_verifier_delta_positive": True,
                "require_mean_step2_action_no_regression": True,
                "contrast_category": "contrast",
                "maximum_mean_contrast_action_regression": 0.0,
                "maximum_mean_other_category_action_regression": 0.125,
                "require_no_mean_increase_in_wrong_side_effecting_actions": True,
                "preregistered_while_test_sealed": True,
            },
        }
        self.baseline = {
            "action_match_rate": 0.60,
            "wrong_side_effect_count": 1,
            "categories": {
                "primary": category(0.50, 0.50, 0.25),
                "contrast": category(1.0, 1.0, 1.0),
                "guard": category(0.75, 0.75, 0.75),
            },
            "category_steps": {
                "primary#step2": {"action_match_rate": 0.75}
            },
        }
        candidate = {
            "action_match_rate": 0.65,
            "wrong_side_effect_count": 1,
            "categories": {
                "primary": category(0.60, 0.55, 0.375),
                "contrast": category(1.0, 1.0, 1.0),
                "guard": category(0.75, 0.75, 0.75),
            },
            "category_steps": {
                "primary#step2": {"action_match_rate": 0.75}
            },
        }
        self.candidates = {
            "seed42": copy.deepcopy(candidate),
            "seed43": copy.deepcopy(candidate),
            "seed44": copy.deepcopy(candidate),
        }
        self.comparison = {"action_match_ci95": [0.01, 0.09]}

    def test_positive_confirmation_passes(self) -> None:
        result = evaluate_test_gate(
            self.study, self.baseline, self.candidates, self.comparison
        )
        self.assertTrue(result["passed"])
        self.assertTrue(result["test_confirmed"])
        self.assertEqual(result["split"], "test")

    def test_zero_primary_effect_does_not_confirm(self) -> None:
        candidates = copy.deepcopy(self.candidates)
        for candidate in candidates.values():
            candidate["categories"]["primary"]["strict_case_action_rate"] = 0.25
        result = evaluate_test_gate(
            self.study, self.baseline, candidates, self.comparison
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["primary_strict_case_action"])

    def test_zero_crossing_action_interval_does_not_confirm(self) -> None:
        comparison = {"action_match_ci95": [0.0, 0.09]}
        result = evaluate_test_gate(
            self.study, self.baseline, self.candidates, comparison
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["entity_clustered_action_ci"])


if __name__ == "__main__":
    unittest.main()
