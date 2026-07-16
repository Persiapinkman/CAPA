from __future__ import annotations

import copy
import unittest

from pipelines.eval.check_runtime_routing_multiseed_gate import evaluate_gate


def category(
    action: float, score: float, strict: float, *, wrong_side_effects: int = 0
) -> dict:
    return {
        "action_match_rate": action,
        "mean_score": score,
        "strict_case_action_rate": strict,
        "wrong_side_effect_count": wrong_side_effects,
    }


class RuntimeRoutingMultiSeedGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.study = {
            "study_id": "study",
            "arm_id": "arm",
            "development_replication_gate": {
                "primary_category": "primary",
                "minimum_three_seed_mean_strict_case_action_delta": 0.125,
                "minimum_three_seed_mean_step_action_delta": 0.10,
                "minimum_positive_seed_count": 2,
                "require_entity_clustered_action_ci_lower_above_zero": True,
                "minimum_three_seed_mean_verifier_delta": 0.05,
                "require_mean_step2_action_no_regression": True,
                "contrast_category": "contrast",
                "maximum_mean_contrast_action_regression": 0.0,
                "maximum_mean_other_category_action_regression": 0.125,
                "require_no_mean_increase_in_wrong_side_effecting_actions": True,
            },
            "test_confirmation": {
                "models": ["baseline", "seed42", "seed43", "seed44"]
            },
        }
        self.baseline = {
            "action_match_rate": 0.60,
            "wrong_side_effect_count": 2,
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
            "action_match_rate": 0.72,
            "wrong_side_effect_count": 2,
            "categories": {
                "primary": category(0.75, 0.62, 0.50),
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
        self.comparison = {"action_match_ci95": [0.03, 0.18]}

    def test_all_preregistered_checks_pass(self) -> None:
        result = evaluate_gate(
            self.study, self.baseline, self.candidates, self.comparison
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["primary"]["positive_seed_count"], 3)

    def test_incomplete_case_seed_does_not_get_step_level_credit(self) -> None:
        candidates = copy.deepcopy(self.candidates)
        candidates["seed43"]["categories"]["primary"][
            "strict_case_action_rate"
        ] = 0.25
        candidates["seed44"]["categories"]["primary"][
            "strict_case_action_rate"
        ] = 0.25
        result = evaluate_gate(self.study, self.baseline, candidates, self.comparison)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["primary_strict_case_action"])
        self.assertFalse(result["checks"]["positive_seeds"])

    def test_mean_side_effect_increase_blocks_test_opening(self) -> None:
        candidates = copy.deepcopy(self.candidates)
        candidates["seed44"]["wrong_side_effect_count"] = 3
        result = evaluate_gate(self.study, self.baseline, candidates, self.comparison)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["wrong_side_effects"])

    def test_missing_preregistered_seed_blocks_test_opening(self) -> None:
        candidates = copy.deepcopy(self.candidates)
        candidates.pop("seed44")
        result = evaluate_gate(self.study, self.baseline, candidates, self.comparison)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["candidate_seed_count"])


if __name__ == "__main__":
    unittest.main()
