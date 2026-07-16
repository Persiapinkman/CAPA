from __future__ import annotations

import unittest

from pipelines.eval.check_model_separation_challenge_gate import (
    calibration_gate,
    confirmation_gate,
)


def aggregate(family_rates: dict[str, float], *, overall: float, runs: int) -> dict:
    return {
        "runs": runs,
        "aggregate": {"pass_all_runs_rate": overall},
        "by_category": {
            family: {"pass_all_runs_rate": value}
            for family, value in family_rates.items()
        },
        "prediction_stats": {"errors_total": {}, "empty_decisions_mean": 0},
    }


class ModelSeparationChallengeGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cases = [
            {
                "case_id": "a1",
                "scenario_id": "multi_good",
                "expected_decisions": [{}, {}],
            },
            {
                "case_id": "b1",
                "scenario_id": "single_good",
                "expected_decisions": [{}],
            },
            {
                "case_id": "c1",
                "scenario_id": "reference_weak",
                "expected_decisions": [{}, {}],
            },
        ]

    def test_calibration_selects_whole_families_only(self) -> None:
        base = aggregate(
            {"multi_good": 0.40, "single_good": 0.60, "reference_weak": 0.10},
            overall=0.36,
            runs=1,
        )
        reference = aggregate(
            {"multi_good": 1.0, "single_good": 0.96, "reference_weak": 0.80},
            overall=0.92,
            runs=1,
        )
        result = calibration_gate(
            base=base,
            reference=reference,
            cases=self.cases,
            reference_min=0.95,
            base_max=0.70,
            gap_min=0.25,
            min_families=2,
            min_multistep_families=1,
        )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["accepted_scenarios"], ["multi_good", "single_good"])
        self.assertNotIn("reference_weak", result["accepted_scenarios"])

    def test_confirmation_is_all_or_nothing(self) -> None:
        cases = (self.cases[:2] * 300)[:600]
        base = aggregate({"multi_good": 0.40, "single_good": 0.60}, overall=0.50, runs=3)
        reference = aggregate({"multi_good": 0.96, "single_good": 0.98}, overall=0.97, runs=3)
        result = confirmation_gate(
            base=base,
            reference=reference,
            cases=cases,
            reference_min=0.95,
            base_max=0.60,
            gap_min=0.30,
            family_reference_min=0.90,
            family_base_max=0.75,
            expected_cases=600,
            required_runs=3,
        )
        self.assertEqual(result["status"], "pass")
        self.assertNotIn("accepted_scenarios", result)

    def test_confirmation_reports_recovered_first_length_truncation(self) -> None:
        cases = (self.cases[:2] * 300)[:600]
        base = aggregate({"multi_good": 0.40, "single_good": 0.60}, overall=0.50, runs=3)
        reference = aggregate({"multi_good": 0.96, "single_good": 0.98}, overall=0.97, runs=3)
        reference["prediction_stats"]["first_length_truncations_total"] = 1
        result = confirmation_gate(
            base=base,
            reference=reference,
            cases=cases,
            reference_min=0.95,
            base_max=0.60,
            gap_min=0.30,
            family_reference_min=0.90,
            family_base_max=0.75,
            expected_cases=600,
            required_runs=3,
        )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(
            result["runtime_diagnostics"]["reference_recovered_first_length_truncations"],
            1,
        )

    def test_confirmation_rejects_retry_length_truncation(self) -> None:
        cases = (self.cases[:2] * 300)[:600]
        base = aggregate({"multi_good": 0.40, "single_good": 0.60}, overall=0.50, runs=3)
        reference = aggregate({"multi_good": 0.96, "single_good": 0.98}, overall=0.97, runs=3)
        reference["prediction_stats"]["retry_length_truncations_total"] = 1
        result = confirmation_gate(
            base=base,
            reference=reference,
            cases=cases,
            reference_min=0.95,
            base_max=0.60,
            gap_min=0.30,
            family_reference_min=0.90,
            family_base_max=0.75,
            expected_cases=600,
            required_runs=3,
        )
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["checks"]["no_unrecovered_length_truncations"])


if __name__ == "__main__":
    unittest.main()
