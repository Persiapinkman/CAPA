from __future__ import annotations

import unittest

from pipelines.eval.check_grpo_value_challenge_gate import operational_valid, select_balanced_families
from training.planner_grpo_seed_v1.scripts import (
    build_planner_multistep_grpo_value_v3 as v3,
)
from training.planner_grpo_seed_v1.scripts.reward_planner_grpo import score_case


class PlannerMultistepGrpoValueV3Tests(unittest.TestCase):
    def test_calibration_is_complete_blocked_counterfactual_design(self) -> None:
        rows = v3.build_split(
            split="calibration",
            accepted_scenarios=[],
            confirmation_cases=v3.DEFAULT_CONFIRMATION_CASES,
        )
        self.assertEqual(len(rows), 384)
        self.assertEqual(len({row["entity_id"] for row in rows}), 24)
        self.assertEqual({row["scenario_id"] for row in rows}, set(v3.SCENARIOS))
        self.assertFalse(v3.validate_rows(rows))
        for entity_id in {row["entity_id"] for row in rows}:
            block = [row for row in rows if row["entity_id"] == entity_id]
            self.assertEqual(len(block), 16)
            qwen_queries = {row["user_query"] for row in block if row["detector_family"] == "qwen"}
            rex_queries = {row["user_query"] for row in block if row["detector_family"] == "rex"}
            self.assertEqual(len(qwen_queries), 1)
            self.assertEqual(len(rex_queries), 1)

    def test_current_observations_never_leak_the_next_action(self) -> None:
        rows = v3.build_split(
            split="calibration",
            accepted_scenarios=[],
            confirmation_cases=v3.DEFAULT_CONFIRMATION_CASES,
        )
        for row in rows:
            for item in row["mock_observations"]:
                summary = item["observation"]["summary"]
                for hint in v3.CURRENT_OBSERVATION_FORBIDDEN_HINTS:
                    self.assertNotIn(hint, summary)

    def test_retry_branches_have_three_steps_and_two_observations(self) -> None:
        rows = v3.build_split(
            split="calibration",
            accepted_scenarios=[],
            confirmation_cases=v3.DEFAULT_CONFIRMATION_CASES,
        )
        retry_rows = [row for row in rows if "retry_" in row["scenario_id"]]
        self.assertTrue(retry_rows)
        self.assertTrue(all(len(row["expected_decisions"]) == 3 for row in retry_rows))
        self.assertTrue(all(len(row["mock_observations"]) == 2 for row in retry_rows))
        self.assertTrue(
            all(
                row["expected_decisions"][0]["action"]
                == row["expected_decisions"][1]["action"]
                for row in retry_rows
            )
        )

    def test_canonical_gold_passes_strict_reward(self) -> None:
        rows = v3.build_split(
            split="calibration",
            accepted_scenarios=[],
            confirmation_cases=v3.DEFAULT_CONFIRMATION_CASES,
        )
        self.assertTrue(all(score_case(row)["passed"] for row in rows))

    def test_qwen35_step_prompt_has_non_thinking_marker(self) -> None:
        cases = v3.build_split(
            split="calibration",
            accepted_scenarios=[],
            confirmation_cases=v3.DEFAULT_CONFIRMATION_CASES,
        )[:2]
        rows = v3.qwen35_step_rows(cases)
        self.assertTrue(rows)
        self.assertTrue(
            all(row["prompt"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n") for row in rows)
        )

    def test_confirmation_is_600_and_split_isolated(self) -> None:
        selected = list(v3.SCENARIOS[:8])
        calibration = v3.build_split(
            split="calibration",
            accepted_scenarios=[],
            confirmation_cases=v3.DEFAULT_CONFIRMATION_CASES,
        )
        confirmation = v3.build_split(
            split="confirmation",
            accepted_scenarios=selected,
            confirmation_cases=v3.DEFAULT_CONFIRMATION_CASES,
        )
        self.assertEqual(len(confirmation), 600)
        self.assertEqual(len({row["entity_id"] for row in confirmation}), 75)
        self.assertEqual({row["scenario_id"] for row in confirmation}, set(selected))
        self.assertFalse(v3.validate_rows(confirmation))
        self.assertEqual(
            v3.v1.split_integrity(
                {"calibration": calibration, "confirmation": confirmation}
            )["status"],
            "pass",
        )

    def test_balanced_selection_is_fixed_by_scenario_order(self) -> None:
        selected = select_balanced_families(list(v3.SCENARIOS), list(v3.SCENARIOS))
        self.assertEqual(selected, list(v3.SCENARIOS[:8]))
        self.assertEqual(sum(value.startswith("qwen_") for value in selected), 4)
        self.assertEqual(sum(value.startswith("rex_") for value in selected), 4)

    def test_operational_gate_rejects_planner_fallback_errors(self) -> None:
        aggregate = {
            "runs": 1,
            "prediction_stats": {
                "errors_total": {},
                "fallback_errors_total": {"ImportError": 1},
                "empty_decisions_mean": 0,
                "retry_length_truncations_total": 0,
            },
        }
        self.assertFalse(operational_valid(aggregate, 1))

    def test_operational_gate_keeps_model_parse_failures_in_denominator(self) -> None:
        aggregate = {
            "runs": 1,
            "prediction_stats": {
                "errors_total": {},
                "fallback_errors_total": {"ValueError": 7},
                "empty_decisions_mean": 0,
                "retry_length_truncations_total": 0,
            },
        }
        self.assertTrue(operational_valid(aggregate, 1))


if __name__ == "__main__":
    unittest.main()
