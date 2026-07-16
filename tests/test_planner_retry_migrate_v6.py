from __future__ import annotations

import json
import re
import unittest
from collections import Counter, defaultdict

from training.planner_grpo_seed_v1.scripts import build_planner_retry_migrate_v6 as v6
from training.planner_grpo_seed_v1.scripts.train_planner_grpo import score_step_completion


class PlannerRetryMigrateV6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        v6.write_fixture_images()
        cls.cases = v6.build_all_cases()

    def test_split_sizes_and_entity_isolation(self) -> None:
        expected = {
            "sft_train": (600, 80),
            "sft_dev": (150, 20),
            "grpo_train": (450, 60),
            "grpo_dev": (225, 30),
            "test": (450, 60),
        }
        for split, (case_count, entity_count) in expected.items():
            rows = self.cases[split]
            self.assertEqual(len(rows), case_count)
            self.assertEqual(len({row["entity_id"] for row in rows}), entity_count)
        self.assertEqual(v6.split_isolation_report(self.cases)["status"], "pass")

    def test_every_core_bundle_is_a_matched_three_state_counterfactual(self) -> None:
        for split, rows in self.cases.items():
            bundles: dict[str, list[dict]] = defaultdict(list)
            for row in rows:
                if not row["guardrail"]:
                    bundles[row["counterfactual_bundle_id"]].append(row)
            for bundle_id, bundle in bundles.items():
                self.assertEqual(len(bundle), 3, (split, bundle_id))
                self.assertEqual(
                    {row["scenario_id"] for row in bundle},
                    {"core_retryable_fresh", "core_nonretryable", "core_budget_exhausted"},
                )
                self.assertEqual(len({row["user_query"] for row in bundle}), 1)
                self.assertEqual(len({row["error_alias"] for row in bundle}), 1)
                self.assertEqual(len({row["badge_condition"] for row in bundle}), 1)
                self.assertEqual(
                    Counter(row["target_action_class"] for row in bundle),
                    Counter({"migrate": 2, "retry": 1}),
                )
                first_observations = [row["mock_observations"][0]["observation"] for row in bundle]
                normalized = {
                    re.sub(
                        r"retryable=(?:true|false)；retry_count=[^；。]+",
                        "retryable=<blocked>；retry_count=<blocked>",
                        observation["summary"],
                    )
                    for observation in first_observations
                }
                self.assertEqual(len(normalized), 1)

    def test_retry_trajectories_consume_a_second_observation(self) -> None:
        for rows in self.cases.values():
            retry_rows = [row for row in rows if row["scenario_id"] == "core_retryable_fresh"]
            self.assertTrue(retry_rows)
            for row in retry_rows:
                self.assertEqual(len(row["expected_decisions"]), 3)
                self.assertEqual(len(row["mock_observations"]), 2)
                self.assertEqual([item["after_step"] for item in row["mock_observations"]], [1, 2])

    def test_badge_and_alias_are_not_action_shortcuts(self) -> None:
        for split, rows in self.cases.items():
            core = [row for row in rows if not row["guardrail"]]
            self.assertAlmostEqual(
                v6.mutual_information(core, "badge_condition", "target_action_class"),
                0.0,
                places=12,
                msg=split,
            )
            self.assertAlmostEqual(
                v6.mutual_information(core, "error_alias", "target_action_class"),
                0.0,
                places=12,
                msg=split,
            )

    def test_independent_state_oracle_matches_every_transition_label(self) -> None:
        for split, rows in self.cases.items():
            for row in rows:
                expected = row["expected_decisions"]
                detector_action = v6.expected_action_name(expected[0])
                for item in row["mock_observations"]:
                    after_step = item["after_step"]
                    summary = item["observation"]["summary"]
                    self.assertEqual(
                        v6.independent_oracle_action(summary),
                        v6.expected_action_class(
                            expected[after_step],
                            detector_action=detector_action,
                        ),
                        (split, row["case_id"], after_step),
                    )

    def test_sanitized_prompts_contain_no_training_artifacts(self) -> None:
        probes = [
            next(row for row in self.cases["sft_train"] if row["scenario_id"] == "core_retryable_fresh"),
            next(row for row in self.cases["sft_train"] if row["scenario_id"] == "guard_stale_history_current_success_end"),
        ]
        for case in probes:
            step_index = len(case["expected_decisions"])
            prompt = v6.build_sanitized_pseudo_prompt(case, step_index)
            self.assertNotIn(case["case_id"], prompt)
            self.assertNotIn(case["entity_id"], prompt)
            self.assertNotIn("/raid/", prompt)
            self.assertNotIn("/tmp/", prompt)
            self.assertNotIn("按训练样本期望", prompt)
            self.assertNotIn('"external_ref"', prompt)
            self.assertNotIn('"_thought"', prompt)
            self.assertIn('"max_steps": 3', prompt)

    def test_final_retry_and_end_gold_receive_unit_step_reward(self) -> None:
        retry_expected = {
            "decision_type": "tool",
            "action": "qwen_detection",
            "required_args": {"finish_after_tool": False},
            "arg_contains": {"label": ["测试标记"]},
        }
        retry_actual = {
            "decision_type": "tool",
            "action": "qwen_detection",
            "action_input": {"label": "测试标记", "finish_after_tool": False},
        }
        end_expected = {
            "decision_type": "end",
            "required_args": {"end_reason": "memory_hit"},
            "arg_contains": {},
        }
        end_actual = {
            "decision_type": "end",
            "end_reason": "memory_hit",
            "final_answer": "",
        }
        reward_spec = json.dumps(v6.VALUE_REWARD, ensure_ascii=False)

        def score(actual: dict, expected: dict, action: str) -> float:
            return score_step_completion(
                completion=json.dumps(actual, ensure_ascii=False),
                expected_step=json.dumps(expected, ensure_ascii=False),
                forbidden_actions="[]",
                reward_spec=reward_spec,
                previous_action="qwen_detection",
                full_expected_actions=json.dumps(["qwen_detection", action]),
                step_index=2,
            )

        self.assertEqual(score(retry_actual, retry_expected, "qwen_detection"), 1.0)
        self.assertEqual(score(end_actual, end_expected, "end"), 1.0)


if __name__ == "__main__":
    unittest.main()
