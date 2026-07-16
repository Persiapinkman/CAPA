from __future__ import annotations

import unittest

from pipelines.eval.audit_grpo_sampling_support import aggregate


class GrpoSupportMetricTests(unittest.TestCase):
    def test_reports_action_and_valid_action_support(self) -> None:
        groups = [
            {
                "category": "route",
                "step_index": 1,
                "mean_reward": 0.5,
                "reward_std": 0.2,
                "min_reward": 0.1,
                "max_reward": 1.0,
                "distinct_actions": 3,
                "distinct_valid_actions": 2,
                "mean_task_reward": 0.4,
                "max_task_reward": 1.0,
                "exact_action_support": True,
            },
            {
                "category": "route",
                "step_index": 2,
                "mean_reward": 0.3,
                "reward_std": 0.0,
                "min_reward": 0.3,
                "max_reward": 0.3,
                "distinct_actions": 1,
                "distinct_valid_actions": 1,
                "mean_task_reward": 0.2,
                "max_task_reward": 0.4,
                "exact_action_support": False,
            },
        ]
        overall = aggregate(groups)["overall"]
        self.assertEqual(overall["exact_action_support_rate"], 0.5)
        self.assertEqual(overall["exact_task_support_rate"], 0.5)
        self.assertEqual(overall["mean_distinct_valid_actions"], 1.5)
        category_steps = aggregate(groups)["category_steps"]
        self.assertEqual(category_steps["route::step1"]["groups"], 1)
        self.assertEqual(category_steps["route::step2"]["groups"], 1)


if __name__ == "__main__":
    unittest.main()
