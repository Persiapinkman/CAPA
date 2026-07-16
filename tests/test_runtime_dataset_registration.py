from __future__ import annotations

import unittest

from pipelines.data.register_runtime_routing_dataset import _default_paths


class RuntimeDatasetRegistrationTests(unittest.TestCase):
    def test_routing_layout(self) -> None:
        cases, steps = _default_paths("planner_runtime_routing_v1", "dev")
        self.assertEqual(
            str(cases),
            "training/planner_grpo_seed_v1/cases/planner_runtime_routing_v1_dev_cases.jsonl",
        )
        self.assertEqual(
            str(steps),
            "training/planner_grpo_seed_v1/sft_data_runtime_routing_v1_chatml/dev.jsonl",
        )

    def test_probe_curriculum_layout(self) -> None:
        cases, steps = _default_paths(
            "planner_runtime_probe_curriculum_v1", "train"
        )
        self.assertEqual(
            str(cases),
            "training/planner_grpo_seed_v1/cases/planner_runtime_probe_curriculum_v1_train_cases.jsonl",
        )
        self.assertEqual(
            str(steps),
            "training/planner_grpo_seed_v1/sft_data_runtime_probe_curriculum_v1_chatml/train.jsonl",
        )

    def test_unknown_layout_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            _default_paths("unknown", "train")


if __name__ == "__main__":
    unittest.main()
