from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DatasetAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(
            (ROOT / "data/datasets/planner_focused_v3/manifest.json").read_text(encoding="utf-8")
        )

    def test_grouped_train_dev_split_has_no_exact_id_leakage(self) -> None:
        overlap = self.manifest["integrity"]["train_dev_overlap"]
        self.assertEqual(overlap["case_id_overlap"], 0)
        self.assertEqual(overlap["exact_query_overlap"], 0)

    def test_regression_overlap_is_explicit(self) -> None:
        self.assertEqual(
            self.manifest["integrity"]["train_regression_overlap"]["case_id_overlap"], 69
        )

    def test_hard_refresh_duplicate_rates_are_audited(self) -> None:
        splits = self.manifest["splits"]
        self.assertGreater(splits["hard_v4_steps"]["duplicate_rate"], 0.6)
        self.assertGreater(splits["hard_v5_steps"]["duplicate_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
