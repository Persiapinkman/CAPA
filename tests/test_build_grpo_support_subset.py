from __future__ import annotations

import unittest

from pipelines.data.build_grpo_support_subset import select_rows


class BuildGrpoSupportSubsetTests(unittest.TestCase):
    def test_selects_complete_steps_for_balanced_entities(self) -> None:
        rows = [
            {
                "case_id": f"{category}-{entity}",
                "category": category,
                "entity_id": entity,
                "step_index": step,
            }
            for category in ("a", "b")
            for entity in ("e1", "e2", "e3")
            for step in (1, 2)
        ]
        selected = select_rows(rows, ["a", "b"], entities_per_category=2)
        self.assertEqual(len(selected), 8)
        self.assertEqual({row["entity_id"] for row in selected}, {"e1", "e2"})
        self.assertEqual({row["step_index"] for row in selected}, {1, 2})

    def test_rejects_insufficient_entities(self) -> None:
        rows = [{"case_id": "a-e1", "category": "a", "entity_id": "e1", "step_index": 1}]
        with self.assertRaisesRegex(ValueError, "requested 2"):
            select_rows(rows, ["a"], entities_per_category=2)


if __name__ == "__main__":
    unittest.main()
