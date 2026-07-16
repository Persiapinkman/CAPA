from __future__ import annotations

import unittest

from pipelines.eval.combine_planner_rollout_shards import combine


class CombinePlannerRolloutShardsTests(unittest.TestCase):
    def test_combines_in_case_order(self) -> None:
        cases = [{"case_id": "a"}, {"case_id": "b"}, {"case_id": "c"}]
        rows = combine(
            cases,
            [[{"case_id": "c", "value": 3}], [{"case_id": "a", "value": 1}, {"case_id": "b", "value": 2}]],
        )
        self.assertEqual([row["case_id"] for row in rows], ["a", "b", "c"])

    def test_rejects_overlap_and_missing_cases(self) -> None:
        cases = [{"case_id": "a"}, {"case_id": "b"}]
        with self.assertRaisesRegex(ValueError, "overlapping"):
            combine(cases, [[{"case_id": "a"}], [{"case_id": "a"}, {"case_id": "b"}]])
        with self.assertRaisesRegex(ValueError, "missing"):
            combine(cases, [[{"case_id": "a"}]])


if __name__ == "__main__":
    unittest.main()
