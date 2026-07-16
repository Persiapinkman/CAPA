from __future__ import annotations

import unittest

from pipelines.eval.bootstrap_model_separation_clusters import analyze


class ClusterBootstrapTests(unittest.TestCase):
    def test_paired_cluster_analysis(self) -> None:
        cases = [
            {"case_id": "a1", "entity_id": "a"},
            {"case_id": "a2", "entity_id": "a"},
            {"case_id": "b1", "entity_id": "b"},
            {"case_id": "b2", "entity_id": "b"},
        ]
        result = analyze(
            cases=cases,
            base_pass={"a1": True, "a2": False, "b1": False, "b2": False},
            reference_pass={"a1": True, "a2": True, "b1": True, "b2": False},
            bootstrap_samples=100,
            seed=1,
        )
        self.assertEqual(result["clusters"], 2)
        self.assertEqual(result["metrics"]["base_pass_all"], 0.25)
        self.assertEqual(result["metrics"]["reference_pass_all"], 0.75)
        self.assertEqual(result["paired_case_counts"]["reference_only"], 2)


if __name__ == "__main__":
    unittest.main()
