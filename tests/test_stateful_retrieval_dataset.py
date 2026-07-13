from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class StatefulRetrievalDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(
            (
                ROOT / "data/datasets/planner_stateful_retrieval_v1/manifest.json"
            ).read_text(encoding="utf-8")
        )

    def test_splits_are_entity_and_template_isolated(self) -> None:
        for pair in ("train_dev", "train_test", "dev_test"):
            integrity = self.manifest["integrity"][pair]
            self.assertEqual(integrity["case_id_overlap"], 0)
            self.assertEqual(integrity["exact_query_overlap"], 0)
            self.assertEqual(integrity["entity_overlap"], 0)
            self.assertEqual(integrity["template_overlap"], 0)

    def test_all_splits_cover_the_five_scenario_families(self) -> None:
        expected = {
            "coref_rewrite_then_rag",
            "direct_rag_guardrail",
            "general_answer_guardrail",
            "memory_hit_end",
            "rag_miss_rewrite_then_rag",
        }
        for split in ("train", "dev", "test"):
            categories = set(self.manifest["splits"][split]["cases"]["categories"])
            self.assertEqual(categories, expected)

    def test_derived_step_rows_have_no_exact_duplicates(self) -> None:
        for split in ("train", "dev", "test"):
            steps = self.manifest["splits"][split]["steps"]
            self.assertEqual(steps["exact_prompt_completion_duplicates"], 0)
            self.assertEqual(steps["duplicate_rate"], 0.0)

    def test_test_split_is_registered_as_sealed(self) -> None:
        self.assertEqual(
            self.manifest["integrity"]["status"], "sealed_test_created_unopened"
        )
        self.assertEqual(self.manifest["bootstrap_cluster"], "entity_id")


if __name__ == "__main__":
    unittest.main()
