from __future__ import annotations

import unittest

from pipelines.eval.compare_generation_runs import paired_comparison, summary


def row(
    case_id: str,
    step_index: int,
    category: str,
    score: float,
    *,
    entity_id: str = "",
    action_match: float | None = None,
) -> dict:
    return {
        "case_id": case_id,
        "step_index": step_index,
        "category": category,
        "score": score,
        "json_valid": 1.0,
        "extra_text": 0.0,
        "entity_id": entity_id,
        "action_match": action_match,
    }


class ComparisonTests(unittest.TestCase):
    def test_case_macro_does_not_double_weight_two_step_case(self) -> None:
        rows = {
            ("case-a", 1): row("case-a", 1, "multi", 1.0),
            ("case-a", 2): row("case-a", 2, "multi", 1.0),
            ("case-b", 1): row("case-b", 1, "single", 0.0),
        }
        result = summary(rows)
        self.assertAlmostEqual(result["step_mean"], 2 / 3)
        self.assertAlmostEqual(result["case_macro_mean"], 0.5)

    def test_paired_bootstrap_detects_uniform_improvement(self) -> None:
        left = {
            ("case-a", 1): row("case-a", 1, "x", 0.2),
            ("case-b", 1): row("case-b", 1, "x", 0.4),
            ("case-c", 1): row("case-c", 1, "x", 0.6),
        }
        right = {
            key: row(value["case_id"], value["step_index"], value["category"], value["score"] + 0.1)
            for key, value in left.items()
        }
        result = paired_comparison(left, right, seed=42, samples=1000)
        self.assertEqual(result["conclusion"], "supported")
        self.assertAlmostEqual(result["case_macro_delta"], 0.1)

    def test_bootstrap_can_cluster_related_cases_by_entity(self) -> None:
        left = {
            ("case-a1", 1): row("case-a1", 1, "x", 0.2, entity_id="entity-a"),
            ("case-a2", 1): row("case-a2", 1, "x", 0.3, entity_id="entity-a"),
            ("case-b1", 1): row("case-b1", 1, "x", 0.4, entity_id="entity-b"),
        }
        right = {
            key: row(
                value["case_id"],
                value["step_index"],
                value["category"],
                value["score"] + 0.1,
                entity_id=value["entity_id"],
            )
            for key, value in left.items()
        }
        result = paired_comparison(
            left, right, seed=42, samples=1000, cluster_key="entity_id"
        )
        self.assertEqual(result["clusters"], 2)
        self.assertEqual(result["bootstrap"]["cluster"], "entity_id")

    def test_summary_reports_category_step_action_accuracy(self) -> None:
        rows = {
            ("case-a", 1): row("case-a", 1, "coref", 0.2, action_match=0.0),
            ("case-a", 2): row("case-a", 2, "coref", 0.8, action_match=1.0),
            ("case-b", 1): row("case-b", 1, "direct", 1.0, action_match=1.0),
        }
        result = summary(rows)
        self.assertAlmostEqual(result["action_match_rate"], 2 / 3)
        self.assertEqual(result["category_steps"]["coref#step1"]["action_match_rate"], 0.0)

    def test_action_match_bootstrap_reports_uniform_gain(self) -> None:
        left = {
            ("case-a", 1): row("case-a", 1, "coref", 0.2, action_match=0.0),
            ("case-b", 1): row("case-b", 1, "coref", 0.3, action_match=0.0),
        }
        right = {
            ("case-a", 1): row("case-a", 1, "coref", 0.9, action_match=1.0),
            ("case-b", 1): row("case-b", 1, "coref", 1.0, action_match=1.0),
        }
        result = paired_comparison(left, right, seed=42, samples=1000)
        self.assertEqual(result["action_match_delta"], 1.0)
        self.assertEqual(result["action_match_ci95"], [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
