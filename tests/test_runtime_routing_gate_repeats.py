from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pipelines.eval.check_runtime_routing_gate import _load_run


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class RuntimeRoutingGateRepeatTests(unittest.TestCase):
    def test_action_requires_unanimous_repeats(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            eval_path = root / "eval.jsonl"
            first_path = root / "first.jsonl"
            second_path = root / "second.jsonl"
            write_jsonl(
                eval_path,
                [
                    {
                        "case_id": "case",
                        "step_index": 1,
                        "category": "probe",
                        "entity_id": "entity",
                        "expected_step": json.dumps(
                            {
                                "decision_type": "tool",
                                "action": "qwen_detection",
                            }
                        ),
                    }
                ],
            )
            base = {
                "case_id": "case",
                "step_index": 1,
                "category": "probe",
                "entity_id": "entity",
                "json_valid": True,
                "extra_text_after_json": False,
            }
            write_jsonl(
                first_path,
                [
                    {
                        **base,
                        "score": 1.0,
                        "scored_completion": json.dumps(
                            {
                                "decision_type": "tool",
                                "action": "qwen_detection",
                            }
                        ),
                    }
                ],
            )
            write_jsonl(
                second_path,
                [
                    {
                        **base,
                        "score": 0.2,
                        "scored_completion": json.dumps(
                            {
                                "decision_type": "tool",
                                "action": "rexomni_detection",
                            }
                        ),
                    }
                ],
            )
            record_path = root / "run_record.json"
            record_path.write_text(
                json.dumps(
                    {
                        "data": {"files": {"eval": str(eval_path)}},
                        "artifacts": {
                            "predictions": [str(first_path), str(second_path)]
                        },
                    }
                ),
                encoding="utf-8",
            )
            loaded = _load_run(record_path)
        row = loaded["rows"][("case", 1)]
        self.assertEqual(loaded["repeat_count"], 2)
        self.assertFalse(row["action_match"])
        self.assertEqual(row["repeat_action_match_rate"], 0.5)
        self.assertAlmostEqual(row["score"], 0.6)
        self.assertEqual(row["actual_action"], "repeat_disagreement")

    def test_any_wrong_side_effect_repeat_is_counted(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            eval_path = root / "eval.jsonl"
            prediction_path = root / "prediction.jsonl"
            write_jsonl(
                eval_path,
                [
                    {
                        "case_id": "case",
                        "step_index": 1,
                        "category": "answer",
                        "entity_id": "entity",
                        "expected_step": json.dumps(
                            {"decision_type": "tool", "action": "answerer"}
                        ),
                    }
                ],
            )
            write_jsonl(
                prediction_path,
                [
                    {
                        "case_id": "case",
                        "step_index": 1,
                        "score": 0.2,
                        "scored_completion": json.dumps(
                            {
                                "decision_type": "tool",
                                "action": "pipeline_eval",
                            }
                        ),
                    }
                ],
            )
            record_path = root / "run_record.json"
            record_path.write_text(
                json.dumps(
                    {
                        "data": {"files": {"eval": str(eval_path)}},
                        "artifacts": {"predictions": [str(prediction_path)]},
                    }
                ),
                encoding="utf-8",
            )
            loaded = _load_run(record_path)
        self.assertTrue(loaded["rows"][("case", 1)]["wrong_side_effect"])


if __name__ == "__main__":
    unittest.main()
