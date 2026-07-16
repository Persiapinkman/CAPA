from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from util.prompts import evaluation_summary_prompt
from util.schemas import evaluation_summary_response_schema


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills/eval-reports-generation/scripts/run_eval_report_generation.py"


def _load_report_module():
    spec = importlib.util.spec_from_file_location("evaluation_report", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EvaluationReportContractTests(unittest.TestCase):
    def test_cross_model_agreement_is_deterministic_and_not_ground_truth(self) -> None:
        module = _load_report_module()
        prediction = [
            {
                "image": "fixture.jpg",
                "source": "generated",
                "image_idx": 0,
                "models": [
                    {"model": "qwen3-vl-8b", "pred_bboxes": [[0, 0, 10, 10]]},
                    {"model": "rex-omni", "pred_bboxes": [[0, 0, 10, 10]]},
                ],
            }
        ]
        agreement = module.cross_model_agreement(prediction)
        self.assertEqual(agreement[0]["matched_at_iou_0_5"], 1)
        self.assertEqual(agreement[0]["match_rate"], 1.0)
        self.assertEqual(agreement[0]["mean_matched_iou"], 1.0)

    def test_non_overlapping_boxes_do_not_match(self) -> None:
        module = _load_report_module()
        prediction = [
            {
                "image": "fixture.jpg",
                "source": "generated",
                "image_idx": 0,
                "models": [
                    {"model": "qwen3-vl-8b", "pred_bboxes": [[0, 0, 10, 10]]},
                    {"model": "rex-omni", "pred_bboxes": [[20, 20, 30, 30]]},
                ],
            }
        ]
        agreement = module.cross_model_agreement(prediction)
        self.assertEqual(agreement[0]["matched_at_iou_0_5"], 0)
        self.assertEqual(agreement[0]["match_rate"], 0.0)

    def test_report_contract_allows_an_inconclusive_recommendation(self) -> None:
        recommendation = evaluation_summary_response_schema["properties"]["recommendation"]
        self.assertIn("inconclusive", recommendation["enum"])
        self.assertIn("{agreement_json}", evaluation_summary_prompt)
        self.assertNotIn("qwen3-vl-8b 在大部分图片", evaluation_summary_prompt)

    def test_default_report_does_not_invent_accuracy_without_gt(self) -> None:
        module = _load_report_module()
        report = module.build_deterministic_report(
            [
                {
                    "image_idx": 0,
                    "image": "fixture.jpg",
                    "source": "original",
                    "qwen_box_count": 1,
                    "rex_box_count": 1,
                    "matched_at_iou_0_5": 1,
                    "match_rate": 1.0,
                    "mean_matched_iou": 0.9,
                }
            ]
        )
        self.assertEqual(report["recommendation"], "inconclusive")
        self.assertIn("无人工GT", report["model_results"]["qwen3-vl-8b"]["accuracy"])
        self.assertIn("不能据此计算准确率", report["overall_conclusion"])


if __name__ == "__main__":
    unittest.main()
