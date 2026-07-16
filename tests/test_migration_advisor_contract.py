from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "demo/migration_advisor.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("migration_advisor_contract", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MigrationAdvisorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module()

    def test_retrieve_artifact_excludes_raw_response_and_bounds_documents(self) -> None:
        result = {
            "success": True,
            "field": "existing_models",
            "query": "渔民检测模型",
            "api_mode": "unified",
            "retrieved_chunks": [
                {"evidence_id": "e1", "content": "有效证据" + "证" * 5000, "score": 0.9}
            ],
            "full_documents": [
                {"doc_id": "d1", "doc_name": "文档", "content": "正文" * 10000}
            ],
            "raw_response": {"payload": "x" * 1_000_000},
        }

        compact = self.module.compact_retrieve_result_for_artifact(result)

        self.assertNotIn("raw_response", compact)
        self.assertEqual(compact["retrieved_count"], 1)
        self.assertEqual(compact["full_document_count"], 1)
        self.assertLess(len(json.dumps(compact, ensure_ascii=False)), 10_000)

    def test_field_artifact_excludes_runtime_query_copies(self) -> None:
        result = {
            "field": "existing_models",
            "queries": ["查询一", "查询二"],
            "coverage": "strong",
            "retrieved_chunks": [{"evidence_id": "e1", "content": "证据"}],
            "full_documents": [{"doc_id": "d1", "content": "正文"}],
            "raw_results": [{"raw_response": {"payload": "x" * 100_000}}],
            "query_summaries": [{"query": "查询一", "success": True, "retrieved_count": 1}],
        }

        compact = self.module.compact_field_results_for_artifact([result])[0]

        self.assertNotIn("raw_results", compact)
        self.assertEqual(compact["evidence_count"], 1)
        self.assertEqual(compact["query_summaries"][0]["query"], "查询一")

    def test_fact_quote_must_exist_in_cited_source(self) -> None:
        packages = [
            {
                "field": "performance_baseline",
                "evidences": [
                    {"evidence_id": "e1", "snippet": "模型甲在测试集上的 mAP 为 0.73。"}
                ],
                "full_documents": [],
            }
        ]
        valid = {
            "fact_type": "accuracy",
            "field": "performance_baseline",
            "subject": "模型甲",
            "fact": "mAP 为 0.73",
            "value": "0.73",
            "unit": "mAP",
            "evidence_ids": ["e1"],
            "doc_ids": [],
            "quote": "模型甲在测试集上的 mAP 为 0.73。",
            "confidence": "high",
        }
        fabricated = dict(valid, fact="mAP 为 0.99", quote="模型甲的 mAP 为 0.99。")

        facts = self.module._normalize_fact_list(
            [valid, fabricated],
            evidence_packages=packages,
        )

        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["value"], "0.73")

    def test_evidence_gate_keeps_relevant_asset_and_excludes_adela(self) -> None:
        plan = {
            "abstract_requirement": {
                "object": "河道监控",
                "attribute": "钓鱼人员检测",
                "task_type": "目标检测",
                "scene": "户外水域",
                "constraints": [],
            }
        }
        field_results = [
            {
                "field": "existing_models",
                "queries": ["钓鱼人员检测"],
                "coverage": "strong",
                "retrieved_chunks": [
                    {
                        "evidence_id": "fishing",
                        "score": 0.9,
                        "payload": {
                            "index_text": "算法名称: 钓鱼人体检测\n模型名称: fisher_det.model",
                            "algorithm_name": "钓鱼人体检测",
                            "model_name": "fisher_det.model",
                        },
                    },
                    {
                        "evidence_id": "face",
                        "score": 0.99,
                        "payload": {
                            "index_text": "算法名称: 人脸检测\n模型名称: face_det.model",
                            "algorithm_name": "人脸检测",
                            "model_name": "face_det.model",
                        },
                    },
                    {
                        "evidence_id": "adela",
                        "score": 1.0,
                        "payload": {
                            "index_text": "Adela 钓鱼人体检测 mAP 0.99",
                            "algorithm_name": "钓鱼人体检测",
                        },
                    },
                ],
                "full_documents": [],
            }
        ]

        packages = self.module._build_evidence_packages(
            field_results,
            user_query="河道钓鱼人员检测",
            plan=plan,
        )
        evidence_ids = [item["evidence_id"] for item in packages[0]["evidences"]]
        structured = self.module._structured_facts_from_packages(packages)

        self.assertEqual(evidence_ids, ["fishing"])
        self.assertTrue(any(fact["fact_type"] == "model_identity" for fact in structured))
        self.assertFalse(any("Adela" in fact["quote"] for fact in structured))

    def test_report_numbers_are_replaced_by_validated_facts(self) -> None:
        report = {
            "requirement_summary": "检测渔民",
            "direct_match": {"exists": False, "summary": "没有直接模型", "evidence": ["e1"]},
            "similar_assets": [
                {
                    "model_or_solution": "模型甲",
                    "label_schema": "100 类",
                    "training_data": "十万张",
                    "reported_metrics": "mAP 0.99",
                    "covered_capability": "完全覆盖",
                    "gap": "无差距",
                    "evidence": ["e1"],
                }
            ],
            "migration_plan": {
                "feasibility": "high",
                "approach": "直接上线",
                "data_requirements": "无需数据",
                "compute_requirements": "单卡",
                "engineering_work": "无需开发",
                "estimated_timeline": "2 周",
                "estimated_cost": "1 万元",
                "dependencies": [],
                "risks": [],
            },
            "expected_performance": {"baseline": "0.99", "target": "0.95", "uncertainty": "低"},
            "recommendation": "立即上线",
        }
        facts = [
            {
                "fact_type": "accuracy",
                "field": "performance_baseline",
                "subject": "模型甲",
                "fact": "在测试集上的 mAP 为 0.73",
                "value": "0.73",
                "unit": "mAP",
                "evidence_ids": ["e1"],
                "doc_ids": [],
                "quote": "模型甲在测试集上的 mAP 为 0.73。",
                "confidence": "high",
            }
        ]
        rex_result = {
            "success": True,
            "label": "渔民",
            "num_boxes": 2,
            "accuracy_estimate": {"accuracy": "90%", "reason": "视觉估计"},
        }

        bounded = self.module.enforce_report_evidence_bounds(
            report,
            facts,
            rex_result=rex_result,
        )

        self.assertIn("0.73", bounded["expected_performance"]["baseline"])
        self.assertNotIn("0.99", bounded["expected_performance"]["baseline"])
        self.assertIn("无人工 GT", bounded["expected_performance"]["target"])
        self.assertEqual(bounded["migration_plan"]["estimated_timeline"], "证据不足")
        self.assertEqual(bounded["migration_plan"]["estimated_cost"], "证据不足")
        self.assertTrue(bounded["recommendation"].startswith("工程推断"))

    def test_default_plan_respects_retrieval_budget(self) -> None:
        plan = self.module._fallback_plan("检测渔民")
        counts = [len(field["queries"]) for field in plan["retrieve_fields"]]
        self.assertTrue(all(count <= self.module.MIGRATION_QUERIES_PER_FIELD for count in counts))

    def test_direct_match_requires_task_object_and_scene_evidence(self) -> None:
        plan = {
            "abstract_requirement": {
                "object": "河道监控",
                "attribute": "钓鱼人员检测",
                "task_type": "目标检测",
                "scene": "户外水域",
                "constraints": [],
            }
        }
        report = {
            "requirement_summary": "fixture",
            "direct_match": {"exists": True, "summary": "fixture", "evidence": ["e1"]},
            "similar_assets": [],
            "migration_plan": {},
            "expected_performance": {},
            "recommendation": "fixture",
        }
        identity = {
            "fact_type": "model_identity",
            "subject": "fisher.model",
            "fact": "算法名称为钓鱼人体检测",
            "evidence_ids": ["e1"],
            "doc_ids": [],
        }
        generic_scene = {
            "fact_type": "task_scope",
            "subject": "fisher.model",
            "fact": "应用场景为户外园区",
            "evidence_ids": ["e1"],
            "doc_ids": [],
        }

        bounded = self.module.enforce_report_evidence_bounds(
            report,
            [identity, generic_scene],
            retrieval_plan=plan,
            user_query="河道钓鱼人员检测",
        )
        self.assertFalse(bounded["direct_match"]["exists"])

        river_scene = dict(generic_scene, fact="应用场景为河道户外水域")
        bounded = self.module.enforce_report_evidence_bounds(
            report,
            [identity, river_scene],
            retrieval_plan=plan,
            user_query="河道钓鱼人员检测",
        )
        self.assertTrue(bounded["direct_match"]["exists"])


if __name__ == "__main__":
    unittest.main()
