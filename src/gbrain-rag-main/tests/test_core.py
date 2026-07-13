import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from gbrain_rag.api import routes
from gbrain_rag.api.routes import _complete_known_field_answer
from gbrain_rag.api.routes import _complete_important_values_answer
from gbrain_rag.api.routes import _domain_alias_terms
from gbrain_rag.api.routes import _full_documents_for_evidences
from gbrain_rag.api.routes import _knowledge_base_fully_answered
from gbrain_rag.api.routes import _knowledge_base_fully_answered_confidence
from gbrain_rag.api.routes import _query_stream
from gbrain_rag.api.routes import _route_sources_with_llm
from gbrain_rag.api.routes import _source_balanced_results
from gbrain_rag.api.routes import _structured_answer_from_rows
from gbrain_rag.api.routes import _try_deterministic_aggregate
from gbrain_rag.api.routes import _try_structured_model_list
from gbrain_rag.api.schemas import EvidenceItem, QueryRequest, RetrieveRequest, RetrieveResponse, RoutePlanResponse
from gbrain_rag.core.types import Chunk, ScoredChunk
from gbrain_rag.ingest.pdf import table_to_structured_text
from gbrain_rag.llm.client import clean_query_expansion_terms, format_evidences
from gbrain_rag.retrieval.aspects import (
    ACCURACY_ASPECT,
    LIMITATION_ASPECT,
    answerability_score,
    classify_chunk_aspects,
)
from gbrain_rag.retrieval.embeddings import HashingEmbeddingBackend
from gbrain_rag.retrieval.query_understanding import build_query_intent, expand_query_terms
from gbrain_rag.retrieval.ranking import reciprocal_rank_fusion
from gbrain_rag.retrieval.service import RetrievalService, evidence_payload
from gbrain_rag.retrieval.store import BrainStore


class CoreTests(unittest.TestCase):
    def test_hashing_embedding_is_normalized_and_deterministic(self):
        backend = HashingEmbeddingBackend(dim=64)
        a = backend.encode_one("GenericModel v0.2.1 输出")
        b = backend.encode_one("GenericModel v0.2.1 输出")
        self.assertTrue(np.allclose(a, b))
        self.assertAlmostEqual(float(np.linalg.norm(a)), 1.0, places=5)

    def test_rrf_prefers_items_seen_in_multiple_lists(self):
        fused = reciprocal_rank_fusion(
            [
                [("a", 0.9), ("b", 0.8)],
                [("b", 0.7), ("c", 0.6)],
            ],
            k=60,
        )
        self.assertGreater(fused["b"], fused["c"])
        self.assertGreater(fused["b"], 0)

    def test_route_keeps_pdf_documents_for_feature_dimension_queries(self):
        service = RetrievalService()
        plan = service.route_sources("GenericEmbedding 模型的特征维度是多少？")
        self.assertTrue(plan.document)
        self.assertIn("document", plan.sources)

    def test_llm_route_defaults_to_all_sources_without_high_confidence_subset(self):
        request = RetrieveRequest(query="这个模型怎么样？")

        async def fake_plan_retrieval_sources(**_kwargs):
            return {
                "sources": ["document"],
                "reason": "问题比较泛，暂时偏向正文",
                "confidence": "medium",
            }

        with patch.object(routes, "plan_retrieval_sources", new=fake_plan_retrieval_sources):
            plan = asyncio.run(_route_sources_with_llm(request))

        self.assertEqual(plan.sources, ["document", "table", "adela"])
        self.assertIn("Default all enabled sources", plan.reason)

    def test_llm_route_uses_high_confidence_subset(self):
        request = RetrieveRequest(query="部署平台和 did 是什么？")

        async def fake_plan_retrieval_sources(**_kwargs):
            return {
                "sources": ["adela"],
                "reason": "问题明确聚焦部署信息",
                "confidence": "high",
            }

        with patch.object(routes, "plan_retrieval_sources", new=fake_plan_retrieval_sources):
            plan = asyncio.run(_route_sources_with_llm(request))

        self.assertEqual(plan.sources, ["adela"])
        self.assertEqual(plan.reason, "问题明确聚焦部署信息")

    def test_query_expansion_terms_use_only_explicit_extra_terms(self):
        terms = expand_query_terms("项目A检测用哪个模型？")
        self.assertIn("项目A检测用哪个模型？", terms)
        self.assertNotIn("project_alpha", terms)
        expanded = expand_query_terms("项目A检测用哪个模型？", extra_terms=["project_alpha"])
        self.assertIn("project_alpha", expanded)

    def test_resolve_query_expansion_merges_request_alias_and_llm_terms(self):
        async def fake_expand_query_with_llm(**_kwargs):
            return ["llm_term"]

        request = RetrieveRequest(
            query="安全绳检测在 T4 上有部署吗？",
            query_expansion_terms=["explicit_term"],
            expand_query_with_llm=True,
        )
        with patch.object(routes, "expand_query_with_llm", new=fake_expand_query_with_llm):
            terms, timing = asyncio.run(routes._resolve_query_expansion_terms(request))

        self.assertIn("explicit_term", terms)
        self.assertIn("safety_rope", terms)
        self.assertIn("llm_term", terms)
        self.assertEqual(timing["query_expansion_method"], "request+alias+llm")

    def test_feature_dimension_answer_keeps_extracted_families(self):
        evidence = EvidenceItem(
            evidence_id="e1",
            source_type="document",
            score=1.0,
            source_rank=1,
            source_score=1.0,
            title="GenericEmbedding",
            snippet="特征维度",
            doc_id="d1",
            doc_name="GenericEmbedding-V3.7.0.pdf",
            block_type="table",
            payload={
                "field_summary": (
                    "字段抽取:\n"
                    "- 模型族=Base224 | 特征维度=512\n"
                    "- 模型族=Large336 | 特征维度=768"
                )
            },
        )
        answer = _complete_known_field_answer(
            "GenericEmbedding 模型的特征维度是多少？",
            "GenericEmbedding 模型（generic-embedding-cn-base-224）的特征维度是 512。[证据1]",
            [evidence],
        )
        self.assertIn("特征维度是 512", answer)
        self.assertIn("Large336 的 特征维度 为 768[证据1]", answer)

    def test_model_record_completion_adds_structured_release_fields(self):
        evidence = EvidenceItem(
            evidence_id="e1",
            source_type="table",
            score=1.0,
            source_rank=1,
            source_score=1.0,
            title="Generic quality",
            snippet="模型名称: KM_generic_quality_T4.model",
            doc_id="d1",
            doc_name="model_release_records.jsonl",
            block_type="row",
            payload={
                "canonical_metadata": {
                    "row_id": "模型汇总源数据-0001",
                    "target_name": "人体",
                    "algorithm_type": "属性",
                    "algorithm_name": "人体质量",
                    "model_name": "KM_generic_quality_T4.model",
                    "oid": "abc123",
                    "supported_device": "T4",
                    "recommended_config": "cuda11-int8-T4",
                    "last_updated": "2026-01-01",
                },
                "count": "1",
            },
        )
        answer = routes._complete_model_record_answer(
            "人体质量过滤用哪个 T4 模型，主要判断哪些质量项？",
            "使用 KM_generic_quality_T4.model，判断模糊和完整性。[证据1]",
            [evidence],
        )
        self.assertIn("abc123", answer)
        self.assertIn("count=1", answer)

    def test_threshold_completion_uses_retrieved_threshold_column(self):
        evidence = EvidenceItem(
            evidence_id="e1",
            source_type="document",
            score=1.0,
            source_rank=1,
            source_score=1.0,
            title="metric table",
            snippet="模型版本 平台 阈值 precision recall V0.0.2 torch 0.2 0.97 0.98",
            doc_id="d1",
            doc_name="release.pdf",
            block_type="table",
            payload={},
        )
        answer = routes._complete_threshold_answer(
            "这个模型文档里推荐阈值多少？",
            "文档中未明确给出统一推荐阈值。",
            [evidence],
        )
        self.assertIn("阈值列/threshold 记录为 0.2", answer)

    def test_knowledge_base_fully_answered_fallback_detects_answerability(self):
        evidence = EvidenceItem(
            evidence_id="e1",
            source_type="document",
            score=1.0,
            source_rank=1,
            source_score=1.0,
            title="release note",
            snippet="top1 acc 从 0.55 提升到 0.97。",
            doc_id="d1",
            doc_name="generic.pdf",
            block_type="text",
            payload={"evidence_role": "primary"},
        )
        self.assertEqual(_knowledge_base_fully_answered("精度从 0.55 提升到 0.97。[证据1]", [evidence]), 0.7)
        self.assertEqual(
            _knowledge_base_fully_answered(
                "抱歉,您提问的相关信息在知识库中没有找到",
                [evidence],
            ),
            0.0,
        )
        self.assertEqual(_knowledge_base_fully_answered("精度从 0.55 提升到 0.97。", []), 0.0)

    def test_knowledge_base_fully_answered_does_not_reject_normal_caveat_phrases(self):
        evidence = EvidenceItem(
            evidence_id="e1",
            source_type="document",
            score=1.0,
            source_rank=1,
            source_score=1.0,
            title="release note",
            snippet="top1 acc 从 0.55 提升到 0.97，夜间场景无法保证。",
            doc_id="d1",
            doc_name="generic.pdf",
            block_type="text",
            payload={"evidence_role": "primary"},
        )
        answer = "当前版本 top1 acc 从 0.55 提升到 0.97，但夜间场景精度无法保证。[证据1]"
        self.assertEqual(_knowledge_base_fully_answered(answer, [evidence]), 0.7)

    def test_knowledge_base_fully_answered_confidence_uses_llm_score(self):
        evidence = EvidenceItem(
            evidence_id="e1",
            source_type="document",
            score=1.0,
            source_rank=1,
            source_score=1.0,
            title="release note",
            snippet="top1 acc 从 0.55 提升到 0.97。",
            doc_id="d1",
            doc_name="generic.pdf",
            block_type="text",
            payload={"evidence_role": "primary"},
        )

        async def fake_score_knowledge_base_fully_answered(*_args, **_kwargs):
            return 0.93

        with patch.object(
            routes,
            "score_knowledge_base_fully_answered",
            new=fake_score_knowledge_base_fully_answered,
        ):
            score = asyncio.run(
                _knowledge_base_fully_answered_confidence(
                    "这个模型精度如何？",
                    "精度从 0.55 提升到 0.97。[证据1]",
                    [evidence],
                    None,
                )
            )

        self.assertEqual(score, 0.93)

    def test_query_stream_sets_knowledge_base_fully_answered_from_final_answer(self):
        evidence = EvidenceItem(
            evidence_id="e1",
            source_type="document",
            score=1.0,
            source_rank=1,
            source_score=1.0,
            title="release note",
            snippet="top1 acc 从 0.55 提升到 0.97。",
            doc_id="d1",
            doc_name="generic.pdf",
            block_type="text",
            payload={"evidence_role": "primary"},
        )
        retrieved = RetrieveResponse(
            query="这个模型精度如何？",
            route_plan=RoutePlanResponse(
                document=True,
                table=False,
                adela=False,
                reason="test",
                sources=["document"],
            ),
            evidences=[evidence],
            timings={"retrieve_ms": 1.0},
            retrieved_count=1,
        )

        async def fake_stream_answer_with_llm(*_args, **_kwargs):
            for chunk in ("精度从 ", "0.55 提升到 0.97。[证据1]"):
                yield chunk

        async def fake_resolve_query_expansion_terms(_request):
            return [], {"query_expansion_method": "disabled"}

        async def fake_try_structured_aggregate(_request):
            return None

        async def fake_retrieve(_request):
            return retrieved

        async def fake_score_knowledge_base_fully_answered(*_args, **_kwargs):
            return 0.91

        async def collect_events():
            request = QueryRequest(query="这个模型精度如何？", stream=True, expand_query_with_llm=False)
            events = []
            async for event in _query_stream(request, 0.0):
                events.append(event)
            return events

        with patch.object(routes, "_resolve_query_expansion_terms", new=fake_resolve_query_expansion_terms):
            with patch.object(routes, "_try_structured_aggregate", new=fake_try_structured_aggregate):
                with patch.object(routes, "_retrieve", new=fake_retrieve):
                    with patch.object(routes, "stream_answer_with_llm", new=fake_stream_answer_with_llm):
                        with patch.object(
                            routes,
                            "score_knowledge_base_fully_answered",
                            new=fake_score_knowledge_base_fully_answered,
                        ):
                            events = asyncio.run(collect_events())

        final_payload = json.loads(events[-2][len("data: "):].strip())
        self.assertEqual(final_payload["knowledge_base_fully_answered"], 0.91)

    def test_generic_field_completion_handles_multiple_families(self):
        evidence = EvidenceItem(
            evidence_id="e1",
            source_type="document",
            score=1.0,
            source_rank=1,
            source_score=1.0,
            title="Generic model table",
            snippet="模型文件列表",
            doc_id="d1",
            doc_name="generic.pdf",
            block_type="table",
            payload={
                "field_summary": (
                    "字段抽取:\n"
                    "- 模型族=Small | 模型名称=KM_generic_small.model | 特征维度=128\n"
                    "- 模型族=Large | 模型名称=KM_generic_large.model | 特征维度=256"
                )
            },
        )
        answer = _complete_known_field_answer(
            "这个模型的特征维度是多少？",
            "Small 模型的特征维度是 128。[证据1]",
            [evidence],
        )
        self.assertIn("Large 的 特征维度 为 256[证据1]", answer)

    def test_source_balanced_results_keeps_each_source_visible(self):
        def scored(chunk_id: str, source_type: str, score: float) -> ScoredChunk:
            return ScoredChunk(
                chunk=Chunk(
                    chunk_id=chunk_id,
                    doc_id=chunk_id,
                    doc_name=f"{source_type}.txt",
                    source_type=source_type,
                    title=chunk_id,
                    text=chunk_id,
                    index_text=chunk_id,
                    block_type="text",
                ),
                score=score,
                source_rank=1,
                source_score=score,
            )

        ranked = [
            scored("d1", "document", 1.0),
            scored("d2", "document", 0.9),
            scored("d3", "document", 0.8),
            scored("t1", "table", 0.7),
            scored("a1", "adela", 0.6),
        ]
        balanced = _source_balanced_results(ranked, ["document", "table", "adela"], top_k=12)
        self.assertEqual([item.chunk.source_type for item in balanced[:3]], ["document", "table", "adela"])
        self.assertEqual([item.chunk.chunk_id for item in balanced[3:5]], ["d2", "d3"])

    def test_structured_count_answer_reports_records_models_without_count_field(self):
        rows = [
            {"target_name": "目标A", "algorithm_name": "目标A检测", "model_name": "m1", "计数": 1},
            {"target_name": "目标A", "algorithm_name": "目标A检测", "model_name": "m2", "计数": 0},
            {"target_name": "目标A", "algorithm_name": "目标A特征", "model_name": "m2", "计数": 1},
        ]
        answer = _structured_answer_from_rows("目标A相关的模型有多少个", "table", rows)
        self.assertIn("3 条模型记录", answer)
        self.assertIn("2 个模型", answer)
        self.assertNotIn("count=1", answer)
        self.assertNotIn("现用/发布记录", answer)

    def test_structured_model_list_answers_from_table_rows(self):
        class FakeStore:
            def load_metadata_rows(self, source_type):
                if source_type != "table":
                    return []
                return [
                    {
                        "row_id": "r1",
                        "target_name": "烟火",
                        "algorithm_type": "检测",
                        "algorithm_name": "烟火三类检测",
                        "model_name": "KM_essos_det_small_nart_cuda11.0-trt7.1-int8-T4_b8_1.0.0.model",
                        "oid": "oid-1",
                        "supported_device": "T4",
                        "recommended_config": "cuda11.0-trt7.1-int8-T4",
                        "source_file": "data_source/模型发版记录汇总.xlsx",
                        "ones_release_link": "https://ones.example/r1",
                        "sheet_name": "模型汇总源数据",
                        "source_row_number": 289,
                    },
                    {
                        "row_id": "r2",
                        "target_name": "烟火",
                        "algorithm_type": "检测",
                        "algorithm_name": "烟火三类检测",
                        "model_name": "KM_essos_det_small_nart_acl-ascend710-fp16_b1_1.0.0.model",
                        "oid": "oid-2",
                        "supported_device": "ascend710",
                        "recommended_config": "acl-ascend710-fp16",
                        "source_file": "data_source/模型发版记录汇总.xlsx",
                        "ones_release_link": "https://ones.example/r2",
                        "sheet_name": "模型汇总源数据",
                        "source_row_number": 290,
                    },
                    {
                        "row_id": "r3",
                        "target_name": "安全绳",
                        "algorithm_type": "检测",
                        "algorithm_name": "安全绳检测",
                        "model_name": "safety_rope.model",
                    },
                ]

        class FakeService:
            store = FakeStore()

        request = RetrieveRequest(
            query="烟火检测有什么模型？",
            sources=["table"],
            expand_query_with_llm=False,
            query_expansion_terms=["fire", "smoke", "det_small_fire"],
        )
        with patch.object(routes, "get_retrieval_service", return_value=FakeService()):
            result = _try_structured_model_list(request)

        self.assertIsNotNone(result)
        retrieved, answer = result
        self.assertEqual(retrieved.retrieved_count, 2)
        self.assertIn("KM_essos_det_small_nart_cuda11.0-trt7.1-int8-T4_b8_1.0.0.model", answer)
        self.assertIn("KM_essos_det_small_nart_acl-ascend710-fp16_b1_1.0.0.model", answer)
        self.assertNotIn("safety_rope", answer)
        self.assertEqual(retrieved.evidences[0].reference_url, "https://ones.example/r1")
        self.assertIn("模型发版记录汇总.xlsx#模型汇总源数据!row=289", retrieved.evidences[0].doc_name)
        self.assertNotEqual(retrieved.evidences[0].doc_name, "structured_table")

    def test_structured_model_list_skips_recommendation_questions(self):
        request = RetrieveRequest(
            query="烟火检测现在应该用哪个 T4 模型？",
            sources=["table"],
            expand_query_with_llm=False,
            query_expansion_terms=["fire", "smoke", "det_small_fire"],
        )
        self.assertIsNone(_try_structured_model_list(request))

    def test_domain_alias_terms_expand_to_generic_domain_terms(self):
        self.assertIn("glove", _domain_alias_terms("行人黄手套识别优化具体是什么？"))
        aliases = _domain_alias_terms("厨房场景有没有更细的物品识别部署？")
        self.assertIn("kitchen", aliases)
        self.assertIn("item", aliases)
        self.assertNotIn("kitchen_item", aliases)
        self.assertIn("struct", _domain_alias_terms("城市结构化检测最新版本到底以哪个来源为准？"))

    def test_deterministic_deployment_status_aggregate(self):
        test_case = self

        class FakeStore:
            def load_metadata_rows(self, source_type):
                test_case.assertEqual(source_type, "adela")
                return [
                    {"model_name": "m1", "status": "SUCCESS", "did": "1", "rid": "10"},
                    {"model_name": "m1", "status": "SUCCESS", "did": "2", "rid": "20"},
                    {"model_name": "m2", "status": "SUCCESS", "did": "3", "rid": "30"},
                ]

        class FakeService:
            store = FakeStore()

        request = RetrieveRequest(
            query="按这批部署资料，线上部署规模大概是多少，状态怎么样？",
            sources=["adela"],
            expand_query_with_llm=False,
        )
        with patch.object(routes, "get_retrieval_service", return_value=FakeService()):
            result = _try_deterministic_aggregate(request)

        self.assertIsNotNone(result)
        retrieved, answer = result
        self.assertIn("共 3 条部署记录", answer)
        self.assertIn("SUCCESS 3", answer)
        self.assertIn("2 个模型名", answer)
        self.assertEqual(retrieved.evidences[0].source_type, "adela")
        self.assertIn("model_name", retrieved.evidences[0].snippet)

    def test_deterministic_hardware_distribution_uses_current_table_rows(self):
        test_case = self

        class FakeStore:
            def load_metadata_rows(self, source_type):
                test_case.assertEqual(source_type, "table")
                return [
                    {"model_name": "m1", "supported_device": "T4", "count": "1"},
                    {"model_name": "m2", "supported_device": "T4", "count": "1"},
                    {"model_name": "m3", "supported_device": "ascend710", "count": "1"},
                    {"model_name": "old", "supported_device": "P4", "count": "0"},
                ]

        class FakeService:
            store = FakeStore()

        request = RetrieveRequest(
            query="当前模型主要适配哪些硬件？",
            sources=["table"],
            expand_query_with_llm=False,
        )
        with patch.object(routes, "get_retrieval_service", return_value=FakeService()):
            result = _try_deterministic_aggregate(request)

        self.assertIsNotNone(result)
        retrieved, answer = result
        self.assertIn("T4 2 条", answer)
        self.assertIn("ascend710 1 条", answer)
        self.assertNotIn("P4", answer)
        self.assertEqual(retrieved.evidences[0].source_type, "table")

    def test_format_evidences_surfaces_release_note_metrics_and_labels(self):
        evidence = EvidenceItem(
            evidence_id="e1",
            source_type="document",
            score=1.0,
            source_rank=1,
            source_score=1.0,
            title="release note",
            snippet="追加了回流误报图片，标签解释见全文。",
            doc_id="d1",
            doc_name="release.pdf",
            block_type="text",
            payload={
                "index_text": (
                    "追加了长客场景回流误报的工人图片，top1 acc从原先 0.55提升到0.97。"
                    "标签解释: \"label_connected\"、\"label_unconnected\"。"
                )
            },
        )
        formatted = format_evidences([evidence])
        self.assertIn("important_values", formatted)
        self.assertIn("0.55提升到0.97", formatted)
        self.assertIn('"label_connected"', formatted)

    def test_clean_query_expansion_terms_filters_generic_noise(self):
        terms = clean_query_expansion_terms(
            ["项目A", "模型", "CNN", "目标检测", "project_alpha", "业务属性模型"],
            query="项目A检测用哪个模型？",
            max_terms=8,
        )
        self.assertEqual(terms, ["project_alpha", "业务属性模型"])

    def test_query_intent_distinguishes_accuracy_from_limitations(self):
        accuracy = build_query_intent("安全绳检测模型的精度如何")
        limitation = build_query_intent("安全绳检测模型在哪些情况下精度无法保证")
        self.assertEqual(accuracy.aspect, ACCURACY_ASPECT)
        self.assertEqual(accuracy.answer_type, "evaluation_summary")
        self.assertEqual(limitation.aspect, LIMITATION_ASPECT)
        self.assertEqual(limitation.answer_type, "limitation_summary")

    def test_answerability_reranks_accuracy_over_boundary_for_same_entity(self):
        metric_chunk = Chunk(
            chunk_id="metric",
            doc_id="d1",
            doc_name="generic_model.pdf",
            source_type="document",
            title="release note",
            text="Highlight: top1 acc 从 0.55 提升到 0.97，通用测试集各项精度指标提升 0.1%-0.5%。",
            index_text="Highlight: top1 acc 从 0.55 提升到 0.97，通用测试集各项精度指标提升 0.1%-0.5%。",
            block_type="text",
        )
        boundary_chunk = Chunk(
            chunk_id="boundary",
            doc_id="d1",
            doc_name="generic_model.pdf",
            source_type="document",
            title="算法边界",
            text="场景要求：夜间、强逆光、雨雪雾等特殊场景下精度无法保证。",
            index_text="场景要求：夜间、强逆光、雨雪雾等特殊场景下精度无法保证。",
            block_type="text",
        )
        metric_score, metric_meta = answerability_score(ACCURACY_ASPECT, metric_chunk)
        boundary_score, boundary_meta = answerability_score(ACCURACY_ASPECT, boundary_chunk)
        self.assertIn(ACCURACY_ASPECT, classify_chunk_aspects(metric_chunk))
        self.assertIn(LIMITATION_ASPECT, classify_chunk_aspects(boundary_chunk))
        self.assertGreater(metric_score, boundary_score)
        self.assertEqual(metric_meta["evidence_role"], "primary")
        self.assertEqual(boundary_meta["evidence_role"], "caveat")

    def test_format_evidences_groups_primary_and_caveat(self):
        primary = EvidenceItem(
            evidence_id="e1",
            source_type="document",
            score=1.0,
            source_rank=1,
            source_score=1.0,
            title="release note",
            snippet="top1 acc 从 0.55 提升到 0.97。",
            doc_id="d1",
            doc_name="generic.pdf",
            block_type="text",
            payload={
                "query_aspect": ACCURACY_ASPECT,
                "aspects": [ACCURACY_ASPECT],
                "section_type": "release_note",
                "evidence_role": "primary",
            },
        )
        caveat = EvidenceItem(
            evidence_id="e2",
            source_type="document",
            score=0.9,
            source_rank=2,
            source_score=0.9,
            title="算法边界",
            snippet="夜间场景精度无法保证。",
            doc_id="d1",
            doc_name="generic.pdf",
            block_type="text",
            payload={
                "query_aspect": ACCURACY_ASPECT,
                "aspects": [LIMITATION_ASPECT],
                "section_type": "algorithm_boundary",
                "evidence_role": "caveat",
            },
        )
        formatted = format_evidences([primary, caveat])
        self.assertIn("<primary_evidence>", formatted)
        self.assertIn("<caveat_evidence>", formatted)
        self.assertIn("evidence_role: primary", formatted)
        self.assertIn("evidence_role: caveat", formatted)

    def test_important_values_complete_metric_answers_generically(self):
        evidence = EvidenceItem(
            evidence_id="e1",
            source_type="document",
            score=1.0,
            source_rank=1,
            source_score=1.0,
            title="release note",
            snippet="主要优化品牌和否定描述。",
            doc_id="d1",
            doc_name="release.pdf",
            block_type="text",
            payload={
                "index_text": (
                    "base 模型在行人品牌 mAP 提升 15.02%，车辆品牌 mAP 提升9.68%。"
                    "large 模型在行人品牌 mAP 提升 10.79%，重点检索词 mAP 提升21.77%。"
                )
            },
        )
        answer = _complete_important_values_answer(
            "这个版本相比旧版本主要优化了哪些能力？",
            "主要优化品牌和否定描述。",
            [evidence],
        )
        self.assertIn("15.02%", answer)
        self.assertIn("10.79%", answer)
        self.assertEqual(answer.count("[证据1]"), 2)

    def test_structured_row_evidence_uses_real_reference_fields(self):
        row = {
            "chunk_id": "table-chunk-1",
            "row_id": "模型汇总源数据-0289",
            "target_name": "烟火",
            "algorithm_type": "检测",
            "algorithm_name": "烟火三类检测",
            "model_name": "KM_fire.model",
            "oid": "oid-fire",
            "supported_device": "T4",
            "recommended_config": "cuda11.0-trt7.1-int8-T4",
            "source_file": "data_source/模型发版记录汇总.xlsx",
            "ones_release_link": "https://ones.example/page/fire",
            "sheet_name": "模型汇总源数据",
            "source_row_number": 289,
        }
        evidence = routes._row_to_evidence(row, "table", 1, 1.0)
        self.assertEqual(evidence.evidence_id, "table-chunk-1")
        self.assertEqual(evidence.reference_url, "https://ones.example/page/fire")
        self.assertIn("模型发版记录汇总.xlsx#模型汇总源数据!row=289", evidence.doc_name)
        self.assertNotEqual(evidence.doc_name, "structured_table")

    def test_pdf_table_evidence_payload_extracts_requested_fields(self):
        chunk = Chunk(
            chunk_id="c1",
            doc_id="d1",
            doc_name="generic.pdf",
            source_type="document",
            title="generic p1 table 1",
            text=(
                "表格结构化行:\n"
                "模型族: Small\n"
                "模型名称: KM_generic_small.model\n"
                "平台: cuda11.0-trt7.1-fp16-T4\n"
                "特征维度: 128\n\n"
                "模型族: Large\n"
                "模型名称: KM_generic_large.model\n"
                "平台: cuda11.0-trt7.1-fp32-P4\n"
                "特征维度: 256"
            ),
            index_text=(
                "表格结构化行:\n"
                "模型族: Small\n"
                "模型名称: KM_generic_small.model\n"
                "平台: cuda11.0-trt7.1-fp16-T4\n"
                "特征维度: 128\n\n"
                "模型族: Large\n"
                "模型名称: KM_generic_large.model\n"
                "平台: cuda11.0-trt7.1-fp32-P4\n"
                "特征维度: 256"
            ),
            block_type="table",
        )
        payload = evidence_payload(
            ScoredChunk(
                chunk=chunk,
                score=1.0,
                source_rank=1,
                source_score=1.0,
            ),
            "这个模型的特征维度和平台是什么？",
        )["payload"]
        self.assertIn("模型族=Small", payload["field_summary"])
        self.assertIn("平台=cuda11.0-trt7.1-fp32-P4", payload["field_summary"])
        self.assertIn("特征维度=256", payload["field_summary"])

    def test_sparse_pdf_model_table_keeps_feature_dimension_binding(self):
        table = [
            ["", "", "模型名称", "", "组件类型", "", "", "", "OID", "", "", "平台", "", "", "", "特征维度"],
            ["", "", "Base224", "", "", "", "", "", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "senu", "", "", "", "", "", "", "acl-ascend710-fp16", "", "", "", "512"],
            [
                "",
                "",
                "KM_generic-embedding-cn-base-224- image_nart_acl-ascend710- fp16_b1_3.7.0.model",
                "",
                "",
                "",
                "",
                "",
                "b712b016e3b496f9dcf60143cf7144f c611d0063cac93f46b41044c1a9a56319",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ],
        ]
        text = table_to_structured_text(table)
        self.assertIn("模型族: Base224", text)
        self.assertIn("组件类型: senu", text)
        self.assertIn("平台: acl-ascend710-fp16", text)
        self.assertIn("特征维度: 512", text)
        self.assertIn("KM_generic-embedding-cn-base-224-image_nart_acl-ascend710-fp16_b1_3.7.0.model", text)

    def test_store_and_retrieval_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = BrainStore(Path(tmp) / "brain.sqlite3")
            backend = HashingEmbeddingBackend(model_name="hashing", dim=64)
            chunks = [
                Chunk(
                    chunk_id="c1",
                    doc_id="d1",
                    doc_name="project_model v0.2.1.pdf",
                    source_type="document",
                    title="项目模型",
                    text="项目检测模型 v0.2.1 输出：单张小图的预测分类结果。",
                    index_text="项目检测模型 v0.2.1 输出：单张小图的预测分类结果。",
                    block_type="text",
                ),
                Chunk(
                    chunk_id="c2",
                    doc_id="d2",
                    doc_name="adela.jsonl",
                    source_type="adela",
                    title="BannerSlogan",
                    text="model_name: BannerSlogan platform: cuda11.0-trt7.1-fp16-T4 did: 129623",
                    index_text="model_name: BannerSlogan platform: cuda11.0-trt7.1-fp16-T4 did: 129623",
                    block_type="row",
                ),
            ]
            for chunk in chunks:
                store.upsert_chunk(chunk, backend.encode_one(chunk.text), "hashing")
            store.commit()

            service = RetrievalService(store=store)
            results, timings = service.retrieve(
                query="项目模型 v0.2.1 的输出是什么？",
                source_types=["document"],
                retrieval_method="hybrid",
                embedding_model="hashing",
                embedding_backend="hashing",
                top_k=3,
            )
            self.assertTrue(results)
            self.assertEqual(results[0].chunk.chunk_id, "c1")
            self.assertIn("retrieve_ms", timings)

    def test_full_documents_deduplicates_evidence_doc_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = BrainStore(Path(tmp) / "brain.sqlite3")
            chunks = [
                Chunk(
                    chunk_id="c1",
                    doc_id="d1",
                    doc_name="release.pdf",
                    source_type="document",
                    title="release p1",
                    text="第一页内容。",
                    index_text="第一页内容。",
                    block_type="text",
                    page_label=1,
                    metadata={"part": 0},
                ),
                Chunk(
                    chunk_id="c2",
                    doc_id="d1",
                    doc_name="release.pdf",
                    source_type="document",
                    title="release p2",
                    text="第二页内容。",
                    index_text="第二页内容。",
                    block_type="text",
                    page_label=2,
                    metadata={"part": 0},
                ),
            ]
            for chunk in chunks:
                store.upsert_chunk(chunk)
            store.commit()

            evidence_1 = EvidenceItem(
                evidence_id="c1",
                source_type="document",
                score=1.0,
                source_rank=1,
                source_score=1.0,
                title="release p1",
                snippet="第一页内容。",
                doc_id="d1",
                doc_name="release.pdf",
                block_type="text",
            )
            evidence_2 = EvidenceItem(
                evidence_id="c2",
                source_type="document",
                score=0.9,
                source_rank=2,
                source_score=0.9,
                title="release p2",
                snippet="第二页内容。",
                doc_id="d1",
                doc_name="release.pdf",
                block_type="text",
            )

            request = RetrieveRequest(query="release", include_full_documents=True)
            with patch.object(routes, "get_retrieval_service", return_value=RetrievalService(store=store)):
                full_documents = _full_documents_for_evidences(request, [evidence_1, evidence_2])

            self.assertEqual(len(full_documents), 1)
            self.assertEqual(full_documents[0].doc_id, "d1")
            self.assertEqual(full_documents[0].chunk_count, 2)
            self.assertIn("第一页内容。", full_documents[0].content)
            self.assertIn("第二页内容。", full_documents[0].content)
            self.assertEqual(full_documents[0].metadata["matched_evidence_ids"], ["c1", "c2"])

    def test_load_metadata_rows_prefers_canonical_structured_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = BrainStore(Path(tmp) / "brain.sqlite3")
            rows = [
                Chunk(
                    chunk_id="table_csv",
                    doc_id="csv",
                    doc_name="模型发版记录汇总.xlsx",
                    source_type="table",
                    title="m1",
                    text="m1",
                    index_text="m1",
                    block_type="row",
                    source_path="/tmp/模型发版记录汇总.xlsx",
                    metadata={"model_name": "m1"},
                ),
                Chunk(
                    chunk_id="table_jsonl",
                    doc_id="jsonl",
                    doc_name="tables/model_release_records.jsonl",
                    source_type="table",
                    title="m2",
                    text="m2",
                    index_text="m2",
                    block_type="row",
                    source_path="/tmp/tables/model_release_records.jsonl",
                    metadata={"model_name": "m2"},
                ),
            ]
            for chunk in rows:
                store.upsert_chunk(chunk)
            store.commit()
            metadata_rows = store.load_metadata_rows("table")
            self.assertEqual([row["model_name"] for row in metadata_rows], ["m2"])


if __name__ == "__main__":
    unittest.main()
