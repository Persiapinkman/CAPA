import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from ace_rag.core.config import get_settings
from ace_rag.api.schemas import FeedbackRequest, PlaybookItem, PlaybookOrganizeRequest, QueryRequest, RetrieveRequest
from ace_rag.ace.service import AceService
from ace_rag.playbook.store import PlaybookStore, collect_expansion_terms, collect_source_hints


class FakeV2Client:
    def __init__(self, evidences=None, query_answer=None):
        self.last_payload = None
        self.last_query_payload = None
        self.evidences = evidences or []
        self.query_answer = query_answer

    async def retrieve(self, payload):
        self.last_payload = payload
        return {
            "query": payload["query"],
            "route_plan": {
                "document": "document" in payload.get("sources", []),
                "table": "table" in payload.get("sources", []),
                "adela": "adela" in payload.get("sources", []),
                "reason": "fake",
                "sources": payload.get("sources", []),
            },
            "evidences": self.evidences,
            "timings": {"fake_ms": 1},
            "retrieved_count": len(self.evidences),
        }

    async def query(self, payload):
        self.last_query_payload = payload
        return {
            "query": payload["query"],
            "route_plan": {
                "document": "document" in payload.get("sources", []),
                "table": "table" in payload.get("sources", []),
                "adela": "adela" in payload.get("sources", []),
                "reason": "fake-query",
                "sources": payload.get("sources", []),
            },
            "evidences": self.evidences,
            "timings": {"fake_query_ms": 2},
            "retrieved_count": len(self.evidences),
            "answer": self.query_answer or "",
            "llm_config": {"mode": "fake_v2_query"},
        }


class FailingV2Client:
    async def retrieve(self, payload):
        raise AssertionError("v2 should not be called")


class PlaybookTests(unittest.IsolatedAsyncioTestCase):
    def make_store(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return PlaybookStore(Path(tmp.name) / "playbook.sqlite3")

    def make_settings(self, **overrides):
        values = get_settings().model_dump()
        values.update(overrides)
        return get_settings().__class__(**values)

    def test_search_prefers_deployment_rule(self):
        store = self.make_store()
        store.upsert_item(
            PlaybookItem(
                item_id="pb-source-1",
                section="source_routing",
                content="询问 did rid 部署 平台 时包含 adela",
                source_hints=["adela"],
                query_intents=["deployment"],
                expansion_terms=["did", "rid"],
                confidence=0.9,
            )
        )
        hits = store.search("这个模型在 T4 上有部署吗 did 是多少", top_k=3)
        self.assertEqual(hits[0].item_id, "pb-source-1")
        self.assertIn("adela", collect_source_hints(hits))
        self.assertNotIn("did", collect_expansion_terms(hits))
        self.assertGreater(hits[0].score_details["bm25"], 0)
        self.assertGreater(hits[0].score_details["keyword"], 0)

    def test_search_uses_keyword_terms_without_embedding(self):
        store = self.make_store()
        store.upsert_item(
            PlaybookItem(
                item_id="pb-alias",
                section="query_expansion",
                content="安全绳相关问题通常需要同时检索 safety_rope",
                source_hints=["document", "table", "adela"],
                query_intents=["deployment"],
                expansion_terms=["safety_rope"],
                tags=["alias", "safety"],
                confidence=0.9,
            )
        )
        store.upsert_item(
            PlaybookItem(
                item_id="pb-other",
                section="answer_strategy",
                content="证据不足时回答知识库中没有找到",
                query_intents=["field_lookup"],
                confidence=0.99,
            )
        )
        hits = store.search("safety_rope 在 T4 上部署了吗", top_k=2)
        self.assertEqual(hits[0].item_id, "pb-alias")
        self.assertGreater(hits[0].score_details["keyword"], 0)

    def test_search_matches_chinese_phrase_ngrams(self):
        store = self.make_store()
        store.upsert_item(
            PlaybookItem(
                item_id="pb-safety-rope",
                section="query_expansion",
                content="安全绳相关问题通常需要同时检索 safety_rope",
                source_hints=["document", "table", "adela"],
                query_intents=["deployment"],
                expansion_terms=["safety_rope"],
                tags=["alias", "safety"],
                confidence=0.9,
            )
        )
        hits = store.search("安全绳检测在 T4 上有部署吗", top_k=3)
        self.assertEqual(hits[0].item_id, "pb-safety-rope")
        self.assertGreater(hits[0].score_details["bm25"], 0)
        self.assertIn("safety_rope", collect_expansion_terms(hits))

    def test_collect_expansion_terms_ignores_intent_only_alias_hits(self):
        store = self.make_store()
        store.upsert_item(
            PlaybookItem(
                item_id="pb-safety-alias",
                section="query_expansion",
                content="安全绳相关问题通常需要同时检索 safety_rope",
                source_hints=["document", "table", "adela"],
                query_intents=["deployment"],
                expansion_terms=["safety_rope"],
                tags=["alias", "safety"],
                confidence=0.9,
            )
        )
        hits = store.search("厨房场景有没有更细的物品识别部署？", top_k=3)
        self.assertEqual(hits[0].item_id, "pb-safety-alias")
        self.assertEqual(hits[0].score_details["bm25"], 0)
        self.assertEqual(hits[0].score_details["keyword"], 0)
        self.assertNotIn("safety_rope", collect_expansion_terms(hits))

    def test_collect_expansion_terms_ignores_generic_retrieval_overlap(self):
        store = self.make_store()
        store.upsert_item(
            PlaybookItem(
                item_id="pb-safety-alias",
                section="query_expansion",
                content="安全绳相关问题通常需要同时检索 safety_rope；安全带/反光衣相关问题通常需要检索 safetybelt、waistcoat。",
                source_hints=["document", "table", "adela"],
                query_intents=["deployment", "field_lookup", "document_detail"],
                expansion_terms=["safety_rope", "safetybelt", "waistcoat"],
                tags=["alias", "safety"],
                confidence=0.9,
            )
        )
        query = "多模态视频检索在 T4 上需要部署哪几个模型？"
        hits = store.search(query, top_k=3)
        self.assertEqual(hits[0].item_id, "pb-safety-alias")
        self.assertGreater(hits[0].score_details["bm25"], 0)
        self.assertNotIn("safety_rope", collect_expansion_terms(hits, query=query))
        self.assertNotIn("waistcoat", collect_expansion_terms(hits, query=query))

    def test_collect_expansion_terms_keeps_matching_chinese_alias(self):
        store = self.make_store()
        store.upsert_item(
            PlaybookItem(
                item_id="pb-safety-alias",
                section="query_expansion",
                content="安全绳相关问题通常需要同时检索 safety_rope",
                source_hints=["document", "table", "adela"],
                query_intents=["deployment"],
                expansion_terms=["safety_rope"],
                tags=["alias", "safety"],
                confidence=0.9,
            )
        )
        query = "安全绳检测在 T4 上有部署吗？"
        hits = store.search(query, top_k=3)
        self.assertIn("safety_rope", collect_expansion_terms(hits, query=query))

    async def test_service_merges_playbook_hints_into_v2_request(self):
        store = self.make_store()
        store.upsert_item(
            PlaybookItem(
                item_id="pb-source-1",
                section="source_routing",
                content="询问 did rid 部署 平台 时包含 adela 和 table",
                source_hints=["adela", "table"],
                query_intents=["deployment"],
                expansion_terms=["did", "rid"],
                confidence=0.9,
            )
        )
        fake = FakeV2Client()
        service = AceService(store=store, v2_client=fake, auto_import_seed=False)
        response = await service.retrieve(RetrieveRequest(query="安全绳检测在 T4 上有部署吗？did 是多少？"))
        self.assertEqual(response.v2_request["sources"], ["adela", "table"])
        self.assertTrue(response.v2_request["expand_query_with_llm"])
        self.assertNotIn("did", response.v2_request["query_expansion_terms"])
        self.assertEqual(response.playbook.items[0].item_id, "pb-source-1")

    async def test_playbook_only_retrieve_skips_v2(self):
        store = self.make_store()
        store.upsert_item(
            PlaybookItem(
                item_id="pb-source-1",
                section="source_routing",
                content="询问 did rid 部署 平台 时包含 adela",
                source_hints=["adela"],
                query_intents=["deployment"],
                expansion_terms=["did", "rid"],
                confidence=0.9,
            )
        )
        service = AceService(store=store, v2_client=FailingV2Client(), auto_import_seed=False)
        response = await service.retrieve(
            RetrieveRequest(
                query="安全绳检测在 T4 上有部署吗？did 是多少？",
                playbook_only=True,
            )
        )
        self.assertEqual(response.evidences, [])
        self.assertEqual(response.retrieved_count, 0)
        self.assertEqual(response.route_plan.sources, ["adela"])
        self.assertTrue(response.timings["playbook_only"])
        self.assertTrue(response.v2_request["_skipped"])
        self.assertEqual(response.v2_request["_skip_reason"], "playbook_only")
        self.assertEqual(response.playbook.items[0].item_id, "pb-source-1")

    def test_feedback_creates_pending_operation(self):
        store = self.make_store()
        feedback_id, op_id = store.insert_feedback(
            FeedbackRequest(
                run_id="run-1",
                feedback_type="correction",
                rating=2,
                comment="需要包含 adela",
            )
        )
        self.assertTrue(feedback_id.startswith("fb-"))
        self.assertTrue(op_id and op_id.startswith("op-"))

    def test_feedback_creates_online_feedback_playbook_item(self):
        store = self.make_store()
        run_id = store.insert_run(
            query="通用车牌文字识别在 T4 上现在用哪个模型？",
            request={},
            v2_request={"sources": ["table"]},
            v2_response={},
            playbook_item_ids=[],
            answer="错误答案",
            timings={},
        )
        feedback_id, _op_id = store.insert_feedback(
            FeedbackRequest(
                run_id=run_id,
                feedback_type="correction",
                corrected_answer="正确答案是 KM_textdetection_HoriTextDetection_Plate_cuda_nart_cuda11.0-trt7.1-int8-T4_b8_3.1.0.model",
                expected_evidence_ids=["table::模型汇总源数据-0135"],
                comment="不要把检测答成识别",
            )
        )
        hits = store.search("通用车牌文字识别在 T4 上现在用哪个模型？", top_k=5)
        online_hit = next(hit for hit in hits if hit.section == "online_feedback")
        self.assertEqual(online_hit.provenance["feedback_id"], feedback_id)
        self.assertIn("table", online_hit.source_hints)
        self.assertIn(
            "KM_textdetection_HoriTextDetection_Plate_cuda_nart_cuda11.0-trt7.1-int8-T4_b8_3.1.0.model",
            online_hit.expansion_terms,
        )
        self.assertEqual(online_hit.score_details["exact_query_match"], 1.0)
        unrelated_hits = store.search("我想做横幅/标语检测，T4 上现在推荐用哪个模型？", top_k=5)
        self.assertFalse(any(hit.section == "online_feedback" for hit in unrelated_hits))

    def test_organize_playbook_returns_memory_candidates(self):
        store = self.make_store()
        store.upsert_item(
            PlaybookItem(
                item_id="pb-source-1",
                section="source_routing",
                content="询问 did 时包含 adela",
                source_hints=["adela"],
                query_intents=["deployment"],
                expansion_terms=["did"],
                tags=["deployment"],
                confidence=0.9,
            )
        )
        store.upsert_item(
            PlaybookItem(
                item_id="pb-source-2",
                section="source_routing",
                content="询问 rid 时包含 adela",
                source_hints=["adela"],
                query_intents=["deployment"],
                expansion_terms=["rid"],
                tags=["deployment"],
                confidence=0.8,
            )
        )
        fake = FakeV2Client()
        service = AceService(store=store, v2_client=fake, auto_import_seed=False)
        response = service.organize_playbook(PlaybookOrganizeRequest())
        self.assertEqual(response.item_count, 2)
        self.assertTrue(response.strategies)
        self.assertEqual(response.candidates[0].strategy, "semantic_merge_and_deduplicate")
        self.assertEqual(response.candidates[0].item_ids, ["pb-source-1", "pb-source-2"])

    async def test_query_uses_online_feedback_override_for_exact_query(self):
        store = self.make_store()
        run_id = store.insert_run(
            query="通用车牌文字识别在 T4 上现在用哪个模型？",
            request={},
            v2_request={"sources": ["table"]},
            v2_response={},
            playbook_item_ids=[],
            answer="错误答案",
            timings={},
        )
        store.insert_feedback(
            FeedbackRequest(
                run_id=run_id,
                feedback_type="correction",
                corrected_answer="正确答案应为 KM_textdetection_HoriTextDetection_Plate_cuda_nart_cuda11.0-trt7.1-int8-T4_b8_3.1.0.model。",
                expected_evidence_ids=["table::模型汇总源数据-0135"],
                comment="不要把检测答成识别",
            )
        )
        fake = FakeV2Client(
            evidences=[
                {
                    "evidence_id": "ev-1",
                    "legacy_evidence_id": "table::模型汇总源数据-0135",
                    "source_type": "table",
                    "score": 0.9,
                    "source_rank": 1,
                    "source_score": 0.9,
                    "title": "KM_textdetection_HoriTextDetection_Plate_cuda_nart_cuda11.0-trt7.1-int8-T4_b8_3.1.0.model",
                    "snippet": "target_name: 车牌",
                    "doc_id": "doc-1",
                    "doc_name": "tables/model_release_records.jsonl",
                    "page_label": None,
                    "block_type": "row",
                    "payload": {"field_summary": "模型名称=KM_textdetection_HoriTextDetection_Plate_cuda_nart_cuda11.0-trt7.1-int8-T4_b8_3.1.0.model"},
                }
            ]
        )
        service = AceService(store=store, v2_client=fake, auto_import_seed=False)
        with patch("ace_rag.ace.service.answer_with_llm", new=AsyncMock(side_effect=AssertionError("LLM should not run"))):
            response = await service.query(QueryRequest(query="通用车牌文字识别在 T4 上现在用哪个模型？"))
        self.assertIn("KM_textdetection_HoriTextDetection_Plate", response.answer)
        self.assertIn("[证据1]", response.answer)
        self.assertEqual(response.llm_config["mode"], "online_feedback_override")
        self.assertEqual(response.playbook.items[0].section, "online_feedback")

    async def test_query_uses_v2_query_answer_without_feedback(self):
        store = self.make_store()
        fake = FakeV2Client(query_answer="gbrain 完整答案 [证据1]")
        service = AceService(store=store, v2_client=fake, auto_import_seed=False)
        with patch("ace_rag.ace.service.answer_with_llm", new=AsyncMock(side_effect=AssertionError("LLM should not run"))):
            response = await service.query(QueryRequest(query="我想做横幅/标语检测，T4 上现在推荐用哪个模型？"))
        self.assertEqual(response.answer, "gbrain 完整答案 [证据1]")
        self.assertEqual(response.llm_config["mode"], "gbrain_v2_query_answer")
        self.assertIsNotNone(fake.last_query_payload)

    async def test_feedback_auto_organizes_playbook_when_delta_reaches_threshold(self):
        store = self.make_store()
        fake = FakeV2Client()
        service = AceService(store=store, v2_client=fake, auto_import_seed=False)
        run_id = store.insert_run(
            query="车牌文字识别 T4 模型应该查哪个字段？",
            request={},
            v2_request={"sources": ["table"]},
            v2_response={},
            playbook_item_ids=[],
            answer="旧答案",
            timings={},
        )
        settings = self.make_settings(
            ENABLE_PLAYBOOK_AUTO_ORGANIZE=True,
            PLAYBOOK_ORGANIZE_DELTA_THRESHOLD=1,
            PLAYBOOK_ORGANIZE_MAX_ITEMS=20,
        )
        organized_item = PlaybookItem(
            item_id="pb-auto-org-field-binding",
            section="field_binding",
            content="遇到车牌文字识别模型查询时，要保持检测/识别字段绑定，优先核对 table 行里的模型名称。",
            tags=["plate", "field-binding"],
            source_hints=["table"],
            query_intents=["field_lookup"],
            expansion_terms=["车牌文字识别", "模型名称"],
            confidence=0.86,
            provenance={"source": "playbook_auto_organizer", "merged_from": []},
        )

        with patch("ace_rag.ace.service.get_settings", return_value=settings), patch(
            "ace_rag.ace.service.organize_playbook_with_llm",
            new=AsyncMock(return_value=([organized_item], [], {"mode": "llm", "accepted_item_count": 1})),
        ) as organizer:
            response = await service.add_feedback(
                FeedbackRequest(
                    run_id=run_id,
                    feedback_type="correction",
                    corrected_answer="应该返回 table 中绑定的车牌文字识别模型名称。",
                    expected_evidence_ids=["table::模型汇总源数据-0135"],
                    comment="不要把检测答成识别",
                )
            )

        self.assertTrue(response.feedback_id.startswith("fb-"))
        organizer.assert_awaited_once()
        items = {item.item_id: item for item in store.list_items(include_inactive=True)}
        self.assertIn("pb-auto-org-field-binding", items)
        online_items = [item for item in items.values() if item.section == "online_feedback"]
        self.assertEqual(len(online_items), 1)
        self.assertEqual(online_items[0].status, "active")
        self.assertEqual(store.get_auto_organize_baseline_count(), store.get_item_count())
        with store.connect() as conn:
            row = conn.execute(
                "SELECT operation_type, status, payload_json FROM playbook_operations "
                "WHERE operation_type = 'AUTO_ORGANIZE_PLAYBOOK'"
            ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "applied")
        self.assertIn("pb-auto-org-field-binding", row["payload_json"])

    async def test_feedback_auto_organize_waits_until_threshold(self):
        store = self.make_store()
        service = AceService(store=store, v2_client=FakeV2Client(), auto_import_seed=False)
        run_id = store.insert_run(
            query="安全绳检测在 T4 上有部署吗？",
            request={},
            v2_request={"sources": ["adela"]},
            v2_response={},
            playbook_item_ids=[],
            answer="旧答案",
            timings={},
        )
        settings = self.make_settings(
            ENABLE_PLAYBOOK_AUTO_ORGANIZE=True,
            PLAYBOOK_ORGANIZE_DELTA_THRESHOLD=3,
            PLAYBOOK_ORGANIZE_MAX_ITEMS=20,
        )

        with patch("ace_rag.ace.service.get_settings", return_value=settings), patch(
            "ace_rag.ace.service.organize_playbook_with_llm",
            new=AsyncMock(side_effect=AssertionError("organizer should not run below threshold")),
        ):
            await service.add_feedback(
                FeedbackRequest(
                    run_id=run_id,
                    feedback_type="correction",
                    corrected_answer="应该查询 adela 部署记录。",
                    expected_evidence_ids=["adela::deploy-1"],
                )
            )

        with store.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS count FROM playbook_operations WHERE operation_type = 'AUTO_ORGANIZE_PLAYBOOK'"
            ).fetchone()
        self.assertEqual(int(row["count"]), 0)


if __name__ == "__main__":
    unittest.main()
