from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from capa.capabilities import (
    CAPABILITY_SPECS,
    build_capability_inventory,
    validate_capability_inventory,
)
from capa.session_audit import audit_llm_debug, audit_sessions
from capa.tools import registry
from capa.tools.contracts import ToolCall, ToolExecutionContext
from capa.tools.executor import ToolExecutor


ROOT = Path(__file__).resolve().parents[1]


class DemoCapabilityInventoryTests(unittest.TestCase):
    def test_inventory_covers_every_declared_tool_and_asset(self) -> None:
        self.assertTrue(set(registry.get_declared_tool_names()).issubset(CAPABILITY_SPECS))
        self.assertEqual(validate_capability_inventory(ROOT), [])
        inventory = build_capability_inventory(ROOT)
        self.assertEqual(len(inventory), len(registry.get_declared_tool_names()))
        self.assertNotIn("adela_cli_eval", registry.get_declared_tool_names())
        self.assertNotIn("adela_cli_eval", {item["tool_name"] for item in inventory})
        answerer = next(item for item in inventory if item["tool_name"] == "answerer")
        self.assertEqual(answerer["runtime_owner"], "orchestrator")

    def test_historical_actions_normalize_to_current_tools(self) -> None:
        expected = {
            "transfer_advisory": "migration_advisor",
            "adela_cli_benchmark": "adela_cli_eval",
            "evidence_synthesis_answer": "answerer",
            "target-detection-evaluation": "pipeline_eval",
            "qwen-vlm-open-set-delection": "qwen_detection",
            "rexomni-open-set-detection": "rexomni_detection",
        }
        for old_action, current_action in expected.items():
            self.assertEqual(registry.normalize_tool_action(old_action), current_action)
            if current_action == "adela_cli_eval":
                self.assertFalse(registry.is_valid_tool_action(old_action))
            else:
                self.assertTrue(registry.is_valid_tool_action(old_action))


class ToolExecutorParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.events: list[dict] = []
        self.calls: list[str] = []

        def callback(name: str):
            def run(*args, **kwargs):
                self.calls.append(name)
                return {"success": True, "summary": name}

            return run

        self.executor = ToolExecutor(
            emit=self.events.append,
            failure_observation=lambda action, message: {
                "action": action,
                "success": False,
                "summary": message,
            },
            run_rag_streaming=callback("rag"),
            run_flux_only_streaming=callback("flux"),
            run_detection_only_streaming=callback("qwen_detection"),
            run_rex_detection_only_streaming=callback("rexomni_detection"),
            run_pipeline_streaming=callback("pipeline"),
            run_migration_advisor_streaming=callback("migration_advisor"),
            run_adela_cli_streaming=callback("adela_cli"),
            resolve_adela_model_reference=lambda **kwargs: {
                "status": "resolved",
                "rawmodel_id": int(kwargs.get("rawmodel_id") or 17),
                "matched_name": "fixture-model",
            },
        )
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.image = root / "image.jpg"
        self.image.write_bytes(b"fixture")
        self.ctx = ToolExecutionContext(
            text="fixture request",
            image_path=str(self.image),
            image_paths=[str(self.image)],
            api_key="fixture-key",
            api_base="http://fixture.invalid/v1",
            run_dir=root,
            run_stamp="fixture-run",
            session_id="fixture-session",
            session={"session_id": "fixture-session"},
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _execute(self, action: str, action_input: dict) -> None:
        result = self.executor.execute(
            tool_call=ToolCall(action=action, action_input=action_input),
            ctx=self.ctx,
        )
        self.assertTrue(result.ok, result.error_message)
        self.assertEqual(result.action, registry.normalize_tool_action(action))

    def test_every_executor_owned_capability_dispatches(self) -> None:
        cases = [
            ("rag_answer", {"query": "fixture"}, "rag"),
            (
                "flux-image-generation",
                {"task_text": "fixture", "source_image_required": False, "num_images": 1},
                "flux",
            ),
            ("qwen_detection", {"label": "fixture"}, "qwen_detection"),
            ("rexomni_detection", {"label": "fixture"}, "rexomni_detection"),
            ("pipeline_eval", {"task_text": "fixture"}, "pipeline"),
            (
                "migration_advisor",
                {"user_query": "fixture", "use_image": False},
                "migration_advisor",
            ),
        ]
        for action, action_input, expected_callback in cases:
            with self.subTest(action=action):
                before = len(self.calls)
                self._execute(action, action_input)
                self.assertEqual(self.calls[before:], [expected_callback])

    @patch("capa.tools.executor.agent.rewrite_query_with_fallback")
    def test_requestion_dispatches_without_network(
        self, rewrite_query_with_fallback
    ) -> None:
        rewrite_query_with_fallback.return_value = "rewritten fixture"
        self._execute(
            "re_question",
            {"query": "fixture", "rewrite_reason": "rag_miss", "retrieval_round": 2},
        )
        rewrite_query_with_fallback.assert_called_once()

    def test_observed_active_legacy_tool_names_still_dispatch(self) -> None:
        self._execute(
            "transfer_advisory",
            {"user_query": "fixture", "use_image": False},
        )
        self.assertEqual(self.calls[-1:], ["migration_advisor"])

    def test_adela_is_disabled_before_executor_dispatch(self) -> None:
        result = self.executor.execute(
            tool_call=ToolCall(
                action="adela_cli_eval",
                action_input={
                    "rawmodel_id": 17,
                    "platform": "fixture-platform",
                    "eval_type": 1,
                },
            ),
            ctx=self.ctx,
        )
        self.assertFalse(result.ok)
        self.assertNotIn("adela_cli", self.calls)
        self.assertIn("已禁用", result.error_message)

    def test_image_preconditions_fail_before_external_calls(self) -> None:
        no_image_ctx = ToolExecutionContext(
            text="fixture request",
            image_path="",
            image_paths=[],
            api_key="fixture-key",
            api_base="http://fixture.invalid/v1",
            run_dir=self.ctx.run_dir,
            run_stamp="fixture-no-image",
            session_id="fixture-session",
            session={"session_id": "fixture-session"},
        )
        cases = [
            ("flux-image-generation", {"task_text": "fixture", "source_image_required": True}),
            ("qwen_detection", {"label": "person"}),
            ("rexomni_detection", {"label": "person"}),
            ("pipeline_eval", {"task_text": "fixture"}),
            ("migration_advisor", {"user_query": "fixture", "use_image": True}),
        ]
        for action, action_input in cases:
            with self.subTest(action=action):
                before = list(self.calls)
                result = self.executor.execute(
                    tool_call=ToolCall(action=action, action_input=action_input),
                    ctx=no_image_ctx,
                )
                self.assertFalse(result.ok)
                self.assertEqual(self.calls, before)
                self.assertTrue(result.error_message)

    def test_detection_label_preconditions_fail_before_external_calls(self) -> None:
        for action in ("qwen_detection", "rexomni_detection"):
            with self.subTest(action=action):
                before = list(self.calls)
                result = self.executor.execute(
                    tool_call=ToolCall(action=action, action_input={"label": ""}),
                    ctx=self.ctx,
                )
                self.assertFalse(result.ok)
                self.assertEqual(self.calls, before)
                self.assertIn("label", result.error_message)


class SessionAuditTests(unittest.TestCase):
    def test_audit_normalizes_legacy_schema_without_emitting_user_text(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "20260101"
            root.mkdir()
            session = {
                "pending_clarification": None,
                "threads": {
                    "thread_fixture": {
                        "last_image_path": "/private/image.jpg",
                        "query_trajectories": {
                            "query_fixture": {
                                "query": "private user text",
                                "steps": [
                                    {"action": "transfer_advisory"},
                                    {"action": "final_answer"},
                                ],
                            }
                        },
                        "raw_ledger": [
                            {
                                "event_type": "PLAN_DECISION",
                                "payload": {"action": "transfer_advisory"},
                            },
                            {
                                "event_type": "OBSERVATION",
                                "payload": {
                                    "_action": "transfer_advisory",
                                    "success": True,
                                    "answer": "private answer",
                                },
                            },
                        ],
                    }
                },
            }
            (root / "session.json").write_text(
                json.dumps(session, ensure_ascii=False), encoding="utf-8"
            )
            result = audit_sessions(Path(tempdir))
        serialized = json.dumps(result, ensure_ascii=False)
        self.assertNotIn("private user text", serialized)
        self.assertNotIn("private answer", serialized)
        self.assertNotIn("/private/image.jpg", serialized)
        self.assertEqual(result["plan_actions"]["migration_advisor"], 1)
        self.assertEqual(result["observation_actions"]["migration_advisor"], 1)
        self.assertEqual(result["queries"], 1)
        self.assertEqual(result["trajectory_steps"], 2)

    def test_llm_debug_audit_does_not_emit_prompt_or_response(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "20260101"
            root.mkdir()
            record = {
                "session_id": "private-session",
                "stage": "planner_response",
                "payload": {
                    "messages": [{"content": "private prompt"}],
                    "raw_response": json.dumps(
                        {
                            "decision_type": "tool",
                            "action": "transfer_advisory",
                            "thought": "private reasoning",
                        }
                    ),
                },
            }
            (root / "record.json").write_text(
                json.dumps(record, ensure_ascii=False), encoding="utf-8"
            )
            result = audit_llm_debug(Path(tempdir))
        serialized = json.dumps(result, ensure_ascii=False)
        self.assertNotIn("private-session", serialized)
        self.assertNotIn("private prompt", serialized)
        self.assertNotIn("private reasoning", serialized)
        self.assertEqual(result["actions"]["migration_advisor"], 1)

    def test_llm_debug_audit_separates_synthetic_smokes(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "20260101"
            root.mkdir()
            for name, session_id in (
                ("smoke.json", "capability_http_smoke_1"),
                ("eval.json", "planner_grpo_fixture"),
                ("runtime.json", "browser_fixture"),
            ):
                (root / name).write_text(
                    json.dumps(
                        {"session_id": session_id, "stage": "planner_request"}
                    ),
                    encoding="utf-8",
                )
            result = audit_llm_debug(Path(tempdir))
        self.assertEqual(
            result["cohorts"],
            {"grpo_eval": 1, "runtime": 1, "synthetic_smoke": 1},
        )


if __name__ == "__main__":
    unittest.main()
