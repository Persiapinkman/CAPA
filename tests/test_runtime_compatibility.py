from __future__ import annotations

import unittest

from capa import agent, memory
from capa.tools.registry import get_declared_tool_names


class RuntimeCompatibilityTests(unittest.TestCase):
    def test_runtime_exports_expected_contract(self) -> None:
        self.assertEqual(memory.CONTEXT_SCHEMA_VERSION, "planner-context-v2")
        self.assertIn("migration_advisor", get_declared_tool_names())
        self.assertTrue(callable(agent.normalize_agent_action))

    def test_api_key_configuration_remains_present(self) -> None:
        self.assertTrue(agent.DEMO_ROUTE_API_KEY)

    def test_rag_hit_threshold_matches_gbrain_answerability_contract(self) -> None:
        self.assertFalse(
            agent.is_rag_miss({"knowledge_base_fully_answered": 0.92})
        )
        self.assertFalse(
            agent.is_rag_miss({"knowledge_base_fully_answered": 0.85})
        )
        self.assertTrue(
            agent.is_rag_miss({"knowledge_base_fully_answered": 0.84})
        )
        self.assertTrue(agent.is_rag_miss({"knowledge_base_fully_answered": 0.0}))

    def test_identical_query_rewrite_is_detected(self) -> None:
        self.assertTrue(agent.queries_equivalent("  Model   OID ", "model oid"))
        self.assertFalse(agent.queries_equivalent("model oid", "model platform"))


if __name__ == "__main__":
    unittest.main()
