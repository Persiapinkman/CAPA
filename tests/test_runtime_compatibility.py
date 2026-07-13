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


if __name__ == "__main__":
    unittest.main()
