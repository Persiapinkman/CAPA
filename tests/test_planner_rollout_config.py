from __future__ import annotations

import argparse
import unittest

from training.planner_grpo_seed_v1.scripts import run_planner_grpo_rollout as rollout


class PlannerRolloutConfigTests(unittest.TestCase):
    def test_cli_overrides_patch_the_runtime_module_used_by_functions(self) -> None:
        original = (
            rollout.agent.DEMO_ROUTE_API_BASE,
            rollout.agent.DEMO_ROUTE_API_KEY,
            rollout.agent.DEMO_ROUTE_MODEL,
        )
        try:
            rollout.configure_generation(
                argparse.Namespace(
                    api_base="http://127.0.0.1:9999/v1",
                    api_key="test-key",
                    model="test-model",
                    temperature=0.0,
                    top_p=1.0,
                    seed=42,
                    do_sample="false",
                    openai_timeout_seconds=30,
                )
            )
            self.assertEqual(rollout.agent.DEMO_ROUTE_API_BASE, "http://127.0.0.1:9999/v1")
            self.assertEqual(rollout.agent.DEMO_ROUTE_API_KEY, "test-key")
            self.assertEqual(rollout.agent.DEMO_ROUTE_MODEL, "test-model")
            self.assertIs(
                rollout.agent.choose_agent_step_llm.__globals__,
                rollout.agent.__dict__,
            )
        finally:
            (
                rollout.agent.DEMO_ROUTE_API_BASE,
                rollout.agent.DEMO_ROUTE_API_KEY,
                rollout.agent.DEMO_ROUTE_MODEL,
            ) = original


if __name__ == "__main__":
    unittest.main()
