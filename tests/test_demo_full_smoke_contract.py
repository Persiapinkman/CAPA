from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "pipelines/demo/run_full_demo_smoke.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("demo_full_smoke_contract", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DemoFullSmokeContractTests(unittest.TestCase):
    def test_generation_quality_reads_nested_event_payload(self) -> None:
        module = _load_module()
        summary = module._generation_quality_summary(
            [
                {
                    "type": "generation_quality",
                    "data": {
                        "passed": True,
                        "content_compliance_checked": False,
                        "warnings": [],
                    },
                }
            ]
        )

        self.assertIs(summary["gate"], True)
        self.assertIs(summary["content_compliance_checked"], False)
        self.assertEqual(summary["warnings"], [])


if __name__ == "__main__":
    unittest.main()
