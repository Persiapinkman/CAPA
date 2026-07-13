from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from capa.experiments.registry import (
    RegistryError,
    append_entry,
    legacy_entry_to_v2,
    load_registry,
    validate_entry,
)


def valid_entry(run_id: str = "run-1") -> dict:
    return {
        "schema_version": "2.0",
        "run_id": run_id,
        "study_id": "study-1",
        "date": "2026-07-12",
        "kind": "eval_generation_repeated",
        "status": "completed",
        "purpose": "Test registry behavior.",
        "hypothesis": "The registry rejects duplicate run IDs.",
        "parent_run_id": None,
        "provenance": {
            "git_commit": "abc123",
            "command": "test",
            "seed": 42,
            "environment": {},
        },
        "data": {"dataset_id": "data-1", "split": "dev", "files": {}},
        "method": {},
        "metrics": {},
        "artifacts": {},
        "decision": {"outcome": "test", "rationale": "unit test"},
    }


class RegistryTests(unittest.TestCase):
    def test_valid_entry_passes_strict_validation(self) -> None:
        self.assertEqual(validate_entry(valid_entry(), strict=True), [])

    def test_append_rejects_duplicate_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.jsonl"
            append_entry(path, valid_entry())
            with self.assertRaises(RegistryError):
                append_entry(path, valid_entry())
            self.assertEqual(len(load_registry(path)), 1)

    def test_legacy_conversion_is_explicit_and_valid(self) -> None:
        converted = legacy_entry_to_v2(
            {
                "date": "2026-07-10",
                "run_id": "legacy-1",
                "kind": "eval_generation",
                "status": "completed",
                "notes": "legacy",
            }
        )
        self.assertEqual(validate_entry(converted, strict=True), [])
        self.assertEqual(converted["provenance"]["git_commit"], "unknown_not_recorded")


if __name__ == "__main__":
    unittest.main()
