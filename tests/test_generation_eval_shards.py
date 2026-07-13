from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from pipelines.eval.combine_generation_eval_shards import (
    combine_predictions,
    validate_and_sort_shards,
)


def record(run_id: str, offset: int, rows: int) -> dict:
    return {
        "run_id": run_id,
        "study_id": "study",
        "provenance": {"seed": 42},
        "data": {
            "dataset_id": "dataset",
            "split": "dev",
            "files": {"eval": "/tmp/eval.jsonl"},
            "sha256": {"eval": "hash"},
            "rows": rows,
            "offset": offset,
            "total_source_rows": 4,
        },
        "method": {
            "model": "/tmp/model",
            "adapter_path": "/tmp/adapter",
            "prompt_format": "qwen_chatml",
            "generation": {"repeats": 1, "do_sample": False},
        },
        "artifacts": {"predictions": []},
    }


class GenerationEvalShardTests(unittest.TestCase):
    def test_validation_sorts_and_requires_full_contiguous_coverage(self) -> None:
        second = record("shard1", 2, 2)
        first = record("shard0", 0, 2)
        ordered = validate_and_sort_shards([second, first])
        self.assertEqual([row["run_id"] for row in ordered], ["shard0", "shard1"])

        gap = copy.deepcopy(second)
        gap["data"]["offset"] = 3
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            validate_and_sort_shards([first, gap])

    def test_validation_rejects_metadata_mismatch(self) -> None:
        first = record("shard0", 0, 2)
        second = record("shard1", 2, 2)
        second["method"]["adapter_path"] = "/tmp/other"
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            validate_and_sort_shards([first, second])

    def test_combine_predictions_rejects_overlap(self) -> None:
        row = {"case_id": "case", "step_index": 1, "repeat": 1}
        with tempfile.TemporaryDirectory() as temp_dir:
            prediction_path = Path(temp_dir) / "predictions.jsonl"
            prediction_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            first = record("shard0", 0, 2)
            second = record("shard1", 2, 2)
            first["artifacts"]["predictions"] = [str(prediction_path)]
            second["artifacts"]["predictions"] = [str(prediction_path)]
            with self.assertRaisesRegex(ValueError, "overlapping prediction"):
                combine_predictions([first, second], 0)


if __name__ == "__main__":
    unittest.main()
