from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

try:
    from training.public_sft_grpo_v1.scripts.combine_math_eval_shards import main as combine_main

    MATH_VERIFY_AVAILABLE = True
except ModuleNotFoundError:
    MATH_VERIFY_AVAILABLE = False


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(MATH_VERIFY_AVAILABLE, "requires the pinned Math-Verify environment")
class MathEvalContractTests(unittest.TestCase):
    def test_two_shards_combine_without_duplicate_or_metric_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = {
                "schema_version": "1.0",
                "status": "completed",
                "runtime_seconds": 1.0,
                "model_name_or_path": "/model",
                "adapter_path": "",
                "data_dir": "/data",
                "dataset_manifest_sha256": "manifest",
                "generation": {
                    "do_sample": False,
                    "max_new_tokens": 8,
                    "batch_size": 1,
                    "seed": 42,
                    "train_limit": 2,
                    "development_limit": 0,
                },
                "nonfinite_metric_count": 0,
            }
            for shard_index in range(2):
                shard = root / f"shard{shard_index}"
                shard.mkdir()
                sample = {
                    "sample_id": f"sample-{shard_index}",
                    "split": "train",
                    "source_index": shard_index,
                    "level": f"Level {shard_index + 1}",
                    "type": "Algebra",
                    "gold_boxed": r"\boxed{1}",
                    "completion": r"\boxed{1}" if shard_index == 0 else r"\boxed{2}",
                    "completion_token_count": 4,
                    "eos_found": True,
                    "clipped": False,
                    "symbolic_accuracy": float(shard_index == 0),
                    "strict_format": 1.0,
                    "strict_exact": float(shard_index == 0),
                    "parse_success": 1.0,
                    "box_count": 1,
                    "parsed_prediction": "[1]" if shard_index == 0 else "[2]",
                }
                sample_path = shard / "samples.jsonl"
                sample_path.write_text(json.dumps(sample, sort_keys=True) + "\n")
                result = {
                    **shared,
                    "sharding": {
                        "num_shards": 2,
                        "shard_index": shard_index,
                        "global_selected_rows": 2,
                        "shard_rows": 1,
                        "global_selected_ids_sha256": "population",
                        "shard_ids_sha256": f"shard-{shard_index}",
                    },
                    "samples_sha256": sha256_file(sample_path),
                }
                (shard / "result.json").write_text(json.dumps(result))
            output = root / "combined"
            original_argv = sys.argv
            try:
                sys.argv = [
                    "combine_math_eval_shards.py",
                    "--input-dir",
                    str(root / "shard0"),
                    "--input-dir",
                    str(root / "shard1"),
                    "--output-dir",
                    str(output),
                ]
                with redirect_stdout(io.StringIO()):
                    combine_main()
            finally:
                sys.argv = original_argv
            combined = json.loads((output / "result.json").read_text())
            self.assertEqual(combined["sharding"]["combined_rows"], 2)
            self.assertEqual(combined["metrics"]["train"]["overall"]["symbolic_accuracy"], 0.5)
            self.assertEqual(combined["metrics"]["train"]["overall"]["strict_format_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
