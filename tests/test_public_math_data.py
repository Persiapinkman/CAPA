from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from training.public_sft_grpo_v1.scripts.public_math_contract import extract_gsm8k_gold


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "training/public_sft_grpo_v1/data/gsm8k_sft32_v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class PublicMathDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
        cls.rows = {
            "train": load_jsonl(DATA_DIR / "train.jsonl"),
            "development": load_jsonl(DATA_DIR / "development.jsonl"),
            "sealed_test": load_jsonl(DATA_DIR / "sealed_test.jsonl"),
        }

    def test_split_sizes_and_sources(self):
        self.assertEqual(len(self.rows["train"]), 32)
        self.assertEqual(len(self.rows["development"]), 32)
        self.assertEqual(len(self.rows["sealed_test"]), 128)
        self.assertTrue(all(row["source_split"] == "train" for row in self.rows["train"]))
        self.assertTrue(all(row["source_split"] == "train" for row in self.rows["development"]))
        self.assertTrue(all(row["source_split"] == "test" for row in self.rows["sealed_test"]))

    def test_derived_splits_are_disjoint(self):
        id_sets = {name: {row["sample_id"] for row in rows} for name, rows in self.rows.items()}
        question_sets = {
            name: {row["question_sha256"] for row in rows} for name, rows in self.rows.items()
        }
        for left, right in (("train", "development"), ("train", "sealed_test"), ("development", "sealed_test")):
            self.assertFalse(id_sets[left] & id_sets[right])
            self.assertFalse(question_sets[left] & question_sets[right])

    def test_conversation_and_reward_contract(self):
        for rows in self.rows.values():
            for row in rows:
                self.assertEqual([message["role"] for message in row["messages"]], ["system", "user", "assistant"])
                self.assertEqual(
                    extract_gsm8k_gold(row["messages"][-1]["content"]),
                    row["ground_truth"],
                )
                self.assertEqual(row["chat_template_kwargs"], {"enable_thinking": False})
                self.assertEqual(row["reward_metadata"]["accuracy_weight"], 0.95)
                self.assertEqual(row["reward_metadata"]["format_weight"], 0.05)

    def test_manifest_hashes(self):
        for split, metadata in self.manifest["files"].items():
            path = Path(metadata["path"])
            self.assertTrue(path.exists())
            self.assertEqual(sha256_file(path), metadata["sha256"], split)
        self.assertEqual(self.manifest["isolation"]["normalized_upstream_train_test_overlap"], 0)
        self.assertFalse(self.manifest["isolation"]["sealed_test_allowed_for_checkpoint_selection"])

    def test_label_audit_is_complete(self):
        for split in self.rows:
            stats = self.manifest["splits"][split]
            self.assertEqual(stats["assistant_mask_nonempty_rate"], 1.0)
            self.assertEqual(stats["eos_supervised_rate"], 1.0)
            self.assertLessEqual(stats["total_tokens"]["max"], self.manifest["max_length"])
            self.assertGreater(stats["prompt_tokens"]["min"], 2)


if __name__ == "__main__":
    unittest.main()
