from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "training/public_sft_grpo_v1/data/math_sft1024_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(split: str) -> list[dict]:
    return [json.loads(line) for line in (DATA_DIR / f"{split}.jsonl").read_text().splitlines()]


class MathSFTDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((DATA_DIR / "manifest.json").read_text())
        cls.rows = {split: load_rows(split) for split in ("train", "development", "sealed_test")}

    def test_frozen_sizes_hashes_and_token_limit(self):
        expected = {"train": 1024, "development": 256, "sealed_test": 512}
        self.assertEqual({split: len(rows) for split, rows in self.rows.items()}, expected)
        for split, count in expected.items():
            path = DATA_DIR / f"{split}.jsonl"
            self.assertEqual(sha256_file(path), self.manifest["files"][split]["sha256"])
            self.assertEqual(self.manifest["splits"][split]["rows"], count)
            self.assertLessEqual(self.manifest["splits"][split]["total_tokens"]["max"], 2048)

    def test_derived_splits_have_no_question_or_sample_overlap(self):
        sample_ids = [row["sample_id"] for rows in self.rows.values() for row in rows]
        questions = [row["question_sha256"] for rows in self.rows.values() for row in rows]
        self.assertEqual(len(sample_ids), len(set(sample_ids)))
        self.assertEqual(len(questions), len(set(questions)))
        self.assertEqual(self.manifest["isolation"]["derived_question_overlap"], 0)
        self.assertFalse(
            self.manifest["isolation"]["sealed_test_allowed_for_checkpoint_selection"]
        )

    def test_source_split_and_all_strata_contracts(self):
        self.assertTrue(all(row["source_split"] == "train" for row in self.rows["train"]))
        self.assertTrue(
            all(row["source_split"] == "train" for row in self.rows["development"])
        )
        self.assertTrue(
            all(row["source_split"] == "test" for row in self.rows["sealed_test"])
        )
        for split in self.rows:
            strata = {(row["level"], row["type"]) for row in self.rows[split]}
            self.assertEqual(len(strata), 35)

    def test_supervision_and_terminal_box_contract(self):
        for rows in self.rows.values():
            for row in rows:
                self.assertTrue(row["token_audit"]["within_max_length"])
                self.assertTrue(row["token_audit"]["eos_supervised"])
                self.assertEqual(row["token_audit"]["assistant_spans"], 1)
                self.assertEqual(row["gold_solution"].count("\\boxed"), 1)
                self.assertTrue(row["gold_solution"].endswith(row["gold_boxed"]))
                self.assertEqual(row["messages"][-1]["content"], row["gold_solution"])

    def test_known_upstream_contamination_was_removed(self):
        self.assertEqual(self.manifest["sources"]["normalized_upstream_train_test_overlap"], 1)
        self.assertEqual(
            self.manifest["isolation"]["known_upstream_cross_overlap_excluded_from_both_sources"],
            1,
        )
        self.assertEqual(self.manifest["filters"]["train"]["duplicate_normalized_question"], 1)
        self.assertGreater(self.manifest["filters"]["train"]["unparseable_strict_gold"], 0)
        self.assertGreater(self.manifest["filters"]["test"]["unparseable_strict_gold"], 0)


if __name__ == "__main__":
    unittest.main()
