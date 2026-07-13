from __future__ import annotations

import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "training/planner_grpo_seed_v1/sft_data_coref_contrast_v1_chatml"
RAG_MISS_DATA = ROOT / "training/planner_grpo_seed_v1/sft_data_rag_miss_state_machine_v1_chatml"
ACTION_REWARD_DATA = ROOT / "training/planner_grpo_seed_v1/sft_data_rag_miss_action_reward_v1_chatml"


class CorefContrastDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = [
            json.loads(line)
            for line in (DATA / "train.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        cls.metadata = json.loads((DATA / "metadata.json").read_text(encoding="utf-8"))

    def test_weighted_distribution_is_locked(self) -> None:
        counts = Counter((row["category"], int(row["step_index"])) for row in self.rows)
        self.assertEqual(len(self.rows), 192)
        self.assertEqual(counts[("coref_rewrite_then_rag", 1)], 96)
        for category in (
            "rag_miss_rewrite_then_rag",
            "direct_rag_guardrail",
            "memory_hit_end",
            "general_answer_guardrail",
        ):
            self.assertEqual(counts[(category, 1)], 24)

    def test_replicas_are_traceable_and_unique(self) -> None:
        self.assertEqual(len({row["training_row_id"] for row in self.rows}), len(self.rows))
        self.assertEqual(len({row["source_case_id"] for row in self.rows}), 120)
        self.assertTrue(all(row["source_case_id"].startswith("SRV1-TRAIN-") for row in self.rows))

    def test_generated_hash_matches_metadata(self) -> None:
        payload = (DATA / "train.jsonl").read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), self.metadata["output_sha256"])


class RagMissStateMachineDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = [
            json.loads(line)
            for line in (RAG_MISS_DATA / "train.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        cls.metadata = json.loads(
            (RAG_MISS_DATA / "metadata.json").read_text(encoding="utf-8")
        )

    def test_weighted_state_machine_distribution_is_locked(self) -> None:
        counts = Counter((row["category"], int(row["step_index"])) for row in self.rows)
        self.assertEqual(len(self.rows), 240)
        for step in (1, 2, 3):
            self.assertEqual(counts[("rag_miss_rewrite_then_rag", step)], 48)
        for category in (
            "coref_rewrite_then_rag",
            "direct_rag_guardrail",
            "memory_hit_end",
            "general_answer_guardrail",
        ):
            self.assertEqual(counts[(category, 1)], 24)

    def test_generated_hash_matches_metadata(self) -> None:
        payload = (RAG_MISS_DATA / "train.jsonl").read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), self.metadata["output_sha256"])


class RagMissActionRewardDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = [
            json.loads(line)
            for line in (ACTION_REWARD_DATA / "train.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        cls.metadata = json.loads(
            (ACTION_REWARD_DATA / "metadata.json").read_text(encoding="utf-8")
        )

    def test_action_reward_profile_is_locked(self) -> None:
        self.assertEqual(len(self.rows), 240)
        specs = {row["reward_spec"] for row in self.rows}
        self.assertEqual(len(specs), 1)
        spec = json.loads(specs.pop())
        self.assertEqual(spec["action_match"], 0.75)
        self.assertEqual(spec["wrong_action_cap"], 0.20)
        self.assertTrue(all(row["reward_profile"] == "action_dominant_v1" for row in self.rows))

    def test_generated_hash_matches_metadata(self) -> None:
        payload = (ACTION_REWARD_DATA / "train.jsonl").read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), self.metadata["output_sha256"])


if __name__ == "__main__":
    unittest.main()
