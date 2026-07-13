from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "training/planner_grpo_seed_v1/cases"
STEP_DIR = ROOT / "training/planner_grpo_seed_v1/sft_data_stateful_retrieval_v2_chatml"
MANIFEST = ROOT / "data/datasets/planner_stateful_retrieval_v2/manifest.json"
FOCUSED_DIR = ROOT / "training/planner_grpo_seed_v1/sft_data_first_miss_transition_v1_chatml"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StatefulRetrievalV2DatasetTests(unittest.TestCase):
    def test_expansion_sizes_and_five_step_primary_scenario(self) -> None:
        expected = {
            "train": (504, 1080, 72),
            "dev": (112, 240, 16),
            "test": (224, 480, 32),
        }
        for split, (case_count, step_count, entity_count) in expected.items():
            cases = load_jsonl(CASE_DIR / f"planner_stateful_retrieval_v2_{split}_cases.jsonl")
            steps = load_jsonl(STEP_DIR / f"{split}.jsonl")
            self.assertEqual(len(cases), case_count)
            self.assertEqual(len(steps), step_count)
            self.assertEqual(len({row["entity_id"] for row in cases}), entity_count)
            primary = [row for row in cases if row["category"] == "rag_double_miss_recovery"]
            self.assertTrue(primary)
            self.assertTrue(all(len(row["expected_decisions"]) == 5 for row in primary))
            self.assertTrue(
                all(
                    [step.get("action") for step in row["expected_decisions"]]
                    == ["rag_answer", "re_question", "rag_answer", "re_question", "rag_answer"]
                    for row in primary
                )
            )

    def test_splits_are_isolated_and_steps_are_unique(self) -> None:
        cases = {
            split: load_jsonl(CASE_DIR / f"planner_stateful_retrieval_v2_{split}_cases.jsonl")
            for split in ("train", "dev", "test")
        }
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
            for key in ("case_id", "entity_id", "template_id", "user_query"):
                self.assertFalse(
                    {row[key] for row in cases[left]} & {row[key] for row in cases[right]},
                    f"{left}/{right} overlap for {key}",
                )
        for split in ("train", "dev", "test"):
            steps = load_jsonl(STEP_DIR / f"{split}.jsonl")
            pairs = {(row["prompt"], row["completion"]) for row in steps}
            self.assertEqual(len(pairs), len(steps))

    def test_action_reward_and_support_subset_are_locked(self) -> None:
        for split in ("train", "dev", "test"):
            for row in load_jsonl(STEP_DIR / f"{split}.jsonl"):
                reward = json.loads(row["reward_spec"])
                self.assertEqual(reward["action_match"], 0.75)
                self.assertEqual(reward["wrong_action_cap"], 0.20)
        support = load_jsonl(STEP_DIR / "support_audit.jsonl")
        self.assertEqual(len(support), 40)
        self.assertEqual({row["category"] for row in support}, {"rag_double_miss_recovery"})
        self.assertEqual({row["step_index"] for row in support}, {1, 2, 3, 4, 5})
        self.assertEqual(len({row["entity_id"] for row in support}), 8)

    def test_registered_hashes_and_frozen_test_status(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for name, relative_path in manifest["files"].items():
            self.assertEqual(sha256(ROOT / relative_path), manifest["sha256"][name])
        self.assertEqual(
            manifest["integrity"]["status"],
            "test_opened_once_after_development_replication_passed_frozen",
        )


class FirstMissTransitionDatasetTests(unittest.TestCase):
    def test_weighted_view_keeps_every_source_and_locks_weights(self) -> None:
        source = load_jsonl(STEP_DIR / "train.jsonl")
        focused = load_jsonl(FOCUSED_DIR / "train.jsonl")
        self.assertEqual(len(source), 1080)
        self.assertEqual(len(focused), 1584)
        source_keys = {(row["case_id"], row["step_index"]) for row in source}
        focused_keys = {(row["source_case_id"], row["step_index"]) for row in focused}
        self.assertEqual(source_keys, focused_keys)
        counts: dict[tuple[str, int], int] = {}
        for row in focused:
            key = (row["category"], row["step_index"])
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(counts[("rag_double_miss_recovery", 1)], 144)
        self.assertEqual(counts[("rag_double_miss_recovery", 2)], 288)
        self.assertEqual(counts[("rag_single_miss_recovery", 2)], 288)
        self.assertEqual(counts[("direct_rag_guardrail", 1)], 72)

    def test_focused_metadata_hash_is_current(self) -> None:
        metadata = json.loads((FOCUSED_DIR / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["rows"], 1584)
        self.assertEqual(metadata["unique_source_rows"], 1080)
        self.assertEqual(sha256(FOCUSED_DIR / "train.jsonl"), metadata["output_sha256"])


if __name__ == "__main__":
    unittest.main()
