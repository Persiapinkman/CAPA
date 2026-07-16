from __future__ import annotations

import json
import unittest

from training.planner_grpo_seed_v1.scripts.train_planner_grpo import score_step_completion


def score(
    completion: dict,
    expected: dict,
    *,
    full_actions: list[str],
    step_index: int,
    previous_action: str = "",
    reward_spec: dict | None = None,
) -> float:
    return score_step_completion(
        completion=json.dumps(completion, ensure_ascii=False),
        expected_step=json.dumps(expected, ensure_ascii=False),
        forbidden_actions="[]",
        reward_spec=json.dumps(reward_spec or {}, ensure_ascii=False),
        previous_action=previous_action,
        full_expected_actions=json.dumps(full_actions, ensure_ascii=False),
        step_index=step_index,
    )


class PlannerSequenceRewardTests(unittest.TestCase):
    def test_end_memory_hit_is_scored_as_a_first_class_decision(self) -> None:
        expected = {
            "decision_type": "end",
            "required_args": {"end_reason": "memory_hit"},
        }
        correct = score(
            {
                "decision_type": "end",
                "end_reason": "memory_hit",
                "final_answer": "",
            },
            expected,
            full_actions=["end"],
            step_index=1,
        )
        wrong_reason = score(
            {
                "decision_type": "end",
                "end_reason": "recheck_done",
                "final_answer": "",
            },
            expected,
            full_actions=["end"],
            step_index=1,
        )
        self.assertEqual(correct, 1.0)
        self.assertLess(wrong_reason, correct)

    def test_intermediate_finish_true_loses_generic_process_reward(self) -> None:
        expected = {
            "decision_type": "tool",
            "action": "rag_answer",
            "required_args": {"finish_after_tool": False},
            "arg_contains": {"query": ["安全绳", "模型版本"]},
        }
        reward_spec = {"no_premature_stop": 0.2}
        correct = score(
            {
                "decision_type": "tool",
                "action": "rag_answer",
                "action_input": {
                    "query": "安全绳 模型版本",
                    "finish_after_tool": False,
                },
            },
            expected,
            full_actions=["rag_answer", "re_question", "rag_answer"],
            step_index=1,
            reward_spec=reward_spec,
        )
        premature = score(
            {
                "decision_type": "tool",
                "action": "rag_answer",
                "action_input": {
                    "query": "安全绳 模型版本",
                    "finish_after_tool": True,
                },
            },
            expected,
            full_actions=["rag_answer", "re_question", "rag_answer"],
            step_index=1,
            reward_spec=reward_spec,
        )
        self.assertEqual(correct, 1.0)
        self.assertLess(premature, correct)

    def test_unexpected_repeat_loses_sequence_reward(self) -> None:
        expected = {
            "decision_type": "tool",
            "action": "re_question",
            "required_args": {
                "rewrite_reason": "rag_miss",
                "retrieval_round": 2,
                "finish_after_tool": False,
            },
            "arg_contains": {
                "query": ["安全绳", "模型版本"],
                "context_hint": ["安全绳"],
            },
        }
        repeated = score(
            {
                "decision_type": "tool",
                "action": "rag_answer",
                "action_input": {
                    "query": "安全绳 模型版本",
                    "finish_after_tool": False,
                },
            },
            expected,
            full_actions=["rag_answer", "re_question", "rag_answer"],
            step_index=2,
            previous_action="rag_answer",
            reward_spec={"no_repeated_tool": 0.2},
        )
        corrected = score(
            {
                "decision_type": "tool",
                "action": "re_question",
                "action_input": {
                    "query": "安全绳 模型版本",
                    "context_hint": "安全绳",
                    "rewrite_reason": "rag_miss",
                    "retrieval_round": 2,
                    "finish_after_tool": False,
                },
            },
            expected,
            full_actions=["rag_answer", "re_question", "rag_answer"],
            step_index=2,
            previous_action="rag_answer",
            reward_spec={"no_repeated_tool": 0.2},
        )
        self.assertEqual(corrected, 1.0)
        self.assertLess(repeated, corrected)

    def test_wrong_action_cap_blocks_dense_partial_credit(self) -> None:
        expected = {
            "decision_type": "tool",
            "action": "rag_answer",
            "required_args": {"finish_after_tool": False},
            "arg_contains": {"query": ["安全绳", "模型版本"]},
        }
        reward_spec = {
            "json_valid": 0.02,
            "decision_type_valid": 0.03,
            "action_match": 0.75,
            "argument_match": 0.10,
            "finish_after_tool": 0.05,
            "no_forbidden_action": 0.05,
            "wrong_action_cap": 0.20,
        }
        wrong_action = score(
            {
                "decision_type": "tool",
                "action": "re_question",
                "action_input": {
                    "query": "安全绳 模型版本",
                    "finish_after_tool": False,
                },
            },
            expected,
            full_actions=["rag_answer", "re_question", "rag_answer"],
            step_index=1,
            reward_spec=reward_spec,
        )
        correct_action = score(
            {
                "decision_type": "tool",
                "action": "rag_answer",
                "action_input": {
                    "query": "安全绳 模型版本",
                    "finish_after_tool": False,
                },
            },
            expected,
            full_actions=["rag_answer", "re_question", "rag_answer"],
            step_index=1,
            reward_spec=reward_spec,
        )
        self.assertLessEqual(wrong_action, 0.20)
        self.assertGreater(correct_action, 0.90)

    def test_strict_action_match_distinguishes_named_detection_models(self) -> None:
        expected = {
            "decision_type": "tool",
            "action": "qwen_detection",
            "required_args": {"finish_after_tool": True},
            "arg_contains": {"label": ["烟雾"]},
        }
        actual = {
            "decision_type": "tool",
            "action": "rexomni_detection",
            "action_input": {"label": "烟雾", "finish_after_tool": True},
        }
        equivalent_score = score(
            actual,
            expected,
            full_actions=["qwen_detection"],
            step_index=1,
        )
        strict_score = score(
            actual,
            expected,
            full_actions=["qwen_detection"],
            step_index=1,
            reward_spec={"strict_action_match": True, "wrong_action_cap": 0.20},
        )
        self.assertGreater(equivalent_score, 0.90)
        self.assertLessEqual(strict_score, 0.20)

    def test_strict_argument_types_rejects_string_booleans(self) -> None:
        expected = {
            "decision_type": "tool",
            "action": "migration_advisor",
            "required_args": {
                "use_image": True,
                "use_visual_probe": True,
                "finish_after_tool": True,
            },
            "arg_contains": {"user_query": ["烟雾迁移"]},
        }
        string_typed = {
            "decision_type": "tool",
            "action": "migration_advisor",
            "action_input": {
                "user_query": "烟雾迁移",
                "use_image": "true",
                "use_visual_probe": "true",
                "finish_after_tool": "true",
            },
        }
        legacy_score = score(
            string_typed,
            expected,
            full_actions=["migration_advisor"],
            step_index=1,
        )
        strict_score = score(
            string_typed,
            expected,
            full_actions=["migration_advisor"],
            step_index=1,
            reward_spec={"strict_argument_types": True},
        )
        self.assertEqual(legacy_score, 1.0)
        self.assertLess(strict_score, 1.0)


if __name__ == "__main__":
    unittest.main()
