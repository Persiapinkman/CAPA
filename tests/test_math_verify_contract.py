from __future__ import annotations

import unittest

try:
    from training.public_sft_grpo_v1.scripts.math_verify_contract import (
        extract_boxed_spans,
        has_strict_terminal_box,
        normalize_solution_with_terminal_box,
        parse_strict_boxed,
        safe_math_reward,
        score_math_completion,
        vulnerable_math_reward,
    )

    MATH_VERIFY_AVAILABLE = True
except ModuleNotFoundError:
    MATH_VERIFY_AVAILABLE = False


@unittest.skipUnless(MATH_VERIFY_AVAILABLE, "requires the pinned Math-Verify environment")
class MathVerifyContractTests(unittest.TestCase):
    def test_balanced_nested_box_extraction(self):
        spans = extract_boxed_spans(r"work \boxed{\frac{1+\sqrt{5}}{2}}")
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].content, r"\frac{1+\sqrt{5}}{2}")
        with self.assertRaises(ValueError):
            extract_boxed_spans(r"\boxed{\frac{1}{2}")

    def test_solution_normalization_preserves_text_and_creates_one_terminal_box(self):
        source = r"The answer is $\boxed{17}$ square feet."
        normalized, gold_boxed, content = normalize_solution_with_terminal_box(source)
        self.assertIn("$17$ square feet.", normalized)
        self.assertTrue(normalized.endswith(r"\boxed{17}"))
        self.assertEqual(gold_boxed, r"\boxed{17}")
        self.assertEqual(content, "17")
        self.assertEqual(len(extract_boxed_spans(normalized)), 1)
        self.assertTrue(has_strict_terminal_box(normalized))

    def test_normalization_rejects_ambiguous_gold(self):
        with self.assertRaises(ValueError):
            normalize_solution_with_terminal_box(r"\boxed{1} and \boxed{2}")
        with self.assertRaises(ValueError):
            normalize_solution_with_terminal_box("no final answer")

    def test_symbolic_equivalence_and_wrong_answer(self):
        correct = score_math_completion(
            "reason\n" + r"\boxed{0.5}", r"\boxed{\frac{1}{2}}"
        )
        self.assertEqual(correct.symbolic_accuracy, 1.0)
        self.assertEqual(correct.strict_exact, 1.0)
        wrong = score_math_completion(r"\boxed{2}", r"\boxed{\frac{1}{2}}")
        self.assertEqual(wrong.symbolic_accuracy, 0.0)
        self.assertEqual(wrong.strict_format, 1.0)

    def test_multiple_boxes_expose_vulnerable_reward_but_safe_reward_fails_closed(self):
        completion = r"\boxed{0} then \boxed{1}"
        score = score_math_completion(completion, r"\boxed{1}")
        self.assertEqual(score.symbolic_accuracy, 1.0)
        self.assertEqual(score.strict_format, 0.0)
        self.assertEqual(score.strict_exact, 0.0)
        self.assertEqual(safe_math_reward(completion, r"\boxed{1}"), 0.0)
        self.assertEqual(vulnerable_math_reward(completion, r"\boxed{1}"), 0.95)

    def test_trailing_content_and_inline_box_are_not_strict(self):
        self.assertFalse(has_strict_terminal_box(r"answer \boxed{1}"))
        self.assertFalse(has_strict_terminal_box("reason\n" + r"\boxed{1}" + "\nextra"))
        self.assertTrue(
            has_strict_terminal_box("reason\n" + r"\boxed{1}" + "<|im_end|>")
        )

    def test_no_box_echo_nan_and_oversized_values_fail_closed(self):
        self.assertEqual(score_math_completion("the answer is 1", r"\boxed{1}").parse_success, 0.0)
        self.assertEqual(score_math_completion(r"\boxed{NaN}", r"\boxed{1}").symbolic_accuracy, 0.0)
        self.assertEqual(parse_strict_boxed("\\boxed{" + "9" * 1025 + "}"), [])

    def test_verify_argument_order_is_intentionally_asymmetric(self):
        solved = score_math_completion(r"\boxed{(1,2)}", r"\boxed{1<x<2}")
        echoed = score_math_completion(r"\boxed{1<x<2}", r"\boxed{(1,2)}")
        self.assertEqual(solved.symbolic_accuracy, 1.0)
        self.assertEqual(echoed.symbolic_accuracy, 0.0)


if __name__ == "__main__":
    unittest.main()
