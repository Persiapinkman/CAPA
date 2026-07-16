from __future__ import annotations

import unittest

from training.public_sft_grpo_v1.scripts.public_math_contract import (
    extract_gsm8k_gold,
    has_strict_gsm8k_format,
    normalize_integer,
    score_gsm8k_completion,
    trim_completion_token_ids,
    weighted_gsm8k_reward,
)


class PublicMathContractTests(unittest.TestCase):
    def test_integer_normalization(self):
        self.assertEqual(normalize_integer("0012"), "12")
        self.assertEqual(normalize_integer("-1,024"), "-1024")
        self.assertIsNone(normalize_integer("12,34"))
        self.assertIsNone(normalize_integer("3.0"))

    def test_gold_requires_unique_final_marker(self):
        self.assertEqual(extract_gsm8k_gold("work\n#### 1,024"), "1024")
        with self.assertRaises(ValueError):
            extract_gsm8k_gold("#### 1\n#### 2")
        with self.assertRaises(ValueError):
            extract_gsm8k_gold("work\n#### 2\ntrailing")

    def test_task_and_format_are_separate(self):
        clean = score_gsm8k_completion("reason\n#### 72", "72")
        self.assertEqual(clean.exact_numeric, 1.0)
        self.assertEqual(clean.strict_format, 1.0)
        self.assertEqual(clean.loose_numeric, 1.0)

        trailing = score_gsm8k_completion("reason\n#### 72\nextra", "72")
        self.assertEqual(trailing.exact_numeric, 1.0)
        self.assertEqual(trailing.strict_format, 0.0)

        duplicated = score_gsm8k_completion("#### 71\n#### 72", "72")
        self.assertEqual(duplicated.exact_numeric, 1.0)
        self.assertEqual(duplicated.strict_format, 0.0)
        self.assertEqual(duplicated.marker_count, 2)

    def test_missing_marker_is_diagnostic_only(self):
        score = score_gsm8k_completion("The final answer is 72.", "72")
        self.assertEqual(score.exact_numeric, 0.0)
        self.assertEqual(score.strict_format, 0.0)
        self.assertEqual(score.loose_numeric, 1.0)

    def test_wrong_answer_can_still_have_format(self):
        score = score_gsm8k_completion("#### 71", "72")
        self.assertEqual(score.exact_numeric, 0.0)
        self.assertEqual(score.strict_format, 1.0)
        self.assertTrue(has_strict_gsm8k_format("#### 71<|im_end|>"))
        self.assertAlmostEqual(weighted_gsm8k_reward("#### 71", "72"), 0.05)

    def test_extreme_or_malformed_values_fail_closed(self):
        self.assertIsNone(normalize_integer("9" * 65))
        score = score_gsm8k_completion("#### NaN", "1")
        self.assertEqual(score.exact_numeric, 0.0)
        self.assertEqual(score.strict_format, 0.0)

    def test_generation_padding_is_not_counted_after_eos(self):
        self.assertEqual(trim_completion_token_ids([7, 8, 99, 0, 0], 99), ([7, 8, 99], True))
        self.assertEqual(trim_completion_token_ids([7, 8, 0, 0], 99), ([7, 8, 0, 0], False))


if __name__ == "__main__":
    unittest.main()
