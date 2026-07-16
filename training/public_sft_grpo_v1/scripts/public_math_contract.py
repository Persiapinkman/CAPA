#!/usr/bin/env python3
"""Strict, auditable contracts for public math SFT and GRPO experiments."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


FINAL_MARKER = "####"
_INTEGER_BODY = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)"
_MARKED_INTEGER_RE = re.compile(rf"(?m)^\s*{re.escape(FINAL_MARKER)}\s*({_INTEGER_BODY})\s*$")
_ANY_INTEGER_RE = re.compile(
    r"(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?!(?:\w|\.\d))"
)
_QWEN_SPECIAL_RE = re.compile(r"<\|(?:im_end|endoftext)\|>")


def normalize_integer(value: str, *, max_digits: int = 64) -> str | None:
    """Return a canonical base-10 integer, rejecting malformed comma grouping."""

    candidate = str(value).strip()
    if not re.fullmatch(_INTEGER_BODY, candidate):
        return None
    compact = candidate.replace(",", "")
    digits = compact.lstrip("+-")
    if not digits or len(digits) > max_digits:
        return None
    return str(int(compact))


def strip_qwen_special_tokens(text: str) -> str:
    """Remove only the Qwen stop/pad strings used by this experiment."""

    return _QWEN_SPECIAL_RE.sub("", str(text)).rstrip()


def trim_completion_token_ids(token_ids: list[int], eos_token_id: int) -> tuple[list[int], bool]:
    """Trim batched-generation padding after the first EOS and report whether EOS occurred."""

    ids = list(token_ids)
    if eos_token_id not in ids:
        return ids, False
    eos_offset = ids.index(eos_token_id)
    return ids[: eos_offset + 1], True


def extract_gsm8k_gold(solution: str) -> str:
    """Extract the unique final integer from an upstream GSM8K solution."""

    text = strip_qwen_special_tokens(solution)
    matches = list(_MARKED_INTEGER_RE.finditer(text))
    if len(matches) != 1 or text[matches[0].end() :].strip():
        raise ValueError("GSM8K gold must end with exactly one '#### integer' line")
    normalized = normalize_integer(matches[0].group(1))
    if normalized is None:
        raise ValueError("GSM8K gold integer is malformed")
    return normalized


def extract_marked_prediction(completion: str) -> str | None:
    """Extract the last valid marked integer; format strictness is scored separately."""

    text = strip_qwen_special_tokens(completion)
    matches = list(_MARKED_INTEGER_RE.finditer(text))
    if not matches:
        return None
    return normalize_integer(matches[-1].group(1))


def extract_loose_last_integer(completion: str) -> str | None:
    """Diagnostic-only fallback: extract the last standalone integer."""

    text = strip_qwen_special_tokens(completion)
    matches = list(_ANY_INTEGER_RE.finditer(text))
    if not matches:
        return None
    return normalize_integer(matches[-1].group(0))


def has_strict_gsm8k_format(completion: str) -> bool:
    """Require one final marker line and no non-whitespace content after it."""

    text = strip_qwen_special_tokens(completion)
    matches = list(_MARKED_INTEGER_RE.finditer(text))
    return len(matches) == 1 and not text[matches[0].end() :].strip()


@dataclass(frozen=True)
class GSM8KScore:
    exact_numeric: float
    strict_format: float
    loose_numeric: float
    parsed_answer: str | None
    loose_answer: str | None
    marker_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "exact_numeric": self.exact_numeric,
            "strict_format": self.strict_format,
            "loose_numeric": self.loose_numeric,
            "parsed_answer": self.parsed_answer,
            "loose_answer": self.loose_answer,
            "marker_count": self.marker_count,
        }


def score_gsm8k_completion(completion: str, gold_answer: str) -> GSM8KScore:
    """Score task correctness separately from output-contract correctness."""

    canonical_gold = normalize_integer(gold_answer)
    if canonical_gold is None:
        raise ValueError(f"invalid GSM8K gold answer: {gold_answer!r}")
    parsed = extract_marked_prediction(completion)
    loose = extract_loose_last_integer(completion)
    return GSM8KScore(
        exact_numeric=float(parsed == canonical_gold),
        strict_format=float(has_strict_gsm8k_format(completion)),
        loose_numeric=float(loose == canonical_gold),
        parsed_answer=parsed,
        loose_answer=loose,
        marker_count=str(completion).count(FINAL_MARKER),
    )


def weighted_gsm8k_reward(
    completion: str,
    gold_answer: str,
    *,
    accuracy_weight: float = 0.95,
    format_weight: float = 0.05,
) -> float:
    """Recommended first reward; diagnostics remain outside the optimization sum."""

    if accuracy_weight < 0 or format_weight < 0:
        raise ValueError("reward weights must be non-negative")
    score = score_gsm8k_completion(completion, gold_answer)
    return accuracy_weight * score.exact_numeric + format_weight * score.strict_format
