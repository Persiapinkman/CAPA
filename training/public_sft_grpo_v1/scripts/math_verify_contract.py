#!/usr/bin/env python3
"""Strict MATH boxed-answer extraction and Math-Verify scoring contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from training.public_sft_grpo_v1.scripts.public_math_contract import strip_qwen_special_tokens


_BOXED_START_RE = re.compile(r"\\boxed\s*\{")
_STRICT_NORMALIZATION = NormalizationConfig(
    basic_latex=True,
    units=True,
    malformed_operators=False,
    nits=False,
    boxed="all",
    equations=False,
)
_STRICT_LATEX_CONFIG = LatexExtractionConfig(
    try_extract_without_anchor=False,
    boxed_match_priority=0,
    normalization_config=_STRICT_NORMALIZATION,
)


@dataclass(frozen=True)
class BoxedSpan:
    start: int
    end: int
    content: str
    raw: str


def _is_escaped(text: str, offset: int) -> bool:
    backslashes = 0
    offset -= 1
    while offset >= 0 and text[offset] == "\\":
        backslashes += 1
        offset -= 1
    return backslashes % 2 == 1


def extract_boxed_spans(text: str) -> list[BoxedSpan]:
    """Return balanced ``\\boxed{...}`` spans, including nested LaTeX braces."""

    source = str(text)
    spans: list[BoxedSpan] = []
    for match in _BOXED_START_RE.finditer(source):
        open_offset = match.end() - 1
        depth = 0
        for offset in range(open_offset, len(source)):
            character = source[offset]
            if character in "{}" and _is_escaped(source, offset):
                continue
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    end = offset + 1
                    spans.append(
                        BoxedSpan(
                            start=match.start(),
                            end=end,
                            content=source[open_offset + 1 : offset],
                            raw=source[match.start() : end],
                        )
                    )
                    break
        else:
            raise ValueError("unclosed \\boxed expression")
    return spans


def normalize_solution_with_terminal_box(solution: str) -> tuple[str, str, str]:
    """Preserve a one-box solution while moving its only box to a strict final line."""

    source = str(solution)
    spans = extract_boxed_spans(source)
    if len(spans) != 1:
        raise ValueError(f"expected exactly one boxed expression, got {len(spans)}")
    span = spans[0]
    if not span.content.strip():
        raise ValueError("empty boxed answer")
    unboxed_solution = source[: span.start] + span.content + source[span.end :]
    canonical_box = f"\\boxed{{{span.content}}}"
    normalized = unboxed_solution.rstrip() + "\n\n" + canonical_box
    if not has_strict_terminal_box(normalized):
        raise RuntimeError("normalized solution failed terminal-box contract")
    return normalized, canonical_box, span.content


def has_strict_terminal_box(completion: str) -> bool:
    """Require exactly one box, alone on the final line, with no trailing content."""

    text = strip_qwen_special_tokens(completion)
    try:
        spans = extract_boxed_spans(text)
    except ValueError:
        return False
    if len(spans) != 1:
        return False
    span = spans[0]
    if text[span.end :].strip():
        return False
    final_line_start = text.rfind("\n", 0, span.start) + 1
    return text[final_line_start:].strip() == span.raw


def parse_strict_boxed(boxed: str, *, timeout_seconds: int = 2, max_box_chars: int = 1024) -> list[Any]:
    """Parse one boxed expression with no unanchored or string fallback."""

    spans = extract_boxed_spans(boxed)
    if len(spans) != 1 or spans[0].raw.strip() != str(boxed).strip():
        return []
    if not spans[0].content.strip() or len(spans[0].content) > max_box_chars:
        return []
    parsed = parse(
        spans[0].raw,
        extraction_config=[_STRICT_LATEX_CONFIG],
        fallback_mode="no_fallback",
        extraction_mode="first_match",
        parsing_timeout=timeout_seconds,
        raise_on_error=False,
    )
    return list(parsed)


@dataclass(frozen=True)
class MATHScore:
    symbolic_accuracy: float
    strict_format: float
    strict_exact: float
    parse_success: float
    box_count: int
    parsed_prediction: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbolic_accuracy": self.symbolic_accuracy,
            "strict_format": self.strict_format,
            "strict_exact": self.strict_exact,
            "parse_success": self.parse_success,
            "box_count": self.box_count,
            "parsed_prediction": self.parsed_prediction,
        }


def score_math_completion(
    completion: str,
    gold_boxed: str,
    *,
    timeout_seconds: int = 2,
    max_box_chars: int = 1024,
) -> MATHScore:
    """Verify the last prediction box while keeping strict format as a separate metric."""

    gold = parse_strict_boxed(
        gold_boxed, timeout_seconds=timeout_seconds, max_box_chars=max_box_chars
    )
    if not gold:
        raise ValueError(f"unparseable gold boxed answer: {gold_boxed!r}")
    text = strip_qwen_special_tokens(completion)
    try:
        boxes = extract_boxed_spans(text)
    except ValueError:
        boxes = []
    prediction = (
        parse_strict_boxed(
            boxes[-1].raw,
            timeout_seconds=timeout_seconds,
            max_box_chars=max_box_chars,
        )
        if boxes
        else []
    )
    correct = bool(
        prediction
        and verify(
            gold,
            prediction,
            strict=True,
            allow_set_relation_comp=False,
            timeout_seconds=timeout_seconds,
            raise_on_error=False,
        )
    )
    strict_format = has_strict_terminal_box(text)
    return MATHScore(
        symbolic_accuracy=float(correct),
        strict_format=float(strict_format),
        strict_exact=float(correct and strict_format),
        parse_success=float(bool(prediction)),
        box_count=len(boxes),
        parsed_prediction=repr(prediction)[:512] if prediction else None,
    )


def safe_math_reward(
    completion: str,
    gold_boxed: str,
    *,
    accuracy_weight: float = 0.95,
    format_weight: float = 0.05,
) -> float:
    """Safe default: only strict-and-correct answers receive the accuracy component."""

    if accuracy_weight < 0 or format_weight < 0:
        raise ValueError("reward weights must be non-negative")
    score = score_math_completion(completion, gold_boxed)
    return accuracy_weight * score.strict_exact + format_weight * score.strict_format


def vulnerable_math_reward(
    completion: str,
    gold_boxed: str,
    *,
    accuracy_weight: float = 0.95,
    format_weight: float = 0.05,
) -> float:
    """Deliberately weaker R1 arm used only to demonstrate format/repetition hacking."""

    score = score_math_completion(completion, gold_boxed)
    return accuracy_weight * score.symbolic_accuracy + format_weight * score.strict_format
