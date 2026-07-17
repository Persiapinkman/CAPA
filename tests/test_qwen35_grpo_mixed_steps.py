import argparse
import hashlib
import json
import sys
from collections import deque

import pytest

from training.planner_grpo_seed_v1.scripts.train_qwen35_4b_grpo import (
    NONTHINKING_SUFFIX,
    flush_reward_extrema,
    load_step_data,
    metric_tail,
    parse_step_indices,
    parse_args,
    record_reward_extrema,
)


class _CharacterTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": list(range(len(text)))}


def _write_mixed_step_fixture(tmp_path):
    path = tmp_path / "mixed.jsonl"
    rows = []
    for step_index in (2, 3):
        prompt = f"prompt-{step_index}" + NONTHINKING_SUFFIX
        rows.append(
            {
                "dataset_id": "mixed-step-test",
                "step_index": step_index,
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "prompt_token_count": len(prompt),
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".manifest.json").write_text(
        json.dumps({"sha256": {"step_data": digest}}) + "\n",
        encoding="utf-8",
    )
    return path


def test_parse_step_indices_is_explicit_and_deduplicated():
    assert parse_step_indices("3,2,3") == (2, 3)
    with pytest.raises(argparse.ArgumentTypeError, match="positive integers"):
        parse_step_indices("0")


def test_safety_reward_weight_is_explicit_and_defaults_off(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_qwen35_4b_grpo.py"])
    assert parse_args().no_forbidden_action_reward_weight == 0.0
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_qwen35_4b_grpo.py",
            "--no-forbidden-action-reward-weight",
            "0.2",
        ],
    )
    assert parse_args().no_forbidden_action_reward_weight == 0.2


def test_metric_tail_supports_trl_deque_buffers():
    assert metric_tail([0.1, 0.2, 0.3], 1) == [0.2, 0.3]
    assert metric_tail(deque((0.1, 0.2, 0.3)), 2) == [0.3]


def test_reward_extrema_span_all_generation_micro_batches():
    store = {}
    metrics = {"train": {}}
    record_reward_extrema(store, "train", {"reward_min": 0.4, "reward_max": 0.8})
    record_reward_extrema(store, "train", {"reward_min": 0.2, "reward_max": 0.7})
    flush_reward_extrema(store, metrics, "train")
    assert metrics["train"]["reward_min"] == [0.2]
    assert metrics["train"]["reward_max"] == [0.8]
    assert store["train"] == {"minimums": [], "maximums": []}


def test_load_step_data_accepts_only_declared_steps(tmp_path):
    path = _write_mixed_step_fixture(tmp_path)
    dataset, stats = load_step_data(
        path,
        _CharacterTokenizer(),
        1000,
        expected_dataset_id="mixed-step-test",
        expected_rows=2,
        allowed_step_indices=(2, 3),
    )
    assert len(dataset) == 2
    assert stats["allowed_step_indices"] == [2, 3]
    assert stats["step_index_counts"] == {"2": 1, "3": 1}

    with pytest.raises(ValueError, match="invalid dataset or target step"):
        load_step_data(
            path,
            _CharacterTokenizer(),
            1000,
            expected_dataset_id="mixed-step-test",
            expected_rows=2,
            allowed_step_indices=(2,),
        )
