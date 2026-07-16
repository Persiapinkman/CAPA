from pathlib import Path

import pytest

from training.planner_grpo_seed_v1.scripts.prepare_planner_retry_safe_end_hard_v9_sealed_test import (
    validate_selection,
)


def _manifest(*, materialized=False):
    return {
        "sealed_test_commitment": {
            "materialized": materialized,
            "rows": 432,
            "entities": 24,
            "sha256": "abc",
        }
    }


def test_validate_selection_requires_promoted_target_and_unopened_commitment(tmp_path):
    adapter = tmp_path / "screen/checkpoint-20"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    selection = {
        "status": "promote",
        "sealed_test_authorized": True,
        "larger_reference_used_for_selection": False,
        "selected": {"checkpoint": 20, "label": "checkpoint-20"},
    }
    result = validate_selection(selection, _manifest(), tmp_path / "screen")
    assert result["checkpoint"] == 20
    assert result["commitment_sha256"] == "abc"

    with pytest.raises(ValueError, match="already materialized"):
        validate_selection(selection, _manifest(materialized=True), tmp_path / "screen")


def test_validate_selection_rejects_reference_contamination(tmp_path):
    selection = {
        "status": "promote",
        "sealed_test_authorized": True,
        "larger_reference_used_for_selection": True,
        "selected": {"checkpoint": 20, "label": "checkpoint-20"},
    }
    with pytest.raises(ValueError, match="contaminated"):
        validate_selection(selection, _manifest(), Path(tmp_path))
