from pathlib import Path

import pytest

from training.planner_grpo_seed_v1.scripts.prepare_planner_retry_optimizer_matched_v12_sealed_test import (
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


def test_validate_selection_accepts_promoted_v12_checkpoint(tmp_path):
    adapter = tmp_path / "screen/checkpoint-8"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    selection = {
        "status": "promote",
        "sealed_test_authorized": True,
        "larger_reference_used_for_selection": False,
        "selected": {"checkpoint": 8, "label": "checkpoint-8"},
    }
    result = validate_selection(selection, _manifest(), tmp_path / "screen")
    assert result["checkpoint"] == 8
    assert result["commitment_sha256"] == "abc"
    assert len(result["adapter_sha256"]) == 64


def test_validate_selection_rejects_unregistered_or_contaminated_target(tmp_path):
    selection = {
        "status": "promote",
        "sealed_test_authorized": True,
        "larger_reference_used_for_selection": False,
        "selected": {"checkpoint": 10, "label": "checkpoint-10"},
    }
    with pytest.raises(ValueError, match="not preregistered"):
        validate_selection(selection, _manifest(), Path(tmp_path))

    selection["larger_reference_used_for_selection"] = True
    with pytest.raises(ValueError, match="contaminated"):
        validate_selection(selection, _manifest(), Path(tmp_path))
