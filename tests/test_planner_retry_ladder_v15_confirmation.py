from __future__ import annotations

import hashlib
from pathlib import Path

from training.planner_grpo_seed_v1.scripts import (
    build_planner_retry_ladder_v15_confirmation as builder,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "experiments/studies/planner_retry_ladder_v15_confirmation_v1/generation_spec.json"


def test_v15_spec_is_frozen_before_materialization() -> None:
    assert hashlib.sha256(SPEC.read_bytes()).hexdigest() == builder.EXPECTED_SPEC_SHA256
    spec = builder.load_spec()
    assert spec["confirmation_policy"]["open_once"] is True
    assert spec["confirmation_policy"]["partial_or_selective_rerun"] is False
    assert spec["frozen_ladder"]["scenario_weights"] == {
        "post_retry_metric_veto_step3": 111,
        "current_success_step2": 14,
    }


def test_v15_in_memory_geometry_is_exact_scene() -> None:
    rows = builder.build_cases_in_memory()
    audit = builder.validate_cases(rows)
    assert audit["status"] == "pass"
    assert audit["rows"] == 24 and audit["entities"] == 6
    assert audit["factorial_cells"] == 6
    assert audit["canonical_complete_trajectory_pass_rate"] == 1.0
    assert all(row["query_style_index"] == 2 for row in rows)


def test_v15_protected_lexicon_is_disjoint() -> None:
    audit = builder.contamination_audit(builder.load_spec())
    assert audit["status"] == "pass"
    assert audit["protected_tokens"] == 20
    assert audit["exact_overlaps"] == []
    assert audit["model_outputs_used_for_case_selection"] is False
