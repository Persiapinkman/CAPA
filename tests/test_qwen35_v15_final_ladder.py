from __future__ import annotations

import hashlib
from pathlib import Path

from training.planner_grpo_seed_v1.scripts import audit_qwen35_v15_final_ladder as auditor


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/eval/qwen35_v15_final_ladder.json"


def test_v15_final_config_and_all_assets_are_frozen() -> None:
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == auditor.EXPECTED_CONFIG_SHA256
    config = auditor.validate_config(CONFIG)
    hashes = auditor.validate_assets(config, include_cases=True)
    assert hashes["cases"] == config["sealed_confirmation"]["cases_sha256"]
    assert hashes["candidate_selection"] == config["arms"]["qwen35_4b_grpo_n64"]["selection_receipt_sha256"]


def test_v15_final_geometry_is_exact24() -> None:
    config = auditor.validate_config(CONFIG)
    cases, geometry = auditor.validate_cases(config)
    assert len(cases) == 24
    assert geometry["entity_clusters"] == 6
    assert geometry["badge_cell_counts_sorted"] == [8, 8, 8]
    assert geometry["image_fixture_cell_counts_sorted"] == [12, 12]


def test_v15_runners_forbid_partial_reruns() -> None:
    all_runner = (ROOT / "scripts/run_qwen35_v15_final_all_scopes.sh").read_text()
    local_runner = (ROOT / "scripts/run_qwen35_v15_local_4b_final_eval.sh").read_text()
    larger_runner = (ROOT / "scripts/run_qwen35_v15_35b_final_eval.sh").read_text()
    assert "Refusing to reuse V15 output root" in all_runner
    assert "no selective rerun permitted" in all_runner
    assert "--runs 3" in local_runner and "--runs 3" in larger_runner
    assert "--max-tokens 4096 --temperature 0 --top-p 1" in local_runner
    assert "--max-tokens 4096 --temperature 0 --top-p 1" in larger_runner
    assert "offset=$((shard * 6))" in larger_runner


def test_v15_adjacent_mean_pairs_cover_all_three_ladder_margins() -> None:
    assert auditor.adjacent_pairs([40.0, 80.0, 96.0, 100.0]) == [
        (40.0, 80.0),
        (80.0, 96.0),
        (96.0, 100.0),
    ]
