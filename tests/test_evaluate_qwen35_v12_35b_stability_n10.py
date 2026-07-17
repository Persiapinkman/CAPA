import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from training.planner_grpo_seed_v1.scripts.evaluate_qwen35_v12_35b_stability_n10 import (
    action_signature,
    combine_primary_run,
    evaluate,
    operational_audit,
    weighted_rate,
)


METRIC = "post_retry_metric_veto_step3"
CURRENT = "current_success_step2"


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decision(action="final_answer", *, finish_reason="stop"):
    return {
        "decision_type": "end",
        "action": action,
        "action_input": {},
        "end_reason": "memory_hit",
        "final_answer": "",
        "clarification_question": "",
        "_planner_metrics": {"first_finish_reason": finish_reason},
    }


def prediction(case_id, **kwargs):
    return {
        "case_id": case_id,
        "decisions": [decision(**kwargs)],
        "errors": [],
    }


def test_weighted_rate_is_ten_to_one_over_whole_scenario_rates():
    table = {
        METRIC: {"cases": 48, "passed": 43},
        CURRENT: {"cases": 48, "passed": 48},
    }
    assert weighted_rate(
        table,
        metric_scenario=METRIC,
        current_scenario=CURRENT,
        metric_weight=10,
        current_weight=1,
    ) == Fraction(239, 264)


def test_operational_audit_counts_empty_runtime_transport_and_clipping():
    rows = [
        prediction("ok"),
        prediction("clip", finish_reason="length"),
        {"case_id": "timeout", "decisions": [], "errors": ["planner rollout step timed out"]},
        {"case_id": "bug", "decisions": [], "errors": ["unexpected assertion"]},
    ]
    result = operational_audit(rows)
    assert result["prediction_rows"] == 4
    assert result["empty_decision_cases"] == 2
    assert result["runtime_error_cases"] == 2
    assert result["transport_failure_cases"] == 1
    assert result["transport_failure_case_ids"] == ["timeout"]
    assert result["clipped_cases"] == 1


def test_action_signature_ignores_thought_and_telemetry_but_not_actions():
    left = prediction("x")
    right = prediction("x")
    right["decisions"][0]["thought"] = "different"
    right["decisions"][0]["_planner_metrics"]["first_api_call_ms"] = 99
    assert action_signature(left) == action_signature(right)
    right["decisions"][0]["action"] = "migration_advisor"
    assert action_signature(left) != action_signature(right)


def test_combine_primary_run_enforces_each_fixed_slice(tmp_path):
    cases = [{"case_id": f"c{i:03d}"} for i in range(96)]
    raw = tmp_path / "raw"
    for shard in range(4):
        rows = [prediction(f"c{i:03d}") for i in range(shard * 24, (shard + 1) * 24)]
        write_jsonl(
            raw / f"shard{shard}" / f"p_shard{shard}_run1_predictions.jsonl",
            rows,
        )
    combined, audit = combine_primary_run(
        cases=cases,
        raw_root=raw,
        prefix="p",
        run=1,
        shards=4,
        shard_size=24,
    )
    assert [row["case_id"] for row in combined] == [row["case_id"] for row in cases]
    assert [item["rows"] for item in audit] == [24, 24, 24, 24]

    bad_path = raw / "shard2" / "p_shard2_run1_predictions.jsonl"
    bad = json.loads(bad_path.read_text(encoding="utf-8").splitlines()[0])
    bad["case_id"] = "wrong"
    rows = [bad] + [prediction(f"c{i:03d}") for i in range(49, 72)]
    write_jsonl(bad_path, rows)
    with pytest.raises(ValueError, match="coverage mismatch"):
        combine_primary_run(
            cases=cases,
            raw_root=raw,
            prefix="p",
            run=1,
            shards=4,
            shard_size=24,
        )


def test_end_to_end_three_run_success_and_refuse_overwrite(tmp_path):
    cases = []
    for index in range(96):
        scenario = METRIC if index < 48 else CURRENT
        cases.append(
            {
                "case_id": f"c{index:03d}",
                "scenario_id": scenario,
                "category": scenario,
                "expected_decisions": [
                    {
                        "decision_type": "end",
                        "required_args": {"end_reason": "memory_hit"},
                    }
                ],
                "forbidden_actions": [],
                "reward_spec": {"strict_action_match": True},
            }
        )
    cases_path = tmp_path / "cases.jsonl"
    write_jsonl(cases_path, cases)
    config = {
        "schema_version": "1.0",
        "study_id": "test-n10",
        "evidence_role": "open_dev_test_fixture",
        "held_out_or_new_test_allowed": False,
        "cases": {"path": str(cases_path), "rows": 96, "sha256": digest(cases_path)},
        "model": {"id": "Qwen3.5-35B-A3B", "api_base": "fixture"},
        "protocol": {
            "temperature": 0.0,
            "top_p": 1.0,
            "do_sample": False,
            "seed": 42,
            "max_steps": 3,
            "max_tokens": 320,
            "timeout_seconds": 300,
            "openai_timeout_seconds": 300,
            "omit_model_image_payload": True,
        },
        "execution": {
            "repetitions": 3,
            "fixed_shards": 4,
            "cases_per_shard": 24,
            "report_prefix": "p",
        },
        "scenario_strata": {"metric_veto": METRIC, "current": CURRENT},
        "weighted_mixture": {
            "metric_veto_weight": 10,
            "current_weight": 1,
            "label": "10:1",
        },
        "hard_gates": {
            "minimum_weighted_rate_percent_exclusive": 85.0,
            "maximum_repetition_range_pp_inclusive": 2.0,
            "minimum_pairwise_strict_pass_fail_agreement_percent_inclusive": 95.0,
            "required_rows_per_repetition": 96,
            "maximum_empty_decision_cases": 0,
            "maximum_runtime_error_cases": 0,
            "maximum_clipped_cases": 0,
        },
        "dev_score_pp": "fixture",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    raw = tmp_path / "raw"
    for run in (1, 2, 3):
        for shard in range(4):
            rows = [
                prediction(f"c{i:03d}")
                for i in range(shard * 24, (shard + 1) * 24)
            ]
            write_jsonl(
                raw / f"shard{shard}" / f"p_shard{shard}_run{run}_predictions.jsonl",
                rows,
            )
    analysis = tmp_path / "analysis"
    report = evaluate(
        config_path=config_path,
        raw_root=raw,
        analysis_dir=analysis,
    )
    assert report["hard_success"] is True
    assert report["stability"]["weighted_rates_percent"] == [100.0, 100.0, 100.0]
    assert report["stability"]["max_minus_min_pp"] == 0.0
    assert report["dev_score"]["score_pp"] == 2.0
    assert report["raw_retry_accounting"]["primary_prediction_rows"] == 288
    assert report["raw_retry_accounting"]["retry_prediction_rows"] == 0
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        evaluate(
            config_path=config_path,
            raw_root=raw,
            analysis_dir=analysis,
        )
