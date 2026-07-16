import pytest

from training.planner_grpo_seed_v1.scripts.combine_planner_rollout_prediction_shards import (
    combine,
)


def test_combine_restores_case_order_with_exact_coverage():
    cases = [{"case_id": "a"}, {"case_id": "b"}, {"case_id": "c"}]
    rows = combine(
        cases=cases,
        prediction_shards=[
            [{"case_id": "c", "value": 3}],
            [{"case_id": "a", "value": 1}, {"case_id": "b", "value": 2}],
        ],
    )
    assert [row["value"] for row in rows] == [1, 2, 3]


def test_combine_rejects_duplicate_or_missing_predictions():
    cases = [{"case_id": "a"}, {"case_id": "b"}]
    with pytest.raises(ValueError, match="disjoint"):
        combine(
            cases=cases,
            prediction_shards=[[{"case_id": "a"}], [{"case_id": "a"}]],
        )
    with pytest.raises(ValueError, match="coverage"):
        combine(cases=cases, prediction_shards=[[{"case_id": "a"}]])
