#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="/raid/zkq/projects/CAPA/.venv-qwen35-grpo/bin/python"
CASES="experiments/studies/planner_retry_ladder_v15_confirmation_v1/sealed_data/v15_confirmation_cases.jsonl"
OUT="/raid/zkq/artifacts/CAPA/evals/qwen3_32b_v15_posthoc_comparison_v1/calibration_20260722T072244Z"
API_BASE="http://127.0.0.1:18081/v1"
RUNNER="training/planner_grpo_seed_v1/scripts/run_repeated_planner_grpo_eval.py"

[[ ! -e "${OUT}" ]] || { echo "Refusing to overwrite ${OUT}"; exit 1; }
mkdir -p "${OUT}"
curl --silent --show-error --fail --max-time 20 "${API_BASE}/models" > "${OUT}/models.json"

CAPA_OMIT_MODEL_IMAGE_PAYLOAD=1 DEMO_OPENAI_STREAM=0 \
  "${PYTHON_BIN}" "${RUNNER}" \
  --cases "${CASES}" \
  --out-dir "${OUT}" \
  --report-prefix qwen3_32b_calibration \
  --model qwen3-32b-v15-posthoc \
  --api-base "${API_BASE}" \
  --api-key local-dummy \
  --runs 1 \
  --offset 0 \
  --limit 1 \
  --max-steps 3 \
  --max-tokens 4096 \
  --temperature 0 \
  --top-p 1 \
  --seed 42 \
  --do-sample false \
  --timeout-seconds 600 \
  --openai-timeout-seconds 600 \
  > "${OUT}/runner_stdout.log" 2>&1

"${PYTHON_BIN}" - "${OUT}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
pred = json.loads(next((root / "qwen3_32b_calibration_run1_predictions.jsonl").open()))
assert pred.get("case_id"), "missing case id"
assert not pred.get("errors"), pred.get("errors")
decisions = pred.get("decisions") or []
assert decisions, "empty decisions"
for decision in decisions:
    metrics = decision.get("_planner_metrics") or {}
    assert metrics.get("first_finish_reason") != "length", metrics
reward = json.load((root / "qwen3_32b_calibration_run1_reward.json").open())
assert reward.get("summary", {}).get("cases") == 1
models = json.load((root / "models.json").open())
assert any(x.get("id") == "qwen3-32b-v15-posthoc" for x in models.get("data", []))
print(json.dumps({"status": "pass", "case_id": pred["case_id"], "decisions": len(decisions)}, ensure_ascii=False))
PY
