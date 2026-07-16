# V9 selection runtime incident

Status: resolved before any sealed-test materialization.

## What happened

The first selection-dev invocation used `.venv-train` (PyTorch 2.8.0,
CUDA 12.8, cuDNN 9.19) on Tesla V100 GPUs. Model loading succeeded, but every
first generation raised `CUDNN_STATUS_NOT_INITIALIZED`. The Planner runtime
converted those exceptions into fallback `answerer` decisions, so the rollout
process exited successfully even though all four model outputs were invalid.

The resulting no-promotion payload is retained as
`selection_decision.invalid_cudnn_runtime.json`; it is operational-failure
evidence and must not be interpreted as a model comparison.

## Diagnosis

A one-case probe reproduced the failure in `.venv-train`. The identical model,
adapter, case, seed, decoding configuration, and GPU succeeded in
`.venv-qwen35-grpo` (PyTorch 2.6.0, CUDA 12.4, cuDNN 9.1), producing a valid
two-decision trajectory. This isolates the fault to the evaluation runtime,
not the model or benchmark.

## Corrective controls

- Pin the V9 local selection launcher to `.venv-qwen35-grpo`.
- Count Planner `_planner_metrics.error_type`/`error` values explicitly.
- Hard-fail checkpoint selection if SFT or any candidate contains a runtime
  fallback, even when the rollout process exits with code zero.
- Preserve the failed run and write the corrected run to a new output
  directory.
- Keep both sealed cohorts unmaterialized until a valid selection decision
  promotes a checkpoint.

The corrective source commit is `249ba22`.
