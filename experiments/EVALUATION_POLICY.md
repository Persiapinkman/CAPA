# CAPA Planner Evaluation Policy

## Evidence Tiers

| Tier | Purpose | Allowed claim |
|---|---|---|
| smoke | Verify code, model loading, schema, and artifact writing | runnable only |
| development | Compare prompts, rewards, checkpoints, and error slices | development evidence |
| sealed test | Evaluate a frozen model once on an untouched, versioned test set | generalization evidence |
| end-to-end | Execute real tools, observations, retries, and completion | system-level evidence |

The current `planner_focused_v3` dev split is development-only because it has already influenced model selection.

## Generation Protocol

Development and sealed deterministic evaluations must use:

- `temperature=0`
- `top_p=1`
- `do_sample=false`
- seed recorded in the run provenance
- three repeats with mean, standard deviation, and output agreement
- raw completions preserved outside Git

Local Transformers evaluation is allowed for the V100 research line. Deployment qualification must use the actual serving stack, including vLLM or the configured gateway.

## Experimental Unit and Metrics

The primary experimental unit is `case_id`, not step. Required outputs are:

- case-macro verifier score with a paired case-clustered confidence interval
- category-macro score and per-category counts
- step-weighted score as a secondary diagnostic
- exact pass rate
- action and parameter accuracy when available
- JSON-valid and exact-stop/tail-text rates
- multi-step pass-all rate
- latency, token use, runtime, and peak memory where measurable

Training-method claims require at least three independent training seeds. Three deterministic inference repeats do not replace training-seed replication.

## Required Artifacts

Every non-smoke run must write:

- `config.json`
- `run_record.json`
- `metrics.json`
- per-repeat summaries
- full prediction URIs in the external artifact store
- dataset ID, split, SHA256, git commit/dirty state, command, seed, and environment
- a final `promote`, `reject`, `baseline`, or `inconclusive` decision with rationale

Runs are appended to `experiments/registry.jsonl`. `reports/CURRENT.md` and `reports/leaderboard.csv` are generated from the registry.

## Claim Gates

Do not claim improvement from a single weighted mean. A method-level promotion requires all of the following:

- case-macro 95% interval excludes zero in the favorable direction
- no critical category regresses beyond its predefined tolerance
- JSON and stopping behavior remain within acceptance thresholds
- the result replicates across training seeds
- the sealed test was not used for prompt, reward, checkpoint, or threshold selection
