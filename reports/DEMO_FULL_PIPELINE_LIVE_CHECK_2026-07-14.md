# Demo Full Pipeline Live Check - 2026-07-14

## Verdict

The eight active, non-Adela Demo tools have working runtime paths. Real calls covered Answerer, RAG, Qwen, Rex-Omni, Flux, the complete generation/detection/report pipeline, and migration advisory. Executor failure preconditions are covered without external calls.

This is not yet an RL-data readiness pass. The complete pipeline ran without technical errors, but Flux did not reliably follow the edit constraint and the generated samples were near-duplicates. There is also no human ground truth, so Qwen/Rex overlap is agreement rather than accuracy. These cases must remain blocked from positive RL rewards.

## Live Matrix

| Capability | Actual check | Result |
|---|---|---|
| `answerer` | Planner routed a general question to the direct answer path and returned a non-empty final answer | runtime passed |
| `rag_answer` | GBrain returned a supported `safety_rope v0.2.1` answer at score `0.92` in one retrieval | runtime passed |
| RAG miss/rewrite | Three low-score candidates were withheld; no internal OID/reference noise was exposed | runtime passed |
| `qwen_detection` | Real fisherman image, two boxes, valid annotated image | runtime passed |
| `rexomni_detection` | Same image, two boxes, valid annotated image | runtime passed |
| `flux-image-generation` | Real generation through the Demo, one 1024x1024 JPEG | runtime passed; semantic constraint failed |
| `pipeline_eval` | Three generated images, Qwen+Rex on original plus generated images, four overlays and one report | runtime passed; quality gate failed |
| `migration_advisor` | Four-field RAG retrieval and grounded report completed | runtime passed after evidence hardening |

Adela is intentionally absent. It is not in the default Planner schema, valid action set, capability inventory, service health probe, executor dispatch, or smoke harness. The legacy implementation is enabled only by `CAPA_ENABLE_ADELA=1`.

A clean final-code HTTP run is recorded in `reports/demo_full_e2e_smoke.json`. It completed in 161.7 seconds: Answerer 5.7 s, RAG 26.9 s, Qwen 43.1 s, Rex-Omni 8.8 s, and migration advisor 76.1 s. All five cases routed to the requested action, emitted no error, returned `done.ok=true`, and deleted their synthetic sessions. The migration event exposed four validated facts with `grounding=validated_quote_and_source_id`.

## Flux And Pipeline Evidence

The single Flux Demo run was `20260714_033100_5934d86f`. It produced a valid JPEG after fixing the downloader, which previously saved PNG bytes under a `.jpg` suffix. The generated image nevertheless contained synthetic text/watermark-like marks despite an explicit no-text constraint. Runtime success therefore cannot be used as a semantic reward.

The complete pipeline run was `20260714_033223_b4bf88d0` and completed in 266.94 seconds. It generated three variants, ran both detectors on the source and all variants, produced four annotated images, and emitted no error event. Both models returned two boxes per image. Their mean matched IoU values were 0.7843, 0.8586, 0.7140, and 0.6840.

Those IoUs measure cross-model agreement only. The original report incorrectly invented Qwen accuracy of 80-90% and Rex accuracy of 50-60%. Reporting is now deterministic by default: accuracy is `N/A` without human GT, the recommendation is `inconclusive`, and an optional VLM summary cannot replace computed metrics.

The generated images were insufficiently diverse. Their normalized dHash distances from the source were 0.0703, 0.0703, and 0.0820; pairwise generated distances were 0.0156, 0.0273, and 0.0195. All are below the current diversity thresholds of 0.12 from the source and 0.08 pairwise. The pipeline now emits `generation_quality`/`quality_warning`, forces an inconclusive recommendation, and records `quality_gate_passed=false` for such runs.

The final-code minimal rerun was `20260714_043913_3bb13007`. It made one new Flux call and completed the full HTTP pipeline in 160.98 seconds with no error: one generated image, Qwen and Rex on both source/generated images, overlays, deterministic evaluation, and session cleanup. This sample passed the reference-diversity gate at dHash distance 0.589844. Across the two evaluated images, both models returned four boxes and their aggregate matched-box IoU was 0.849; accuracy remained `N/A` and recommendation `inconclusive`. The run still records `content_compliance_checked=false`, and one successful stochastic sample does not override the earlier three-sample diversity failure.

## Migration Evidence

The initial migration run `20260714_035559_0958e6ab` completed without a runtime error but persisted 41.6 MB in `migration_advisor_report.json`, plus roughly 2-3 MB per retrieval. It also presented unsupported performance, schedule, and cost claims.

The fixed path separates in-memory evidence from bounded audit artifacts. Recompacting the same real retrieval reduced field results from 39.4 MB to 83.9 KB, about 469 times smaller. Rebuilding from cached evidence produced a roughly 105 KB report with four quote-and-source-ID-validated facts and two real candidates:

- `KM_essos_det_small_nart_acl-ascend710-fp16_b1_1.0.0.model`
- `KM_essos_fisher_det_small_nart_cuda11.0-trt7.1-fp16-T4_b1_1.0.0.model`

Both records identify a fishing-person detection algorithm, but their documented application scene is not the requested river-monitoring scene. The corrected conclusion is therefore: candidates exist, direct match is false, feasibility is medium, and performance baseline, target, timeline, and cost are evidence-insufficient. Adela-derived evidence and unrelated high-scoring face/banner/worker records are excluded.

## Reproduction

Start the RAG tunnel, load the proxy/model environment, then run the no-side-effect suite:

```bash
bash pipelines/demo/open_rag_tunnel.sh
source init_env.sh
uv run python pipelines/demo/run_full_demo_smoke.py --include-migration
```

Explicitly authorize cost-bearing generation when needed:

```bash
source init_env.sh
uv run python pipelines/demo/run_full_demo_smoke.py \
  --include-migration --include-flux --include-pipeline --allow-side-effects
```

The harness reports `runtime_status` separately from `rl_readiness`. A technically successful pipeline remains RL-blocked when its generation-quality gate fails.

## RL Gate

The Demo is suitable as an environment-integration baseline, not yet as a source of positive long-horizon RL trajectories. Before GRPO, require all of the following:

1. A scenario with multiple policy-controlled decisions rather than hard-coded transitions.
2. Hidden or independently labeled GT for terminal correctness.
3. Deterministic tool-contract, evidence-grounding, cost, latency, and safety sub-rewards.
4. A generation-quality gate that passes on a held-out prompt/image set, not one example.
5. Frozen test episodes and a multi-seed SFT-versus-GRPO comparison.
