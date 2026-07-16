# Demo Capability Live Check - 2026-07-14

> Scope: initial read-only RAG plus single-image Qwen and Rex-Omni detection. The later Flux/full-pipeline phase is documented in `reports/DEMO_FULL_PIPELINE_LIVE_CHECK_2026-07-14.md`; Adela is excluded from the current scope.

## Conclusion

RAG, Qwen detection and Rex-Omni detection are now usable through the actual Demo Agent path. The verification covered service health, real semantic responses, Planner routing, executor parsing, persisted observation and annotated-image output.

| Capability | Service check | Demo end-to-end | Result |
|---|---|---|---|
| GBrain RAG `6061` | health and generated query passed | one `rag_answer`, score `0.92`, no retry | passed |
| ACE RAG `6062` | health and structured query passed | available as playbook endpoint; Demo currently uses unified `6061` | passed |
| Qwen `9012` | model list and real image inference passed | selected Qwen, 2 boxes, annotated image, no error | passed |
| Rex-Omni | model list and real image inference passed | selected Rex, 2 boxes, annotated image, no error | passed |

## Network Setup

The RAG host is reached by SSH through the local SOCKS proxy. Local forwards preserve the Demo defaults:

```bash
bash pipelines/demo/open_rag_tunnel.sh
```

Model and detection requests use the proxy and endpoint variables in `init_env.sh`:

```bash
source init_env.sh
```

No password or additional credential was written to the repository.

## RAG Evidence

Both health endpoints returned `status=ok`. GBrain reported 5,673 indexed chunks: 3,745 document, 954 table and 974 Adela chunks. ACE reported eight active playbook items and confirmed that its GBrain dependency was reachable.

The GBrain generated query asked what `safety_rope v0.2.1` outputs. It returned the four supported classification outcomes with evidence citations, `knowledge_base_fully_answered=0.92`, and no query-expansion error. The direct request completed in about 12.3 seconds.

The ACE query asked for a T4 deployment and deployment identifier. It selected the Adela structured source, returned one concrete deployment record with evidence, and completed in about 2.4 seconds. Query expansion used the LLM successfully.

Before the service restart, GBrain inherited an invalid HTTP proxy and degraded to evidence snippets with score `0`. The restart removed that failure mode; the post-restart results above are authoritative.

## Detection Evidence

The fixture was `examples/images/fisherman.jpg`, a 1920x1080 image with two people fishing at the lower right.

Qwen returned two absolute-coordinate boxes:

```json
[[1340, 775, 1486, 981], [1432, 862, 1584, 1051]]
```

Rex-Omni returned two COCO `[x, y, width, height]` boxes:

```json
[[1333.81, 763.24, 142.22, 206.49], [1422.22, 849.73, 146.07, 189.19]]
```

All boxes were positive, inside image bounds and visually aligned with the two people. The raw skill checks and the Demo executor checks agreed on a count of two.

## Demo Contract Fix

The first live Demo RAG run exposed a client-side mismatch: GBrain defines fully supported answers as scores in the `0.85-1.0` band, while CAPA used a default hit threshold of `0.97`. A correct `0.92` answer was therefore treated as a miss, causing three redundant retrievals and an incorrect migration-advisor offer.

The default `DEMO_KB_ANSWER_THRESHOLD` fallback is now `0.85`. It remains configurable by environment variable. After the fix, the same query produced exactly one `rag_answer`, returned the cited answer and did not emit a migration offer. Boundary tests cover `0.85` as hit and `0.84` as miss.

The Demo previously launched skills through a hard-coded system `python3`, which can differ from the interpreter that runs the server. All skill subprocesses now use `sys.executable`. A clean start using only `source init_env.sh` and `.venv/bin/python demo/demo_server.py` was rechecked: the Qwen subprocess resolved to the project virtual environment and again returned two boxes with no error.

## Remaining Boundaries

- This initial phase did not call Flux or `pipeline_eval`; both were subsequently executed and audited in `reports/DEMO_FULL_PIPELINE_LIVE_CHECK_2026-07-14.md`.
- Adela is no longer part of the default Demo or Agent RL action space.
- Service health is not sufficient evidence by itself; the semantic smoke cases in this report should remain part of future readiness checks.
