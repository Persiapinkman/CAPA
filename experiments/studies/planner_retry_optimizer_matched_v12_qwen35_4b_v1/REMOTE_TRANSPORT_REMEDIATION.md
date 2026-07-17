# V12 larger-reference runtime remediation

The first sealed `Qwen3.5-35B-A3B` attempt was an invalid runtime run: each of four shards wrote one row with zero decisions and `planner rollout step timed out`. A first diagnosis attributed this to non-streaming schema handling and was committed as `5bf9971`; that diagnosis was incomplete because its successful and failing probes used different proxy routes.

The corrected root cause is network routing. The evaluation script forced `10.111.32.253` into `NO_PROXY`, while this compute host reaches that gateway through its configured SOCKS proxy. Four fresh direct connections remained in TCP `SYN-SENT` for more than 200 seconds and wrote zero rows. At the same time, proxy-routed models/chat checks returned HTTP 200 in 0.554/1.900 seconds.

With the proxy route inherited, the original non-streaming protocol completed the same real prompt and frozen schema twice in 4.878 s and 4.470 s. Both outputs were valid, byte-identical, and had the same hash as two streaming probes (`5661c014...f2cd3`). No output content or formal score was inspected.

Before the valid retry, the following is frozen:

- restore the original non-streaming transport and 300-second timeout;
- keep model, prompt, schema, temperature, sampling, seed, step/token limits, and verifier unchanged;
- remove only the script's forced gateway `NO_PROXY` override and inherit the configured proxy;
- write all 432 predictions to a new `larger_proxy_retry` directory;
- retain, but never merge or score, both failed runtime attempts;
- compute no sealed result until SFT, GRPO, and the valid larger reference all have complete 432/432 coverage.

The corrected machine-readable audit is `remote_transport_remediation.json`.
