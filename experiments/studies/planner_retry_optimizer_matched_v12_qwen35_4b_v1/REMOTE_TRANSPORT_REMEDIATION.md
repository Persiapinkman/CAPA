# V12 larger-reference transport remediation

The first sealed `Qwen3.5-35B-A3B` attempt was stopped as an invalid runtime run: each of four shards wrote exactly one row, every row had zero decisions, and every error was `planner rollout step timed out`. No larger-reference decision or score was available, and neither target model had been scored.

The fault was isolated to the shared gateway's non-streaming transport for the frozen complex JSON schema. With the same real prompt, model, schema, temperature, sampling settings, seed, and 2048-token limit, streaming completed twice in 5.097 s and 4.732 s. Both outputs were valid JSON and byte-identical (`5661c014...f2cd3`). A fake-client regression also confirmed that the new transport path reassembles the same content as the non-streaming path.

Before the formal retry, the following remediation is frozen:

- keep every inference and scoring parameter unchanged;
- set `DEMO_OPENAI_STREAM=1` and extend only the infrastructure timeout from 300 s to 900 s;
- write all 432 reference predictions to a new `larger_stream_retry` directory;
- retain, but never merge or score, the four timeout rows;
- compute no sealed result until SFT, GRPO, and the valid larger reference all have complete 432/432 coverage.

The machine-readable audit is `remote_transport_remediation.json`.
