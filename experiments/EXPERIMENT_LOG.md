# Experiment Log Compatibility Entry

The manually maintained experiment log was frozen on 2026-07-12 to remove conflicting sources of truth.

- Current human-readable status: `reports/CURRENT.md`
- Authoritative append-only registry: `experiments/registry.jsonl`
- Current study definitions: `experiments/studies/`
- Immutable run records: `experiments/runs/<run_id>/run_record.json`
- Historical snapshot: `experiments/archive/2026-07-12_legacy_tracking/EXPERIMENT_LOG.md`

Run the following after adding a registry entry:

```bash
python pipelines/experiments/registry_cli.py validate
python pipelines/experiments/registry_cli.py render
```
