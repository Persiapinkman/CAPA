# Artifact Migration: 2026-07-12

No artifact content was deleted. Large and regenerated directories were moved from the Git workspace to `/raid/zkq/artifacts/CAPA`; compatibility symlinks preserve active runtime paths.

| Destination | Files | Size (MiB) | Contents |
|---|---:|---:|---|
| `outputs/` | 628 | 31,825.81 | checkpoints, merged models, adapters |
| `runtime/` | 165,792 | 27,030.03 | three local Python environments |
| `cache/` | 37 | 1,924.59 | Hugging Face and codegraph caches |
| `traces/llm_debug/` | 2,162 | 34.20 | historical LLM request/response traces |
| `legacy/training_reports/` | 1,342 | 2.65 | generated prompt contexts and raw rollout trees |
| `snapshots/test_skills_original/` | 2,518 | 81.97 | vendored predecessor repository snapshot |

Small source datasets, dataset cards, run summaries, comparison reports, and error audits remain versioned in Git.

## Auditable Tree Digests

The digest is SHA256 over sorted `relative_path NUL file_sha256 LF` records.

| Tree | Files | Digest |
|---|---:|---|
| `legacy/training_reports/` | 1,342 | `3c58a171b5f00f611bf7f578f3e6e4104aefa47022478d65ce078d88b2ce2aaa` |
| `snapshots/test_skills_original/` | 2,518 | `dbfc530a8c1aa2f0a3d0adecbd8cc9d690c5bb11b1a48bbe75fd175f74ad0104` |
