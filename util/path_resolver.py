"""Path resolution helpers for migrated CAPA local assets."""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_STORAGE_ROOT = Path(os.environ.get("CAPA_STORAGE_ROOT", "/raid/zkq"))
DEFAULT_MODELS_ROOT = Path(os.environ.get("CAPA_MODELS_ROOT", str(DEFAULT_STORAGE_ROOT / "models")))


def resolve_model_name_or_path(value: str, project_root: Path) -> str:
    """Resolve local model paths while preserving remote model ids.

    Rules:
    - absolute paths are kept as-is
    - models/foo resolves to /raid/zkq/models/foo by default
    - ./foo, ../foo, and existing repo paths resolve from the project root
    - bare local model names resolve from /raid/zkq/models when present
    - unresolved Hugging Face style repo ids are kept unchanged
    """

    raw = str(value).strip()
    if not raw or "://" in raw:
        return raw

    path = Path(raw)
    if path.is_absolute():
        return raw

    parts = path.parts
    if parts and parts[0] == "models":
        return str((DEFAULT_STORAGE_ROOT / path).resolve())

    if raw.startswith(("./", "../")):
        return str((project_root / path).resolve())

    repo_candidate = project_root / path
    if repo_candidate.exists():
        return str(repo_candidate.resolve())

    storage_candidate = DEFAULT_STORAGE_ROOT / path
    if len(parts) > 1 and storage_candidate.exists():
        return str(storage_candidate.resolve())

    model_candidate = DEFAULT_MODELS_ROOT / raw
    if "/" not in raw or model_candidate.exists():
        return str(model_candidate)

    return raw
