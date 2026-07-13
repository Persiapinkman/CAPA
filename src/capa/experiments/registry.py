"""Versioned, append-only experiment registry utilities."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "2.0"
VALID_STATUSES = {"planned", "running", "completed", "failed", "stopped", "rejected"}
REQUIRED_FIELDS = {
    "schema_version",
    "run_id",
    "study_id",
    "date",
    "kind",
    "status",
    "purpose",
    "hypothesis",
    "parent_run_id",
    "provenance",
    "data",
    "method",
    "metrics",
    "artifacts",
    "decision",
}


class RegistryError(ValueError):
    """Raised when a registry entry is malformed or conflicts with existing data."""


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RegistryError(f"{path} must contain a JSON object")
    return value


def load_registry(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RegistryError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise RegistryError(f"{path}:{line_number}: entry must be an object")
        rows.append(value)
    return rows


def validate_entry(entry: dict[str, Any], *, strict: bool = True) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_FIELDS - set(entry))
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")
    if str(entry.get("schema_version")) != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if not str(entry.get("run_id") or "").strip():
        errors.append("run_id must be non-empty")
    if not str(entry.get("study_id") or "").strip():
        errors.append("study_id must be non-empty")
    if entry.get("status") not in VALID_STATUSES:
        errors.append(f"status must be one of {sorted(VALID_STATUSES)}")
    for field in ("provenance", "data", "method", "metrics", "artifacts", "decision"):
        if field in entry and not isinstance(entry[field], dict):
            errors.append(f"{field} must be an object")

    if strict:
        provenance = entry.get("provenance") if isinstance(entry.get("provenance"), dict) else {}
        for field in ("git_commit", "command", "seed", "environment"):
            if field not in provenance:
                errors.append(f"provenance.{field} is required")
        data = entry.get("data") if isinstance(entry.get("data"), dict) else {}
        for field in ("dataset_id", "split", "files"):
            if field not in data:
                errors.append(f"data.{field} is required")
        decision = entry.get("decision") if isinstance(entry.get("decision"), dict) else {}
        for field in ("outcome", "rationale"):
            if field not in decision:
                errors.append(f"decision.{field} is required")
    return errors


def validate_registry(rows: Iterable[dict[str, Any]], *, strict: bool = True) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        run_id = str(row.get("run_id") or "")
        for message in validate_entry(row, strict=strict):
            errors.append(f"entry {index} ({run_id or '<missing>'}): {message}")
        if run_id in seen:
            errors.append(f"entry {index}: duplicate run_id {run_id!r}")
        seen.add(run_id)
    return errors


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def append_entry(path: Path, entry: dict[str, Any], *, strict: bool = True) -> None:
    errors = validate_entry(entry, strict=strict)
    if errors:
        raise RegistryError("; ".join(errors))
    rows = load_registry(path)
    run_id = str(entry["run_id"])
    if any(str(row.get("run_id")) == run_id for row in rows):
        raise RegistryError(f"run_id already exists: {run_id}")
    rows.append(entry)
    atomic_write_jsonl(path, rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def legacy_entry_to_v2(entry: dict[str, Any]) -> dict[str, Any]:
    """Preserve a legacy row while making every missing provenance field explicit."""
    training = entry.get("training") if isinstance(entry.get("training"), dict) else {}
    generation = entry.get("generation") if isinstance(entry.get("generation"), dict) else {}
    environment = entry.get("environment") if isinstance(entry.get("environment"), dict) else {}
    status = str(entry.get("status") or "completed")
    outcome = "reject" if status in {"failed", "stopped", "rejected"} else "historical"
    data_files = {
        key: entry[key]
        for key in ("cases", "train_file", "eval_file", "source_cases", "source_split")
        if entry.get(key)
    }
    reports = entry.get("reports") if isinstance(entry.get("reports"), dict) else {}
    notes = str(entry.get("notes") or "Legacy record; purpose was not captured separately.")
    model = entry.get("model") or entry.get("base_model") or "unknown"
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(entry.get("run_id") or "unknown"),
        "study_id": "qwen25_post_training_legacy_v1",
        "date": str(entry.get("date") or "unknown"),
        "kind": str(entry.get("kind") or "legacy"),
        "status": status if status in VALID_STATUSES else "completed",
        "purpose": notes,
        "hypothesis": "unknown_not_recorded_in_legacy_manifest",
        "parent_run_id": None,
        "provenance": {
            "git_commit": "unknown_not_recorded",
            "command": "unknown_not_recorded",
            "seed": training.get("seed", generation.get("seed", "unknown_not_recorded")),
            "environment": environment or {"source": "legacy_manifest"},
        },
        "data": {
            "dataset_id": "legacy_unversioned",
            "split": "unknown_not_recorded",
            "files": data_files,
            "sha256": {},
        },
        "method": {
            "model": model,
            "adapter_path": entry.get("adapter_path") or entry.get("output_dir") or "",
            "training": training,
            "generation": generation,
            "legacy_kind": entry.get("kind"),
        },
        "metrics": entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {},
        "artifacts": reports,
        "decision": {"outcome": outcome, "rationale": notes},
        "legacy_source": "experiments/manifest.jsonl",
    }
