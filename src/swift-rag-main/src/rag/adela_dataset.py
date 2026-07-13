import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_id(value: Any) -> str | None:
    text = _normalize_text(value)
    if text is None:
        return None
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _coerce_id(value: Any) -> int | str | None:
    text = _normalize_id(value)
    if text is None:
        return None
    if re.fullmatch(r"\d+", text):
        try:
            return int(text)
        except Exception:
            return text
    return text


def _build_version_fields(version_obj: Any) -> Tuple[str | None, int | None, int | None, int | None, str | None]:
    if not isinstance(version_obj, dict):
        return None, None, None, None, None

    major = version_obj.get("major")
    minor = version_obj.get("minor")
    patch = version_obj.get("patch")
    train_date = _normalize_text(version_obj.get("train_date"))

    if major is None or minor is None or patch is None:
        version_text = None
    else:
        version_text = f"{major}.{minor}.{patch}"

    return version_text, major, minor, patch, train_date


def _build_search_text(record: Dict[str, Any], searchable_fields: Iterable[str]) -> str:
    lines: List[str] = []
    for field in searchable_fields:
        value = record.get(field)
        if value is None:
            continue
        if isinstance(value, list):
            text = " ".join(str(item).strip() for item in value if str(item).strip())
        elif isinstance(value, dict):
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            text = str(value).strip()
        if text:
            lines.append(f"{field}: {text}")
    return "\n".join(lines)


def _build_labels(label_list: Any) -> List[str]:
    normalized = _normalize_text(label_list)
    if normalized is None:
        return []
    return [item.strip() for item in normalized.split("|") if item.strip()]


def _parse_single_adela_payload(
    payload: Dict[str, Any],
    path: Path,
    source_root: Path,
) -> Dict[str, Any]:
    record_obj = payload.get("record")
    base_obj = record_obj if isinstance(record_obj, dict) and record_obj else payload

    version_obj = base_obj.get("version")
    if not isinstance(version_obj, dict):
        version_obj = payload.get("version")
    version_text, major, minor, patch, train_date = _build_version_fields(version_obj)

    command_obj = payload.get("command")
    if isinstance(command_obj, list):
        command = " ".join(str(part) for part in command_obj if str(part).strip())
    else:
        command = _normalize_text(command_obj)

    stderr = _normalize_text(payload.get("stderr"))
    if stderr is not None:
        stderr = stderr.replace("\n", " ").strip() or None

    source_file = path.resolve().relative_to(source_root.resolve()).as_posix()
    row: Dict[str, Any] = {
        "row_id": path.stem,
        "did": _coerce_id(base_obj.get("did") or payload.get("did")),
        "rid": _coerce_id(base_obj.get("rid") or payload.get("rid")),
        "model_name": _normalize_text(base_obj.get("name") or payload.get("name")),
        "name": _normalize_text(base_obj.get("name") or payload.get("name")),
        "type": _normalize_text(base_obj.get("type") or payload.get("type")),
        "platform": _normalize_text(base_obj.get("platform") or payload.get("platform")),
        "status": _normalize_text(base_obj.get("status") or payload.get("status")),
        "version": version_text,
        "version_major": major,
        "version_minor": minor,
        "version_patch": patch,
        "version_train_date": train_date,
        "returncode": payload.get("returncode"),
        "source_file": source_file,
        "source_filename": path.name,
        "command": command,
        "stderr": stderr,
        "model_info": payload.get("model_info"),
        "benchmark_info": payload.get("benchmark_info"),
        "source_path": str(path.resolve()),
    }
    return row


def _merge_csv_fields_into_row(
    row: Dict[str, Any],
    csv_row: Dict[str, Any],
) -> Dict[str, Any]:
    csv_model_name = _normalize_text(csv_row.get("model_name"))
    csv_platform = _normalize_text(csv_row.get("platform"))
    csv_rid = _coerce_id(csv_row.get("rid"))
    csv_did = _coerce_id(csv_row.get("did"))
    label_list = _normalize_text(csv_row.get("label_list"))
    labels = _build_labels(label_list)

    if row.get("model_name") is None and csv_model_name is not None:
        row["model_name"] = csv_model_name
    if row.get("name") is None and csv_model_name is not None:
        row["name"] = csv_model_name
    if row.get("platform") is None and csv_platform is not None:
        row["platform"] = csv_platform
    if row.get("rid") is None and csv_rid is not None:
        row["rid"] = csv_rid
    if row.get("did") is None and csv_did is not None:
        row["did"] = csv_did

    row["csv_model_name"] = csv_model_name
    row["csv_platform"] = csv_platform
    row["label_list"] = label_list
    row["labels"] = labels
    return row


def _build_metadata_index(input_dir: Path) -> Dict[Tuple[str, str], Path]:
    index: Dict[Tuple[str, str], Path] = {}
    for path in sorted(input_dir.glob("*.json")):
        stem = path.stem
        match = re.search(r"_(\d+)_(\d+)$", stem)
        if not match:
            continue
        rid_key, did_key = match.group(1), match.group(2)
        index[(rid_key, did_key)] = path
    return index


def _resolve_metadata_file_path(
    input_dir: Path,
    metadata_index: Dict[Tuple[str, str], Path],
    csv_row: Dict[str, Any],
) -> Optional[Path]:
    rid_key = _normalize_id(csv_row.get("rid"))
    did_key = _normalize_id(csv_row.get("did"))
    if rid_key and did_key:
        indexed = metadata_index.get((rid_key, did_key))
        if indexed is not None:
            return indexed

    model_name = _normalize_text(csv_row.get("model_name")) or ""
    platform = _normalize_text(csv_row.get("platform")) or ""
    if rid_key and did_key and model_name and platform:
        candidates = [
            input_dir / f"{model_name}-{platform}_{rid_key}_{did_key}.json",
            input_dir / f"{model_name}_{platform}_{rid_key}_{did_key}.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

    return None


def _parse_single_adela_file(path: Path, source_root: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _parse_single_adela_payload(payload, path=path, source_root=source_root)


def _build_row_from_csv_record(
    csv_row: Dict[str, Any],
    input_dir: Path,
    metadata_index: Dict[Tuple[str, str], Path],
) -> Optional[Dict[str, Any]]:
    path = _resolve_metadata_file_path(input_dir, metadata_index, csv_row)
    if path is None:
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    row = _parse_single_adela_payload(payload, path=path, source_root=input_dir.parent)
    return _merge_csv_fields_into_row(row, csv_row)


def _iter_csv_rows(csv_path: Path) -> Iterable[Dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def _build_rows_from_csv(
    input_dir: Path,
    csv_path: Path,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    metadata_index = _build_metadata_index(input_dir)

    for csv_row in _iter_csv_rows(csv_path):
        try:
            row = _build_row_from_csv_record(
                csv_row=csv_row,
                input_dir=input_dir,
                metadata_index=metadata_index,
            )
        except Exception:
            continue
        if row is None:
            continue
        rows.append(row)
    return rows


def _build_rows_by_scanning_json(input_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.json")):
        try:
            row = _parse_single_adela_file(path, source_root=input_dir.parent)
        except Exception:
            continue
        row["label_list"] = None
        row["labels"] = []
        rows.append(row)
    return rows


def export_adela_records_to_jsonl(
    input_dir: Path,
    output_path: Path,
    searchable_fields: List[str],
    records_csv_path: Optional[Path] = None,
) -> int:
    input_dir = input_dir.resolve()
    output_path = output_path.resolve()
    if not input_dir.exists():
        raise ValueError(f"ADELA data directory not found: {input_dir}")

    rows: List[Dict[str, Any]] = []
    csv_path = records_csv_path.resolve() if records_csv_path is not None else None
    if csv_path is not None and csv_path.exists():
        rows = _build_rows_from_csv(input_dir=input_dir, csv_path=csv_path)
    if not rows:
        rows = _build_rows_by_scanning_json(input_dir=input_dir)

    for row in rows:
        row["search_text"] = _build_search_text(row, searchable_fields)

    if not rows:
        raise ValueError(f"No valid adela records found under: {input_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    return len(rows)
