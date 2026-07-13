from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from gbrain_rag.core.config import get_settings
from gbrain_rag.core.text import compact_json_text, stable_id
from gbrain_rag.core.types import Chunk
from gbrain_rag.ingest.chunker import doc_id_for_path, make_chunk, text_to_chunks
from gbrain_rag.ingest.pdf import extract_pdf_chunks


SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt", ".json", ".jsonl", ".csv", ".xlsx"}
SKIP_DIR_NAMES = {"embedding_artifacts", "__pycache__", ".git", ".cache"}


def iter_source_files(data_source_dir: Path) -> Iterable[Path]:
    for path in sorted(data_source_dir.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        # The raw adela/data JSON files are noisy and already normalized into
        # adela_release_records.jsonl by swift-rag.
        if "adela/data" in path.as_posix():
            continue
        yield path


def infer_source_type(path: Path) -> str:
    p = path.as_posix().lower()
    if "adela_release_records" in p:
        return "adela"
    if "model_release_records" in p or path.suffix.lower() in {".csv", ".xlsx"}:
        return "table"
    return "document"


def _relative_doc_name(path: Path, data_source_dir: Path) -> str:
    try:
        return path.relative_to(data_source_dir).as_posix()
    except ValueError:
        return path.name


def load_jsonl_rows(
    path: Path,
    *,
    doc_id: str,
    doc_name: str,
    source_type: str,
) -> list[Chunk]:
    settings = get_settings()
    if source_type == "adela":
        searchable = settings.ADELA_SEARCHABLE_FIELDS
    else:
        searchable = settings.TABLE_SEARCHABLE_FIELDS
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_idx, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            row_id = str(row.get("row_id") or f"{path.stem}-{row_idx:05d}")
            if row.get("search_text"):
                text = str(row["search_text"])
            else:
                text = compact_json_text((field, row.get(field)) for field in searchable)
            if not text:
                text = compact_json_text(row.items())
            title = row.get("model_name") or row.get("name") or row.get("algorithm_name") or row_id
            metadata = dict(row)
            if source_type == "adela" and row.get("did"):
                metadata["reference"] = settings.ADELA_DEPLOYMENT_URL_TEMPLATE.format(did=row["did"])
            chunks.append(
                make_chunk(
                    doc_id=doc_id,
                    doc_name=doc_name,
                    source_type=source_type,
                    text=text,
                    index_text=compact_json_text(row.items(), max_value_len=1600),
                    block_type="row",
                    title=str(title),
                    source_path=str(path),
                    metadata=metadata,
                    ordinal=row_idx,
                )
            )
    return chunks


def load_csv_rows(path: Path, *, doc_id: str, doc_name: str, source_type: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader, start=1):
            text = compact_json_text(row.items())
            if not text:
                continue
            chunks.append(
                make_chunk(
                    doc_id=doc_id,
                    doc_name=doc_name,
                    source_type=source_type,
                    text=text,
                    index_text=text,
                    block_type="row",
                    title=str(row.get("model_name") or row.get("name") or f"row {row_idx}"),
                    source_path=str(path),
                    metadata=dict(row),
                    ordinal=row_idx,
                )
            )
    return chunks


def load_xlsx_rows(path: Path, *, doc_id: str, doc_name: str, source_type: str) -> list[Chunk]:
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("Excel ingestion requires pandas and openpyxl.") from exc

    chunks: list[Chunk] = []
    sheets = pd.read_excel(path, sheet_name=None)
    ordinal = 0
    for sheet_name, df in sheets.items():
        df = df.fillna("")
        for row_idx, row in df.iterrows():
            payload = {str(key): value for key, value in row.to_dict().items()}
            text = compact_json_text(payload.items())
            if not text:
                continue
            ordinal += 1
            chunks.append(
                make_chunk(
                    doc_id=doc_id,
                    doc_name=doc_name,
                    source_type=source_type,
                    text=text,
                    index_text=text,
                    block_type="row",
                    title=str(payload.get("model_name") or payload.get("算法名称") or f"{sheet_name} row {row_idx + 1}"),
                    source_path=str(path),
                    metadata={"sheet_name": sheet_name, "source_row_number": int(row_idx) + 2, **payload},
                    ordinal=ordinal,
                )
            )
    return chunks


def load_file(path: Path, data_source_dir: Path) -> list[Chunk]:
    doc_name = _relative_doc_name(path, data_source_dir)
    doc_id = doc_id_for_path(path, data_source_dir)
    source_type = infer_source_type(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return extract_pdf_chunks(
            path,
            doc_id=doc_id,
            doc_name=doc_name,
            source_type=source_type,
            source_path=str(path),
        )
    if suffix == ".jsonl":
        return load_jsonl_rows(path, doc_id=doc_id, doc_name=doc_name, source_type=source_type)
    if suffix == ".csv":
        return load_csv_rows(path, doc_id=doc_id, doc_name=doc_name, source_type=source_type)
    if suffix == ".xlsx":
        return load_xlsx_rows(path, doc_id=doc_id, doc_name=doc_name, source_type=source_type)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        text = compact_json_text(payload.items()) if isinstance(payload, dict) else json.dumps(payload, ensure_ascii=False)
        return text_to_chunks(
            doc_id=doc_id,
            doc_name=doc_name,
            source_type=source_type,
            text=text,
            title=path.stem,
            source_path=str(path),
            metadata={"file_type": "json"},
        )

    text = path.read_text(encoding="utf-8", errors="ignore")
    return text_to_chunks(
        doc_id=doc_id,
        doc_name=doc_name,
        source_type=source_type,
        text=text,
        title=path.stem,
        source_path=str(path),
        metadata={"file_type": suffix.lstrip(".")},
    )


def load_corpus(data_source_dir: Path, *, limit: int | None = None) -> Iterable[tuple[Path, list[Chunk]]]:
    count = 0
    for path in iter_source_files(data_source_dir):
        yield path, load_file(path, data_source_dir)
        count += 1
        if limit is not None and count >= limit:
            break
