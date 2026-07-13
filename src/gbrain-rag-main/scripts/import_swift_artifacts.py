#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gbrain_rag.core.config import get_settings
from gbrain_rag.core.text import compact_json_text
from gbrain_rag.core.types import Chunk
from gbrain_rag.retrieval.embeddings import vector_to_blob
from gbrain_rag.retrieval.store import BrainStore


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Import swift-rag offline artifacts into the GBrain SQLite index."
    )
    parser.add_argument(
        "--swift-root",
        default=str(PROJECT_ROOT.parent / "swift-rag"),
        help="Path to the source swift-rag project.",
    )
    parser.add_argument("--db-path", default=str(settings.INDEX_DB_PATH))
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--documents", action="store_true", default=True)
    parser.add_argument("--structured", action="store_true", default=True)
    parser.add_argument("--commit-every", type=int, default=250)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _doc_chunk_from_artifact(row: dict[str, Any], artifact_path: Path) -> Chunk:
    metadata = dict(row.get("metadata") or {})
    page_label = metadata.get("page_label")
    heading = metadata.get("heading") or metadata.get("header")
    doc_name = str(row.get("doc_name") or artifact_path.stem)
    return Chunk(
        chunk_id=str(row["id"]),
        doc_id=str(row.get("doc_id") or doc_name),
        doc_name=doc_name,
        source_type="document",
        title=str(heading or f"{doc_name} p{page_label}") if page_label is not None else doc_name,
        text=str(row.get("text") or ""),
        index_text=str(row.get("index_text") or row.get("text") or ""),
        block_type=str(metadata.get("content_type") or "text"),
        page_label=page_label,
        source_path=str(PROJECT_ROOT / "data_source" / doc_name),
        metadata={**metadata, "swift_artifact_path": str(artifact_path)},
    )


def import_document_artifacts(store: BrainStore, artifacts_dir: Path, commit_every: int) -> tuple[int, int]:
    chunk_seen: set[str] = set()
    chunk_count = 0
    embedding_count = 0
    pending = 0
    for artifact_path in sorted(artifacts_dir.glob("*.jsonl")):
        with artifact_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                chunk = _doc_chunk_from_artifact(row, artifact_path)
                if chunk.chunk_id not in chunk_seen:
                    store.upsert_chunk(chunk)
                    chunk_seen.add(chunk.chunk_id)
                    chunk_count += 1
                for item in row.get("embeddings") or []:
                    model = item.get("model")
                    embedding = item.get("embedding")
                    if not model or not embedding:
                        continue
                    store.upsert_embedding(chunk.chunk_id, str(model), np.asarray(embedding, dtype=np.float32))
                    embedding_count += 1
                pending += 1
                if pending >= commit_every:
                    store.commit()
                    pending = 0
    store.commit()
    return chunk_count, embedding_count


def _structured_chunk(row: dict[str, Any], source_type: str, data_path: Path) -> Chunk:
    row_id = str(row.get("row_id") or data_path.stem)
    text = str(row.get("search_text") or compact_json_text(row.items()))
    title = str(row.get("model_name") or row.get("name") or row.get("algorithm_name") or row_id)
    return Chunk(
        chunk_id=f"{source_type}::{row_id}",
        doc_id=data_path.stem,
        doc_name=data_path.relative_to(PROJECT_ROOT).as_posix()
        if data_path.is_relative_to(PROJECT_ROOT)
        else data_path.name,
        source_type=source_type,
        title=title,
        text=text,
        index_text=compact_json_text(row.items(), max_value_len=1600),
        block_type="row",
        source_path=str(data_path),
        metadata=dict(row),
    )


def _import_structured_namespace(
    store: BrainStore,
    *,
    swift_root: Path,
    namespace: str,
    data_path: Path,
    artifacts_dir: Path,
    commit_every: int,
) -> tuple[int, int]:
    rows = _load_jsonl(data_path)
    by_row_id = {str(row.get("row_id")): row for row in rows}
    chunk_ids: dict[str, str] = {}
    chunk_count = 0
    for row in rows:
        chunk = _structured_chunk(row, namespace, data_path)
        store.upsert_chunk(chunk)
        chunk_ids[str(row.get("row_id"))] = chunk.chunk_id
        chunk_count += 1
        if chunk_count % commit_every == 0:
            store.commit()

    embedding_count = 0
    imported_models: set[str] = set()
    for meta_path in sorted(artifacts_dir.glob("*.meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        model = str(meta.get("model_name") or "")
        if not model or model in imported_models:
            continue
        matrix_path = Path(str(meta.get("matrix_path") or ""))
        if not matrix_path.exists():
            matrix_path = swift_root / matrix_path
        if not matrix_path.exists():
            continue
        matrix = np.load(matrix_path).astype(np.float32)
        row_ids = [str(row_id) for row_id in meta.get("row_ids") or []]
        if len(row_ids) != matrix.shape[0]:
            continue
        for idx, row_id in enumerate(row_ids):
            if row_id not in by_row_id:
                continue
            chunk_id = chunk_ids.get(row_id)
            if not chunk_id:
                continue
            store.upsert_embedding(chunk_id, model, matrix[idx])
            embedding_count += 1
            if embedding_count % commit_every == 0:
                store.commit()
        imported_models.add(model)
    store.commit()
    return chunk_count, embedding_count


def import_structured_artifacts(store: BrainStore, swift_root: Path, commit_every: int) -> dict[str, tuple[int, int]]:
    data_source = swift_root / "data_source"
    artifact_root = data_source / "embedding_artifacts"
    return {
        "table": _import_structured_namespace(
            store,
            swift_root=swift_root,
            namespace="table",
            data_path=data_source / "tables" / "model_release_records.jsonl",
            artifacts_dir=artifact_root / "tables",
            commit_every=commit_every,
        ),
        "adela": _import_structured_namespace(
            store,
            swift_root=swift_root,
            namespace="adela",
            data_path=data_source / "adela" / "adela_release_records.jsonl",
            artifacts_dir=artifact_root / "adela",
            commit_every=commit_every,
        ),
    }


def main() -> int:
    args = parse_args()
    swift_root = Path(args.swift_root).resolve()
    artifacts_dir = swift_root / "data_source" / "embedding_artifacts" / "documents"
    if not swift_root.exists():
        raise FileNotFoundError(f"swift-rag root not found: {swift_root}")

    store = BrainStore(Path(args.db_path).resolve())
    if args.reset:
        print(f"Resetting index: {store.db_path}")
        store.reset()

    started = time.perf_counter()
    doc_stats = (0, 0)
    structured_stats: dict[str, tuple[int, int]] = {}
    if args.documents:
        doc_stats = import_document_artifacts(store, artifacts_dir, args.commit_every)
        print(f"Imported document artifacts: chunks={doc_stats[0]}, embeddings={doc_stats[1]}")
    if args.structured:
        structured_stats = import_structured_artifacts(store, swift_root, args.commit_every)
        for source, (chunks, embeddings) in structured_stats.items():
            print(f"Imported {source} artifacts: chunks={chunks}, embeddings={embeddings}")

    print(
        "Import complete: "
        f"stored_chunks={store.count_chunks()}, "
        f"sources={store.source_counts()}, "
        f"embeddings={store.embedding_counts()}, "
        f"elapsed={time.perf_counter() - started:.2f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
