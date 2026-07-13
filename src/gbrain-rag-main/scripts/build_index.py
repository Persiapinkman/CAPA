#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Ensure logs stay UTF-8 even in non-interactive shells / tmux sessions.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from gbrain_rag.core.config import get_settings
from gbrain_rag.ingest.loaders import iter_source_files, load_file
from gbrain_rag.retrieval.embeddings import EmbeddingManager
from gbrain_rag.retrieval.store import BrainStore


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build the local GBrain RAG SQLite index.")
    parser.add_argument("--data-source", default=str(settings.DATA_SOURCE_DIR))
    parser.add_argument("--db-path", default=str(settings.INDEX_DB_PATH))
    parser.add_argument("--embedding-model", default=settings.EMBEDDING_MODEL)
    parser.add_argument("--embedding-backend", default=settings.EMBEDDING_BACKEND)
    parser.add_argument(
        "--embedding-device",
        default=None,
        help="Embedding device override, e.g. cpu / cuda / cuda:0.",
    )
    parser.add_argument("--reset", action="store_true", help="Drop and rebuild the SQLite brain.")
    parser.add_argument("--no-embeddings", action="store_true", help="Only build keyword/graph index.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of source files for smoke tests.")
    parser.add_argument("--commit-every", type=int, default=100)
    parser.add_argument(
        "--embedding-max-chars",
        type=int,
        default=2200,
        help="Truncate per-chunk embedding input to this many characters (0 disables truncation).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-file and batch timing details.")
    return parser.parse_args()


def build_embedding_text(chunk, max_chars: int) -> str:
    parts = [chunk.doc_name, chunk.title or "", chunk.text]
    context = (chunk.index_text or "").strip()
    if context and context != chunk.text:
        parts.append(context)
    text = "\n".join(part for part in parts if part).strip()
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars]
    return text


def main() -> int:
    args = parse_args()
    data_source = Path(args.data_source).resolve()
    db_path = Path(args.db_path).resolve()
    if not data_source.exists():
        raise FileNotFoundError(f"Data source directory not found: {data_source}")

    store = BrainStore(db_path)
    if args.reset:
        print(f"Resetting index: {db_path}")
        store.reset()

    embedding_backend = None
    if not args.no_embeddings:
        manager = EmbeddingManager()
        if args.embedding_device:
            manager.settings.EMBEDDING_DEVICE = args.embedding_device
        embedding_backend = manager.get(args.embedding_model, args.embedding_backend)
        device = getattr(embedding_backend, "device", "n/a")
        print(
            f"Embedding backend: {embedding_backend.__class__.__name__}, "
            f"model={embedding_backend.model_name}, dim={embedding_backend.dim}, device={device}"
        )

    started = time.perf_counter()
    total_files = 0
    total_chunks = 0
    failed: list[tuple[str, str]] = []
    batch_texts: list[str] = []
    batch_chunks = []

    def flush_batch(reason: str = "") -> None:
        nonlocal batch_texts, batch_chunks, total_chunks
        if not batch_chunks:
            return
        batch_started = time.perf_counter()
        if args.verbose:
            detail = f", reason={reason}" if reason else ""
            print(f"[BATCH] flush {len(batch_chunks)} chunks{detail}")
        vectors = None
        if embedding_backend is not None:
            embed_started = time.perf_counter()
            vectors = embedding_backend.encode(batch_texts)
            if args.verbose:
                print(f"[BATCH] embedding done in {time.perf_counter() - embed_started:.2f}s")
        for idx, chunk in enumerate(batch_chunks):
            vector = vectors[idx] if vectors is not None else None
            store.upsert_chunk(chunk, embedding=vector, model_name=args.embedding_model if vector is not None else None)
            total_chunks += 1
        store.commit()
        if args.verbose:
            print(f"[BATCH] commit done in {time.perf_counter() - batch_started:.2f}s")
        batch_texts = []
        batch_chunks = []

    for path in iter_source_files(data_source):
        if args.limit is not None and total_files >= args.limit:
            break
        total_files += 1
        rel_path = path.relative_to(data_source)
        file_started = time.perf_counter()
        print(f"[START] {rel_path}")
        try:
            chunks = load_file(path, data_source)
            for chunk in chunks:
                batch_chunks.append(chunk)
                batch_texts.append(build_embedding_text(chunk, args.embedding_max_chars))
                if len(batch_chunks) >= args.commit_every:
                    flush_batch(reason=str(rel_path))
            print(f"[OK] {rel_path} -> {len(chunks)} chunks, elapsed={time.perf_counter() - file_started:.2f}s")
        except Exception as exc:
            failed.append((str(path), str(exc)))
            print(f"[FAILED] {path}: {exc}")

    flush_batch(reason="final")
    elapsed = time.perf_counter() - started
    print(
        f"Index built: files={total_files}, chunks={total_chunks}, "
        f"stored_chunks={store.count_chunks()}, embeddings={store.count_embeddings(args.embedding_model)}, "
        f"failed={len(failed)}, elapsed={elapsed:.2f}s"
    )
    if failed:
        print("Failed files:")
        for file_path, error in failed[:30]:
            print(f"- {file_path}: {error}")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
