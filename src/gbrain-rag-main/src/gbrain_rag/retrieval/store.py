from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np

from gbrain_rag.core.text import stable_id
from gbrain_rag.core.text import text_tokens
from gbrain_rag.core.types import Chunk
from gbrain_rag.retrieval.embeddings import blob_to_vector, vector_to_blob
from gbrain_rag.retrieval.entities import extract_entities


class BrainStore:
    """SQLite-backed local brain.

    It keeps chunks, embeddings, deterministic entities, and typed co-mention
    links together so retrieval can combine vector, keyword, and graph signals.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.init_schema()

    def init_schema(self) -> None:
        self.conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;

            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                doc_name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_path TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                doc_name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                title TEXT,
                text TEXT NOT NULL,
                index_text TEXT NOT NULL,
                block_type TEXT NOT NULL,
                page_label TEXT,
                source_path TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                updated_at REAL NOT NULL,
                FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                dim INTEGER NOT NULL,
                vector BLOB NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY(chunk_id, model_name),
                FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id)
            );

            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                name TEXT NOT NULL,
                normalized_name TEXT NOT NULL,
                UNIQUE(kind, normalized_name)
            );

            CREATE TABLE IF NOT EXISTS chunk_entities (
                chunk_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                PRIMARY KEY(chunk_id, entity_id),
                FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id),
                FOREIGN KEY(entity_id) REFERENCES entities(entity_id)
            );

            CREATE TABLE IF NOT EXISTS entity_links (
                source_entity_id TEXT NOT NULL,
                target_entity_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                evidence_count INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY(source_entity_id, target_entity_id, relation)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                doc_id UNINDEXED,
                source_type UNINDEXED,
                title,
                text,
                index_text
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_source_type ON chunks(source_type);
            CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
            CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model_name);
            CREATE INDEX IF NOT EXISTS idx_chunk_entities_entity ON chunk_entities(entity_id);
            """
        )
        self.conn.commit()

    def reset(self) -> None:
        self.conn.executescript(
            """
            DROP TABLE IF EXISTS entity_links;
            DROP TABLE IF EXISTS chunk_entities;
            DROP TABLE IF EXISTS entities;
            DROP TABLE IF EXISTS embeddings;
            DROP TABLE IF EXISTS chunks;
            DROP TABLE IF EXISTS documents;
            DROP TABLE IF EXISTS chunks_fts;
            """
        )
        self.conn.commit()
        self.init_schema()

    def upsert_document(
        self,
        *,
        doc_id: str,
        doc_name: str,
        source_type: str,
        source_path: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO documents(doc_id, doc_name, source_type, source_path, metadata_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
              doc_name=excluded.doc_name,
              source_type=excluded.source_type,
              source_path=excluded.source_path,
              metadata_json=excluded.metadata_json,
              updated_at=excluded.updated_at
            """,
            (
                doc_id,
                doc_name,
                source_type,
                source_path,
                json.dumps(metadata or {}, ensure_ascii=False),
                time.time(),
            ),
        )

    def upsert_chunk(self, chunk: Chunk, embedding: np.ndarray | None = None, model_name: str | None = None) -> None:
        self.upsert_document(
            doc_id=chunk.doc_id,
            doc_name=chunk.doc_name,
            source_type=chunk.source_type,
            source_path=chunk.source_path,
            metadata={"doc_name": chunk.doc_name},
        )
        self.conn.execute(
            """
            INSERT INTO chunks(
              chunk_id, doc_id, doc_name, source_type, title, text, index_text,
              block_type, page_label, source_path, metadata_json, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
              doc_id=excluded.doc_id,
              doc_name=excluded.doc_name,
              source_type=excluded.source_type,
              title=excluded.title,
              text=excluded.text,
              index_text=excluded.index_text,
              block_type=excluded.block_type,
              page_label=excluded.page_label,
              source_path=excluded.source_path,
              metadata_json=excluded.metadata_json,
              updated_at=excluded.updated_at
            """,
            (
                chunk.chunk_id,
                chunk.doc_id,
                chunk.doc_name,
                chunk.source_type,
                chunk.title,
                chunk.text,
                chunk.index_text,
                chunk.block_type,
                str(chunk.page_label) if chunk.page_label is not None else None,
                chunk.source_path,
                json.dumps(chunk.metadata or {}, ensure_ascii=False),
                time.time(),
            ),
        )
        self.conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk.chunk_id,))
        self.conn.execute(
            """
            INSERT INTO chunks_fts(chunk_id, doc_id, source_type, title, text, index_text)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                chunk.chunk_id,
                chunk.doc_id,
                chunk.source_type,
                chunk.title or "",
                chunk.text,
                chunk.index_text,
            ),
        )
        if embedding is not None and model_name:
            self.upsert_embedding(chunk.chunk_id, model_name, embedding)
        self._refresh_chunk_entities(chunk)

    def upsert_embedding(self, chunk_id: str, model_name: str, embedding: np.ndarray) -> None:
        vector = np.asarray(embedding, dtype=np.float32)
        self.conn.execute(
            """
            INSERT INTO embeddings(chunk_id, model_name, dim, vector, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id, model_name) DO UPDATE SET
              dim=excluded.dim,
              vector=excluded.vector,
              updated_at=excluded.updated_at
            """,
            (chunk_id, model_name, int(vector.shape[0]), vector_to_blob(vector), time.time()),
        )

    def _refresh_chunk_entities(self, chunk: Chunk) -> None:
        self.conn.execute("DELETE FROM chunk_entities WHERE chunk_id = ?", (chunk.chunk_id,))
        entities = extract_entities(
            "\n".join(
                part
                for part in [chunk.doc_name, chunk.title or "", chunk.text, chunk.index_text]
                if part
            )
        )
        entity_ids: list[str] = []
        for kind, name in entities:
            normalized = name.lower()
            entity_id = stable_id(kind, normalized)
            self.conn.execute(
                """
                INSERT OR IGNORE INTO entities(entity_id, kind, name, normalized_name)
                VALUES (?, ?, ?, ?)
                """,
                (entity_id, kind, name, normalized),
            )
            self.conn.execute(
                "INSERT OR IGNORE INTO chunk_entities(chunk_id, entity_id) VALUES (?, ?)",
                (chunk.chunk_id, entity_id),
            )
            entity_ids.append(entity_id)

        for idx, source_id in enumerate(entity_ids):
            for target_id in entity_ids[idx + 1 :]:
                if source_id == target_id:
                    continue
                a, b = sorted([source_id, target_id])
                self.conn.execute(
                    """
                    INSERT INTO entity_links(source_entity_id, target_entity_id, relation, evidence_count)
                    VALUES (?, ?, 'co_mentions', 1)
                    ON CONFLICT(source_entity_id, target_entity_id, relation)
                    DO UPDATE SET evidence_count = evidence_count + 1
                    """,
                    (a, b),
                )

    def commit(self) -> None:
        self.conn.commit()

    def count_chunks(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def count_embeddings(self, model_name: str | None = None) -> int:
        if model_name:
            return int(
                self.conn.execute(
                    "SELECT COUNT(*) FROM embeddings WHERE model_name = ?", (model_name,)
                ).fetchone()[0]
            )
        return int(self.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])

    def source_counts(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT source_type, COUNT(*) AS count FROM chunks GROUP BY source_type"
        ).fetchall()
        return {row["source_type"]: int(row["count"]) for row in rows}

    def embedding_counts(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT model_name, COUNT(*) AS count FROM embeddings GROUP BY model_name"
        ).fetchall()
        return {row["model_name"]: int(row["count"]) for row in rows}

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        row = self.conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        return self._row_to_chunk(row) if row else None

    def load_chunks(self, source_types: list[str] | None = None) -> dict[str, Chunk]:
        params: list[Any] = []
        clause = ""
        if source_types:
            placeholders = ",".join("?" for _ in source_types)
            clause = f"WHERE source_type IN ({placeholders})"
            params.extend(source_types)
        rows = self.conn.execute(f"SELECT * FROM chunks {clause}", params).fetchall()
        return {row["chunk_id"]: self._row_to_chunk(row) for row in rows}

    def load_chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = self.conn.execute(
            f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        return {row["chunk_id"]: self._row_to_chunk(row) for row in rows}

    def load_chunks_by_doc_ids(self, doc_ids: list[str]) -> dict[str, list[Chunk]]:
        if not doc_ids:
            return {}
        unique_doc_ids = list(dict.fromkeys(doc_ids))
        placeholders = ",".join("?" for _ in unique_doc_ids)
        rows = self.conn.execute(
            f"""
            SELECT *
            FROM chunks
            WHERE doc_id IN ({placeholders})
            ORDER BY doc_id, chunk_id
            """,
            unique_doc_ids,
        ).fetchall()
        chunks_by_doc_id: dict[str, list[Chunk]] = {doc_id: [] for doc_id in unique_doc_ids}
        for row in rows:
            chunk = self._row_to_chunk(row)
            chunks_by_doc_id.setdefault(chunk.doc_id, []).append(chunk)
        for chunks in chunks_by_doc_id.values():
            chunks.sort(key=self._chunk_document_order_key)
        return chunks_by_doc_id

    def _chunk_document_order_key(self, chunk: Chunk) -> tuple[int, str, int, int, str]:
        page_label = chunk.page_label
        try:
            page_number = int(str(page_label))
        except (TypeError, ValueError):
            page_number = 0
        block_priority = 0 if chunk.block_type == "text" else 1
        metadata = chunk.metadata or {}
        try:
            part = int(metadata.get("part", 0) or 0)
        except (TypeError, ValueError):
            part = 0
        try:
            table_index = int(metadata.get("table_index", 0) or 0)
        except (TypeError, ValueError):
            table_index = 0
        return (page_number, str(page_label or ""), block_priority, part or table_index, chunk.chunk_id)

    def load_metadata_rows(self, source_type: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT chunk_id, doc_name, source_path, metadata_json
            FROM chunks
            WHERE source_type = ?
            ORDER BY chunk_id
            """,
            (source_type,),
        ).fetchall()
        filtered_rows = self._preferred_structured_rows(source_type, rows)
        result: list[dict[str, Any]] = []
        for row in filtered_rows:
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except Exception:
                metadata = {}
            metadata.setdefault("chunk_id", row["chunk_id"])
            result.append(metadata)
        return result

    def _preferred_structured_rows(self, source_type: str, rows: list[sqlite3.Row]) -> list[sqlite3.Row]:
        if source_type == "table":
            preferred = [
                row
                for row in rows
                if str(row["doc_name"] or "").endswith("tables/model_release_records.jsonl")
                or str(row["source_path"] or "").endswith("tables/model_release_records.jsonl")
            ]
            return preferred or rows
        if source_type == "adela":
            preferred = [
                row
                for row in rows
                if str(row["doc_name"] or "").endswith("adela/adela_release_records.jsonl")
                or str(row["source_path"] or "").endswith("adela/adela_release_records.jsonl")
            ]
            return preferred or rows
        return rows

    def metadata_source_profile(
        self,
        source_type: str,
        fields: list[str],
        *,
        max_values_per_field: int = 20,
    ) -> str:
        record_count = int(
            self.conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE source_type = ?",
                (source_type,),
            ).fetchone()[0]
        )
        lines = [f"{source_type}: total_records={record_count}"]
        for field in fields:
            path = f"$.{field}"
            rows = self.conn.execute(
                """
                SELECT json_extract(metadata_json, ?) AS value, COUNT(*) AS count
                FROM chunks
                WHERE source_type = ?
                  AND json_extract(metadata_json, ?) IS NOT NULL
                  AND json_extract(metadata_json, ?) != ''
                GROUP BY value
                ORDER BY count DESC, value ASC
                LIMIT ?
                """,
                (path, source_type, path, path, max_values_per_field),
            ).fetchall()
            if not rows:
                continue
            values = "; ".join(f"{row['value']}({row['count']})" for row in rows)
            lines.append(f"- {field}: {values}")
        return "\n".join(lines)

    def load_embedding_matrix(
        self,
        *,
        model_name: str,
        source_types: list[str] | None = None,
    ) -> tuple[list[str], np.ndarray]:
        params: list[Any] = [model_name]
        clause = ""
        if source_types:
            placeholders = ",".join("?" for _ in source_types)
            clause = f"AND c.source_type IN ({placeholders})"
            params.extend(source_types)
        rows = self.conn.execute(
            f"""
            SELECT e.chunk_id, e.vector
            FROM embeddings e
            JOIN chunks c ON c.chunk_id = e.chunk_id
            WHERE e.model_name = ? {clause}
            """,
            params,
        ).fetchall()
        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32)
        ids = [row["chunk_id"] for row in rows]
        matrix = np.vstack([blob_to_vector(row["vector"]) for row in rows]).astype(np.float32)
        return ids, matrix

    def fts_search(
        self,
        query: str,
        *,
        source_types: list[str] | None = None,
        limit: int = 80,
    ) -> list[tuple[str, float]]:
        terms = [term.replace('"', "") for term in text_tokens(query) if len(term.strip()) >= 2]
        if not terms:
            terms = [term.strip().replace('"', "") for term in query.split() if term.strip()]
        match_query = " OR ".join(f'"{term}"' for term in terms if term)
        if not match_query:
            return []
        params: list[Any] = [match_query]
        clause = ""
        if source_types:
            placeholders = ",".join("?" for _ in source_types)
            clause = f"AND source_type IN ({placeholders})"
            params.extend(source_types)
        params.append(limit)
        try:
            rows = self.conn.execute(
                f"""
                SELECT chunk_id, bm25(chunks_fts) AS score
                FROM chunks_fts
                WHERE chunks_fts MATCH ? {clause}
                ORDER BY score
                LIMIT ?
                """,
                params,
            ).fetchall()
        except sqlite3.Error:
            return []
        # SQLite FTS BM25 is lower-is-better and often negative; normalize later
        return [(row["chunk_id"], float(-row["score"])) for row in rows]

    def chunk_entities(self, chunk_ids: list[str]) -> dict[str, list[str]]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = self.conn.execute(
            f"""
            SELECT ce.chunk_id, e.name
            FROM chunk_entities ce
            JOIN entities e ON e.entity_id = ce.entity_id
            WHERE ce.chunk_id IN ({placeholders})
            """,
            chunk_ids,
        ).fetchall()
        result: dict[str, list[str]] = {chunk_id: [] for chunk_id in chunk_ids}
        for row in rows:
            result.setdefault(row["chunk_id"], []).append(row["name"])
        return result

    def _row_to_chunk(self, row: sqlite3.Row) -> Chunk:
        metadata_raw = row["metadata_json"] or "{}"
        try:
            metadata = json.loads(metadata_raw)
        except Exception:
            metadata = {}
        return Chunk(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            doc_name=row["doc_name"],
            source_type=row["source_type"],
            title=row["title"],
            text=row["text"],
            index_text=row["index_text"],
            block_type=row["block_type"],
            page_label=row["page_label"],
            source_path=row["source_path"],
            metadata=metadata,
        )
