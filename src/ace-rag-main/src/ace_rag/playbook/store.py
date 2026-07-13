from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ace_rag.api.schemas import (
    FeedbackRequest,
    PlaybookHit,
    PlaybookItem,
    PlaybookOrganizeCandidate,
)
from ace_rag.core.text import dedupe_keep_order, infer_query_intents, normalize_query, tokenize


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_loads(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def _lexical_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in tokenize(text):
        tokens.append(token)
        if re.fullmatch(r"[\u4e00-\u9fff]{3,}", token):
            for width in (2, 3):
                tokens.extend(token[index : index + width] for index in range(0, len(token) - width + 1))
    return dedupe_keep_order(tokens)


AUTO_ORGANIZE_STATE_KEY = "playbook_auto_organize"


@dataclass(frozen=True)
class FeedbackInsertResult:
    feedback_id: str
    operation_id: str | None
    online_item_id: str | None = None


class PlaybookStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS playbook_items (
                  item_id TEXT PRIMARY KEY,
                  section TEXT NOT NULL,
                  content TEXT NOT NULL,
                  status TEXT NOT NULL DEFAULT 'active',
                  tags_json TEXT NOT NULL DEFAULT '[]',
                  source_hints_json TEXT NOT NULL DEFAULT '[]',
                  query_intents_json TEXT NOT NULL DEFAULT '[]',
                  expansion_terms_json TEXT NOT NULL DEFAULT '[]',
                  helpful_count INTEGER NOT NULL DEFAULT 0,
                  harmful_count INTEGER NOT NULL DEFAULT 0,
                  confidence REAL NOT NULL DEFAULT 0.5,
                  provenance_json TEXT NOT NULL DEFAULT '{}',
                  created_at REAL NOT NULL,
                  updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS qa_runs (
                  run_id TEXT PRIMARY KEY,
                  query TEXT NOT NULL,
                  request_json TEXT NOT NULL DEFAULT '{}',
                  v2_request_json TEXT NOT NULL DEFAULT '{}',
                  v2_response_json TEXT NOT NULL DEFAULT '{}',
                  playbook_item_ids_json TEXT NOT NULL DEFAULT '[]',
                  answer TEXT NOT NULL,
                  timings_json TEXT NOT NULL DEFAULT '{}',
                  created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS qa_feedback (
                  feedback_id TEXT PRIMARY KEY,
                  run_id TEXT NOT NULL,
                  feedback_type TEXT NOT NULL,
                  rating INTEGER,
                  corrected_answer TEXT,
                  expected_evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                  comment TEXT,
                  status TEXT NOT NULL DEFAULT 'pending',
                  created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS playbook_operations (
                  op_id TEXT PRIMARY KEY,
                  feedback_id TEXT,
                  operation_type TEXT NOT NULL,
                  target_item_id TEXT,
                  payload_json TEXT NOT NULL DEFAULT '{}',
                  status TEXT NOT NULL DEFAULT 'pending',
                  created_at REAL NOT NULL,
                  applied_at REAL
                );

                CREATE TABLE IF NOT EXISTS playbook_state (
                  key TEXT PRIMARY KEY,
                  value_json TEXT NOT NULL DEFAULT '{}',
                  updated_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_playbook_items_status
                  ON playbook_items(status);
                CREATE INDEX IF NOT EXISTS idx_qa_runs_created
                  ON qa_runs(created_at);
                CREATE INDEX IF NOT EXISTS idx_qa_feedback_run
                  ON qa_feedback(run_id);
                """
            )

    def upsert_item(self, item: PlaybookItem) -> None:
        now = time.time()
        created_at = item.created_at or now
        updated_at = item.updated_at or now
        with self.connect() as conn:
            self._upsert_item_with_conn(conn, item, created_at=created_at, updated_at=updated_at)

    def _upsert_item_with_conn(
        self,
        conn: sqlite3.Connection,
        item: PlaybookItem,
        *,
        created_at: float | None = None,
        updated_at: float | None = None,
    ) -> None:
        now = time.time()
        created_at = item.created_at if created_at is None else created_at
        updated_at = item.updated_at if updated_at is None else updated_at
        created_at = created_at or now
        updated_at = updated_at or now
        existing = conn.execute(
            "SELECT created_at FROM playbook_items WHERE item_id = ?",
            (item.item_id,),
        ).fetchone()
        if existing:
            created_at = float(existing["created_at"])
        conn.execute(
            """
            INSERT INTO playbook_items (
              item_id, section, content, status, tags_json, source_hints_json,
              query_intents_json, expansion_terms_json, helpful_count,
              harmful_count, confidence, provenance_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(item_id) DO UPDATE SET
              section=excluded.section,
              content=excluded.content,
              status=excluded.status,
              tags_json=excluded.tags_json,
              source_hints_json=excluded.source_hints_json,
              query_intents_json=excluded.query_intents_json,
              expansion_terms_json=excluded.expansion_terms_json,
              helpful_count=excluded.helpful_count,
              harmful_count=excluded.harmful_count,
              confidence=excluded.confidence,
              provenance_json=excluded.provenance_json,
              updated_at=excluded.updated_at
            """,
            (
                item.item_id,
                item.section,
                item.content,
                item.status,
                _json_dumps(item.tags),
                _json_dumps(item.source_hints),
                _json_dumps(item.query_intents),
                _json_dumps(item.expansion_terms),
                item.helpful_count,
                item.harmful_count,
                item.confidence,
                _json_dumps(item.provenance),
                created_at,
                updated_at,
            ),
        )

    def import_items(self, items: Iterable[PlaybookItem]) -> int:
        count = 0
        for item in items:
            self.upsert_item(item)
            count += 1
        return count

    def get_item_count(self) -> int:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS count FROM playbook_items WHERE status = 'active'"
            ).fetchone()
            return int(row["count"] if row else 0)

    def get_state(self, key: str, default: Any = None) -> Any:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT value_json FROM playbook_state WHERE key = ?",
                (key,),
            ).fetchone()
        if not row:
            return default
        return _json_loads(row["value_json"], default)

    def set_state(self, key: str, value: Any) -> None:
        with self.connect() as conn:
            self._set_state_with_conn(conn, key, value)

    def _set_state_with_conn(self, conn: sqlite3.Connection, key: str, value: Any) -> None:
        conn.execute(
            """
            INSERT INTO playbook_state (key, value_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
              value_json=excluded.value_json,
              updated_at=excluded.updated_at
            """,
            (key, _json_dumps(value), time.time()),
        )

    def get_auto_organize_state(self) -> dict[str, Any]:
        state = self.get_state(AUTO_ORGANIZE_STATE_KEY, {})
        return state if isinstance(state, dict) else {}

    def get_auto_organize_baseline_count(self) -> int | None:
        value = self.get_auto_organize_state().get("last_organized_active_item_count")
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def ensure_auto_organize_baseline(self) -> int:
        existing = self.get_auto_organize_baseline_count()
        if existing is not None:
            return existing
        current_count = self.get_item_count()
        self.set_state(
            AUTO_ORGANIZE_STATE_KEY,
            {
                "last_organized_active_item_count": current_count,
                "last_organized_at": None,
                "last_auto_organize_operation_id": None,
                "reason": "baseline_initialized",
            },
        )
        return current_count

    def list_items(self, include_inactive: bool = False) -> list[PlaybookItem]:
        sql = "SELECT * FROM playbook_items"
        if not include_inactive:
            sql += " WHERE status = 'active'"
        sql += " ORDER BY section, item_id"
        with self.connect() as conn:
            rows = conn.execute(sql).fetchall()
        return [self._row_to_item(row) for row in rows]

    def search(self, query: str, top_k: int, intents: list[str] | None = None) -> list[PlaybookHit]:
        if top_k <= 0:
            return []
        normalized_query = normalize_query(query)
        raw_query_tokens = _lexical_tokens(query)
        query_tokens = set(raw_query_tokens)
        query_intents = set(intents or infer_query_intents(query))
        rows = self.list_items(include_inactive=False)
        indexed_rows: list[tuple[PlaybookItem, list[str], Counter[str]]] = []
        document_frequency: Counter[str] = Counter()
        total_length = 0
        for item in rows:
            weighted_tokens = self._search_tokens(item)
            token_counts = Counter(weighted_tokens)
            indexed_rows.append((item, weighted_tokens, token_counts))
            document_frequency.update(token_counts.keys())
            total_length += len(weighted_tokens)

        avg_doc_length = total_length / max(len(indexed_rows), 1)
        hits: list[PlaybookHit] = []
        for item, content_token_list, content_token_counts in indexed_rows:
            content_tokens = set(content_token_list)
            overlap = len(query_tokens & content_tokens)
            intent_overlap = len(query_intents & set(item.query_intents))
            section_bonus = 0.0
            feedback_bonus = 0.0
            exact_query_match = 0.0
            if item.section == "online_feedback":
                source_query = str(item.provenance.get("source_query") or "")
                normalized_source_query = normalize_query(source_query)
                if normalized_source_query and normalized_query != normalized_source_query:
                    continue
                if normalized_query and normalized_query == normalized_source_query:
                    exact_query_match = 1.0
                    feedback_bonus += 0.9
                if str(item.provenance.get("feedback_type") or "") == "correction":
                    feedback_bonus += 0.08
            bm25_score = self._bm25_score(
                raw_query_tokens,
                content_token_counts,
                document_frequency,
                document_count=len(indexed_rows),
                document_length=len(content_token_list),
                avg_doc_length=avg_doc_length,
            )
            keyword_score = self._keyword_score(raw_query_tokens, item)
            if "deployment" in query_intents and "source_routing" == item.section:
                section_bonus += 0.15
            if "field_lookup" in query_intents and item.section in {"field_binding", "answer_strategy"}:
                section_bonus += 0.12
            if "aggregate" in query_intents and item.section == "aggregate_semantics":
                section_bonus += 0.16
            if not overlap and not intent_overlap and not section_bonus and not feedback_bonus and not keyword_score:
                continue
            token_score = min(overlap / max(len(query_tokens), 1), 1.0)
            intent_score = min(intent_overlap / max(len(query_intents), 1), 1.0) if query_intents else 0.0
            score = (
                (0.42 * bm25_score)
                + (0.23 * keyword_score)
                + (0.15 * token_score)
                + (0.3 * intent_score)
                + section_bonus
                + feedback_bonus
                + (0.15 * item.confidence)
            )
            hits.append(
                PlaybookHit(
                    **item.model_dump(),
                    score=round(score, 6),
                    score_details={
                        "bm25": round(bm25_score, 6),
                        "keyword": round(keyword_score, 6),
                        "token": round(token_score, 6),
                        "intent": round(intent_score, 6),
                        "section_bonus": round(section_bonus, 6),
                        "feedback_bonus": round(feedback_bonus, 6),
                        "exact_query_match": round(exact_query_match, 6),
                        "confidence": round(item.confidence, 6),
                    },
                )
            )
        hits.sort(key=lambda hit: (hit.score, hit.confidence, hit.updated_at or 0), reverse=True)
        return hits[:top_k]

    def organize_candidates(
        self,
        *,
        include_sections: list[str] | None = None,
        include_inactive: bool = False,
        min_confidence: float = 0.0,
        max_items: int = 200,
    ) -> tuple[int, list[PlaybookOrganizeCandidate]]:
        items = [
            item
            for item in self.list_items(include_inactive=include_inactive)
            if item.confidence >= min_confidence
            and (not include_sections or item.section in set(include_sections))
        ][:max_items]

        groups: dict[str, list[PlaybookItem]] = defaultdict(list)
        for item in items:
            group_key = self._organize_group_key(item)
            groups[group_key].append(item)

        candidates: list[PlaybookOrganizeCandidate] = []
        for group_key, group_items in groups.items():
            if not group_items:
                continue
            group_items.sort(key=lambda item: (item.confidence, item.updated_at or 0), reverse=True)
            item_ids = [item.item_id for item in group_items]
            sections = dedupe_keep_order(item.section for item in group_items)
            tags = dedupe_keep_order(tag for item in group_items for tag in item.tags)[:20]
            source_hints = dedupe_keep_order(source for item in group_items for source in item.source_hints)
            query_intents = dedupe_keep_order(intent for item in group_items for intent in item.query_intents)
            expansion_terms = dedupe_keep_order(term for item in group_items for term in item.expansion_terms)[:24]
            confidence = sum(item.confidence for item in group_items) / max(len(group_items), 1)
            summary = self._summarize_group(group_items)
            candidates.append(
                PlaybookOrganizeCandidate(
                    candidate_id=f"org-{hashlib.sha1('|'.join(item_ids).encode('utf-8')).hexdigest()[:16]}",
                    title=self._organize_title(group_key, group_items),
                    summary=summary,
                    strategy=self._organize_strategy(group_items),
                    item_ids=item_ids,
                    sections=sections,
                    tags=tags,
                    source_hints=source_hints,
                    query_intents=query_intents,
                    expansion_terms=expansion_terms,
                    confidence=round(confidence, 6),
                    rationale=self._organize_rationale(group_items),
                )
            )
        candidates.sort(key=lambda candidate: (len(candidate.item_ids), candidate.confidence), reverse=True)
        return len(items), candidates

    def insert_run(
        self,
        *,
        query: str,
        request: dict[str, Any],
        v2_request: dict[str, Any],
        v2_response: dict[str, Any],
        playbook_item_ids: list[str],
        answer: str,
        timings: dict[str, Any],
    ) -> str:
        run_id = f"run-{uuid.uuid4().hex[:16]}"
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO qa_runs (
                  run_id, query, request_json, v2_request_json, v2_response_json,
                  playbook_item_ids_json, answer, timings_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    query,
                    _json_dumps(request),
                    _json_dumps(v2_request),
                    _json_dumps(v2_response),
                    _json_dumps(playbook_item_ids),
                    answer,
                    _json_dumps(timings),
                    time.time(),
                ),
            )
        return run_id

    def insert_feedback(self, feedback: FeedbackRequest) -> tuple[str, str | None]:
        result = self.insert_feedback_detailed(feedback)
        return result.feedback_id, result.operation_id

    def insert_feedback_detailed(self, feedback: FeedbackRequest) -> FeedbackInsertResult:
        feedback_id = f"fb-{uuid.uuid4().hex[:16]}"
        online_item_id: str | None = None
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO qa_feedback (
                  feedback_id, run_id, feedback_type, rating, corrected_answer,
                  expected_evidence_ids_json, comment, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (
                    feedback_id,
                    feedback.run_id,
                    feedback.feedback_type,
                    feedback.rating,
                    feedback.corrected_answer,
                    _json_dumps(feedback.expected_evidence_ids),
                    feedback.comment,
                    time.time(),
                ),
            )
            run_payload = self._load_run(conn, feedback.run_id)
            if run_payload and (feedback.corrected_answer or feedback.comment or feedback.expected_evidence_ids):
                online_item_id = self._upsert_online_feedback_item(
                    conn=conn,
                    feedback_id=feedback_id,
                    feedback=feedback,
                    run_payload=run_payload,
                )
            op_id: str | None = None
            if feedback.corrected_answer or feedback.comment or feedback.feedback_type in {"harmful", "correction", "missing_evidence"}:
                op_id = f"op-{uuid.uuid4().hex[:16]}"
                conn.execute(
                    """
                    INSERT INTO playbook_operations (
                      op_id, feedback_id, operation_type, target_item_id,
                      payload_json, status, created_at, applied_at
                    ) VALUES (?, ?, 'REVIEW_FEEDBACK', NULL, ?, 'pending', ?, NULL)
                    """,
                    (
                        op_id,
                        feedback_id,
                        _json_dumps(
                            {
                                "run_id": feedback.run_id,
                                "feedback_type": feedback.feedback_type,
                                "rating": feedback.rating,
                                "corrected_answer": feedback.corrected_answer,
                                "expected_evidence_ids": feedback.expected_evidence_ids,
                                "comment": feedback.comment,
                            }
                        ),
                        time.time(),
                    ),
                )
        return FeedbackInsertResult(
            feedback_id=feedback_id,
            operation_id=op_id,
            online_item_id=online_item_id,
        )

    def apply_auto_organization(
        self,
        *,
        feedback_id: str | None,
        new_items: list[PlaybookItem],
        retire_item_ids: list[str],
        payload: dict[str, Any],
    ) -> str:
        op_id = f"op-{uuid.uuid4().hex[:16]}"
        now = time.time()
        new_item_ids = {item.item_id for item in new_items}
        retire_ids = dedupe_keep_order(
            item_id for item_id in retire_item_ids if item_id and item_id not in new_item_ids
        )
        with self.connect() as conn:
            before_row = conn.execute(
                "SELECT COUNT(*) AS count FROM playbook_items WHERE status = 'active'"
            ).fetchone()
            before_count = int(before_row["count"] if before_row else 0)

            for item in new_items:
                self._upsert_item_with_conn(conn, item, updated_at=now)

            if retire_ids:
                placeholders = ",".join("?" for _ in retire_ids)
                conn.execute(
                    f"""
                    UPDATE playbook_items
                    SET status = 'inactive', updated_at = ?
                    WHERE status = 'active' AND item_id IN ({placeholders})
                    """,
                    (now, *retire_ids),
                )

            after_row = conn.execute(
                "SELECT COUNT(*) AS count FROM playbook_items WHERE status = 'active'"
            ).fetchone()
            after_count = int(after_row["count"] if after_row else 0)
            operation_payload = {
                **payload,
                "active_item_count_before_apply": before_count,
                "active_item_count_after_apply": after_count,
                "applied_item_ids": [item.item_id for item in new_items],
                "retired_item_ids": retire_ids,
            }
            conn.execute(
                """
                INSERT INTO playbook_operations (
                  op_id, feedback_id, operation_type, target_item_id,
                  payload_json, status, created_at, applied_at
                ) VALUES (?, ?, 'AUTO_ORGANIZE_PLAYBOOK', NULL, ?, 'applied', ?, ?)
                """,
                (op_id, feedback_id, _json_dumps(operation_payload), now, now),
            )
            self._set_state_with_conn(
                conn,
                AUTO_ORGANIZE_STATE_KEY,
                {
                    "last_organized_active_item_count": after_count,
                    "last_organized_at": now,
                    "last_auto_organize_operation_id": op_id,
                    "last_auto_organize_status": "applied",
                },
            )
        return op_id

    def record_auto_organization_attempt(
        self,
        *,
        feedback_id: str | None,
        status: str,
        payload: dict[str, Any],
        update_baseline: bool = False,
    ) -> str:
        op_id = f"op-{uuid.uuid4().hex[:16]}"
        now = time.time()
        with self.connect() as conn:
            current_row = conn.execute(
                "SELECT COUNT(*) AS count FROM playbook_items WHERE status = 'active'"
            ).fetchone()
            current_count = int(current_row["count"] if current_row else 0)
            conn.execute(
                """
                INSERT INTO playbook_operations (
                  op_id, feedback_id, operation_type, target_item_id,
                  payload_json, status, created_at, applied_at
                ) VALUES (?, ?, 'AUTO_ORGANIZE_PLAYBOOK', NULL, ?, ?, ?, ?)
                """,
                (
                    op_id,
                    feedback_id,
                    _json_dumps({**payload, "active_item_count": current_count}),
                    status,
                    now,
                    now if status in {"applied", "skipped", "noop"} else None,
                ),
            )
            if update_baseline:
                self._set_state_with_conn(
                    conn,
                    AUTO_ORGANIZE_STATE_KEY,
                    {
                        "last_organized_active_item_count": current_count,
                        "last_organized_at": now,
                        "last_auto_organize_operation_id": op_id,
                        "last_auto_organize_status": status,
                    },
                )
        return op_id

    def _load_run(self, conn: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
        row = conn.execute(
            """
            SELECT run_id, query, request_json, v2_request_json, v2_response_json,
                   playbook_item_ids_json, answer, timings_json, created_at
            FROM qa_runs
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "run_id": row["run_id"],
            "query": row["query"],
            "request": _json_loads(row["request_json"], {}),
            "v2_request": _json_loads(row["v2_request_json"], {}),
            "v2_response": _json_loads(row["v2_response_json"], {}),
            "playbook_item_ids": _json_loads(row["playbook_item_ids_json"], []),
            "answer": row["answer"],
            "timings": _json_loads(row["timings_json"], {}),
            "created_at": float(row["created_at"]),
        }

    def _upsert_online_feedback_item(
        self,
        *,
        conn: sqlite3.Connection,
        feedback_id: str,
        feedback: FeedbackRequest,
        run_payload: dict[str, Any],
    ) -> str | None:
        source_query = str(run_payload.get("query") or "")
        normalized_source_query = normalize_query(source_query)
        if not normalized_source_query:
            return None

        item_id = f"online-feedback-{hashlib.sha1(normalized_source_query.encode('utf-8')).hexdigest()[:16]}"
        expected_evidence_ids = list(feedback.expected_evidence_ids or [])
        source_hints = self._feedback_source_hints(expected_evidence_ids, run_payload)
        query_intents = infer_query_intents(source_query)
        expansion_terms = self._feedback_expansion_terms(
            source_query=source_query,
            corrected_answer=feedback.corrected_answer,
            comment=feedback.comment,
            expected_evidence_ids=expected_evidence_ids,
        )
        content_parts = [
            f"用户针对问题“{source_query}”提交了在线反馈，后续遇到相同或高度相似的问题应优先纠正。",
        ]
        if feedback.corrected_answer:
            content_parts.append(f"修正答案：{feedback.corrected_answer}")
        if expected_evidence_ids:
            content_parts.append(f"优先核对证据：{', '.join(expected_evidence_ids)}")
        if feedback.comment:
            content_parts.append(f"备注：{feedback.comment}")

        item = PlaybookItem(
            item_id=item_id,
            section="online_feedback",
            content="\n".join(content_parts),
            status="active",
            tags=dedupe_keep_order(["online-feedback", feedback.feedback_type, *tokenize(source_query)[:8]]),
            source_hints=source_hints,
            query_intents=query_intents,
            expansion_terms=expansion_terms,
            confidence=1.0 if feedback.feedback_type == "correction" else 0.92,
            provenance={
                "source": "qa_feedback_online",
                "feedback_id": feedback_id,
                "run_id": feedback.run_id,
                "feedback_type": feedback.feedback_type,
                "source_query": source_query,
                "corrected_answer": feedback.corrected_answer,
                "comment": feedback.comment,
                "expected_evidence_ids": expected_evidence_ids,
                "created_from_run_at": run_payload.get("created_at"),
            },
        )
        self._upsert_item_with_conn(conn, item)
        return item_id

    def _feedback_source_hints(
        self,
        expected_evidence_ids: list[str],
        run_payload: dict[str, Any],
    ) -> list[str]:
        hints: list[str] = []
        for evidence_id in expected_evidence_ids:
            source_type = _source_type_from_evidence_id(evidence_id)
            if source_type:
                hints.append(source_type)
        if not hints:
            v2_sources = run_payload.get("v2_request", {}).get("sources") or []
            hints.extend(str(source) for source in v2_sources if str(source) in {"document", "table", "adela"})
        return dedupe_keep_order(hints)

    def _feedback_expansion_terms(
        self,
        *,
        source_query: str,
        corrected_answer: str | None,
        comment: str | None,
        expected_evidence_ids: list[str],
    ) -> list[str]:
        prioritized: list[str] = []
        tail_terms: list[str] = []
        for text in (corrected_answer or "", comment or ""):
            prioritized.extend(re.findall(r"[A-Za-z][A-Za-z0-9_./+-]{4,120}", text))
        for text in (source_query, corrected_answer or "", comment or ""):
            prioritized.extend(tokenize(text))
        for evidence_id in expected_evidence_ids:
            parts = [part.strip() for part in str(evidence_id).split("::") if part.strip()]
            if parts:
                tail_terms.append(parts[-1])
        return dedupe_keep_order([*prioritized, *tail_terms])[:16]

    def _search_tokens(self, item: PlaybookItem) -> list[str]:
        weighted_texts = [
            item.section,
            item.content,
            " ".join(item.tags),
            " ".join(item.query_intents),
            " ".join(item.expansion_terms),
        ]
        tokens: list[str] = []
        for text in weighted_texts:
            tokens.extend(_lexical_tokens(text))
        tokens.extend(_lexical_tokens(" ".join(item.tags)) * 2)
        tokens.extend(_lexical_tokens(" ".join(item.expansion_terms)) * 3)
        tokens.extend(_lexical_tokens(" ".join(item.query_intents)) * 2)
        source_query = str(item.provenance.get("source_query") or "")
        if source_query:
            tokens.extend(_lexical_tokens(source_query) * 3)
        return tokens

    def _bm25_score(
        self,
        query_tokens: list[str],
        document_tokens: Counter[str],
        document_frequency: Counter[str],
        *,
        document_count: int,
        document_length: int,
        avg_doc_length: float,
    ) -> float:
        if not query_tokens or not document_tokens:
            return 0.0
        k1 = 1.5
        b = 0.75
        score = 0.0
        for token in query_tokens:
            term_frequency = document_tokens.get(token, 0)
            if term_frequency <= 0:
                continue
            df = document_frequency.get(token, 0)
            idf = math.log(1 + (document_count - df + 0.5) / (df + 0.5))
            denominator = term_frequency + k1 * (1 - b + b * (document_length / max(avg_doc_length, 1.0)))
            score += idf * ((term_frequency * (k1 + 1)) / denominator)
        return min(score / max(len(set(query_tokens)), 1), 1.0)

    def _keyword_score(self, query_tokens: list[str], item: PlaybookItem) -> float:
        if not query_tokens:
            return 0.0
        exact_terms = {
            *[term.lower() for term in item.expansion_terms],
            *[tag.lower() for tag in item.tags],
            *[intent.lower() for intent in item.query_intents],
        }
        content = normalize_query(
            " ".join(
                [
                    item.section,
                    item.content,
                    " ".join(item.tags),
                    " ".join(item.expansion_terms),
                    str(item.provenance.get("source_query") or ""),
                ]
            )
        )
        matches = 0.0
        for token in query_tokens:
            if token in exact_terms:
                matches += 1.4
            elif token in content:
                matches += 1.0
        return min(matches / max(len(set(query_tokens)), 1), 1.0)

    def _organize_group_key(self, item: PlaybookItem) -> str:
        if item.section == "online_feedback":
            intents = "-".join(item.query_intents or ["feedback"])
            hints = "-".join(item.source_hints or ["general"])
            return f"online_feedback:{intents}:{hints}"
        if item.query_intents:
            return f"intent:{'-'.join(item.query_intents[:2])}"
        if item.source_hints:
            return f"source:{'-'.join(item.source_hints)}"
        return f"section:{item.section}"

    def _organize_title(self, group_key: str, items: list[PlaybookItem]) -> str:
        if group_key.startswith("online_feedback:"):
            return "在线反馈记忆合并候选"
        if group_key.startswith("intent:"):
            return f"按意图归纳: {', '.join(dedupe_keep_order(intent for item in items for intent in item.query_intents)[:3])}"
        if group_key.startswith("source:"):
            return f"按数据源路由归纳: {', '.join(dedupe_keep_order(source for item in items for source in item.source_hints))}"
        return f"按章节归纳: {items[0].section}"

    def _organize_strategy(self, items: list[PlaybookItem]) -> str:
        if any(item.section == "online_feedback" for item in items):
            return "episodic_feedback_to_semantic_rule"
        if len(items) > 1:
            return "semantic_merge_and_deduplicate"
        return "single_item_summary"

    def _summarize_group(self, items: list[PlaybookItem]) -> str:
        lead = items[0].content.strip().replace("\n", " ")
        if len(lead) > 140:
            lead = f"{lead[:137]}..."
        if len(items) == 1:
            return lead
        return f"合并 {len(items)} 条规则：{lead}"

    def _organize_rationale(self, items: list[PlaybookItem]) -> str:
        sections = dedupe_keep_order(item.section for item in items)
        intents = dedupe_keep_order(intent for item in items for intent in item.query_intents)
        hints = dedupe_keep_order(source for item in items for source in item.source_hints)
        parts = [f"sections={sections}"]
        if intents:
            parts.append(f"intents={intents}")
        if hints:
            parts.append(f"source_hints={hints}")
        return "; ".join(parts)

    def _row_to_item(self, row: sqlite3.Row) -> PlaybookItem:
        return PlaybookItem(
            item_id=row["item_id"],
            section=row["section"],
            content=row["content"],
            status=row["status"],
            tags=_json_loads(row["tags_json"], []),
            source_hints=_json_loads(row["source_hints_json"], []),
            query_intents=_json_loads(row["query_intents_json"], []),
            expansion_terms=_json_loads(row["expansion_terms_json"], []),
            helpful_count=int(row["helpful_count"]),
            harmful_count=int(row["harmful_count"]),
            confidence=float(row["confidence"]),
            provenance=_json_loads(row["provenance_json"], {}),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )


def _has_query_specific_signal(item: PlaybookHit) -> bool:
    details = item.score_details or {}
    return any(
        float(details.get(key) or 0.0) > 0.0
        for key in ("bm25", "keyword", "token", "feedback_bonus", "exact_query_match")
    )


GENERIC_EXPANSION_MATCH_TERMS = {
    "模型",
    "算法",
    "检测",
    "识别",
    "检索",
    "问题",
    "部署",
    "平台",
    "需要",
    "相关",
    "通常",
    "model",
    "deployment",
    "field_lookup",
    "document_detail",
    "alias",
}


EXPANSION_TRIGGER_ALIASES: dict[str, tuple[str, ...]] = {
    "safety_rope": ("安全绳", "safety rope", "safety_rope"),
    "safetybelt": ("安全带", "safetybelt"),
    "safety_belt": ("安全带", "safety belt", "safety_belt"),
    "waistcoat": ("反光衣", "红马甲", "waistcoat"),
    "par_waistcoat_safetybelt": ("反光衣", "红马甲", "安全带", "par_waistcoat_safetybelt"),
}


def _query_expansion_item_applies(item: PlaybookHit, query: str | None) -> bool:
    if not query:
        return _has_query_specific_signal(item)
    normalized_query = normalize_query(query)
    compact_query = re.sub(r"\s+", "", normalized_query)
    if not compact_query:
        return False
    if item.section == "online_feedback":
        return _has_query_specific_signal(item)

    triggers: list[str] = []
    for term in item.expansion_terms:
        lowered = str(term or "").strip().lower()
        if not lowered or lowered in GENERIC_EXPANSION_MATCH_TERMS:
            continue
        triggers.append(lowered)
        triggers.extend(EXPANSION_TRIGGER_ALIASES.get(lowered, ()))

    for token in _lexical_tokens(item.content):
        lowered = token.lower()
        if lowered in GENERIC_EXPANSION_MATCH_TERMS or len(lowered) < 3:
            continue
        if any(lowered in str(term).lower() for term in item.expansion_terms):
            continue
        triggers.append(lowered)

    for trigger in dedupe_keep_order(triggers):
        token = re.sub(r"\s+", "", trigger.lower())
        if token and token in compact_query:
            return True
    return False


def collect_expansion_terms(
    items: list[PlaybookHit],
    explicit_terms: list[str] | None = None,
    *,
    query: str | None = None,
) -> list[str]:
    values: list[str] = []
    for item in items:
        if item.section not in {"query_expansion", "online_feedback"}:
            continue
        if not _query_expansion_item_applies(item, query):
            continue
        values.extend(item.expansion_terms)
    values.extend(explicit_terms or [])
    return dedupe_keep_order(values)


def collect_source_hints(items: list[PlaybookHit]) -> list[str]:
    values: list[str] = []
    for item in items:
        values.extend(item.source_hints)
    return dedupe_keep_order(values)


def _source_type_from_evidence_id(evidence_id: str) -> str | None:
    parts = [part.strip().lower() for part in str(evidence_id or "").split("::") if part.strip()]
    if not parts:
        return None
    if parts[0] in {"document", "table", "adela"}:
        return parts[0]
    if len(parts) >= 2 and parts[0] == "structured" and parts[1] in {"document", "table", "adela"}:
        return parts[1]
    return None
