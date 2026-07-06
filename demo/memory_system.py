"""
Demo memory system module.

主要功能：
- Raw Ledger：append-only，全量 observation；Planner 的 SSOT。
- Query Trajectory：按 user query（query_id）组织 steps；当前 thread 全部轨迹供 Planner，
  最多 10 条 query；较早的 query 仅携带 result_summary。
- Thread 辅助状态 thread_aux_state：仅用于 RAG 数字事实、资产索引等内部维护，不进入 Planner 主上下文。

主要模块：
- `LedgerStore`：账本规范化、追加、游标同步。
- `QueryTrajectoryStore`：query 生命周期、steps（OBSERVATION / ASSISTANT_OUTPUT + pointer）、Planner 组装。
- `MemoryProjector`：落盘步骤、从轨迹派生 working_memory、更新 session_memory。
- `ContextBuilder`：输出 planner_context（query_trajectories）。
"""

from __future__ import annotations

import json
import os
import re
import secrets
from pathlib import Path

import agent

SESSION_MEMORY_LIMIT = int(os.environ.get("DEMO_SESSION_MEMORY_LIMIT", "6"))
RAW_LEDGER_MAX_EVENTS = int(os.environ.get("DEMO_RAW_LEDGER_MAX", "4000"))
QUERY_TRAJECTORY_MAX_QUERIES_STORED = int(os.environ.get("DEMO_QUERY_TRAJECTORY_MAX_STORED", "48"))
QUERY_TRAJECTORY_PLANNER_WINDOW = int(os.environ.get("DEMO_QUERY_TRAJECTORY_PLANNER_WINDOW", "10"))
QUERY_TRAJECTORY_MAX_STEPS = int(os.environ.get("DEMO_QUERY_TRAJECTORY_MAX_STEPS", "200"))
EXTERNAL_REF_MAX = 2000
CONTEXT_SCHEMA_VERSION = "planner-context-v2"


def _trim_text(value: str, limit: int = 400) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _normalize_text_list(values, *, limit: int = 6, item_limit: int = 160) -> list[str]:
    if not isinstance(values, list):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for item in values:
        text = _trim_text(str(item or "").strip(), item_limit)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _normalize_dict_list(values, *, limit: int = 10) -> list[dict]:
    if not isinstance(values, list):
        return []
    out: list[dict] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, dict):
            continue
        try:
            key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def _drop_references_from_observation(value):
    if not isinstance(value, dict):
        return value
    out = dict(value)
    out.pop("references", None)
    return out


def _ledger_index_by_id(ledger: list) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for e in ledger:
        if not isinstance(e, dict):
            continue
        eid = str(e.get("event_id") or "").strip()
        if eid:
            out[eid] = e
    return out


class LedgerStore:
    """底层账本：每条事件含 session_id + thread_id。"""

    @staticmethod
    def normalize_thread_aux_state(session_id: str, data: dict | None) -> dict:
        src = data if isinstance(data, dict) else {}
        sid = str(src.get("session_id") or session_id or "").strip() or session_id
        user_focus = _trim_text(str(src.get("user_focus") or "").strip(), 300)
        gkv = src.get("global_kv_state")
        if not isinstance(gkv, dict):
            gkv = {}
        try:
            cursor = int(src.get("ledger_cursor") or 0)
        except (TypeError, ValueError):
            cursor = 0
        assets = src.get("asset_registry")
        if not isinstance(assets, list):
            assets = []
        clean_assets = []
        for item in assets:
            if not isinstance(item, dict):
                continue
            aid = str(item.get("id") or "").strip()
            if not aid:
                continue
            clean_assets.append(
                {
                    "id": aid,
                    "type": _trim_text(str(item.get("type") or "asset"), 32),
                    "brief": _trim_text(str(item.get("brief") or ""), 220),
                }
            )
        return {
            "session_id": sid,
            "user_focus": user_focus,
            "global_kv_state": gkv,
            "ledger_cursor": max(0, cursor),
            "asset_registry": clean_assets[:200],
        }

    @staticmethod
    def normalize_ledger_event(
        raw: dict | None,
        *,
        session_id: str,
        default_seq: int,
        default_thread_id: str = "",
    ) -> dict | None:
        if not isinstance(raw, dict):
            return None
        eid = str(raw.get("event_id") or "").strip() or f"evt_{secrets.token_hex(12)}"
        try:
            seq = int(raw.get("seq") or default_seq)
        except (TypeError, ValueError):
            seq = default_seq
        payload = raw.get("payload")
        if payload is None:
            payload = {}
        observation = _trim_text(str(raw.get("observation") or ""), 120)
        tid = str(raw.get("thread_id") or default_thread_id or "").strip()
        return {
            "session_id": str(raw.get("session_id") or session_id or "").strip() or session_id,
            "thread_id": tid,
            "event_id": eid,
            "seq": max(1, seq),
            "event_type": _trim_text(str(raw.get("event_type") or "UNKNOWN"), 64).upper(),
            "observation": observation,
            "payload": payload,
            "external_ref": str(raw.get("external_ref") or "").strip()[:EXTERNAL_REF_MAX],
        }

    @classmethod
    def normalize_raw_ledger(cls, session_id: str, values, *, default_thread_id: str = "") -> list[dict]:
        if not isinstance(values, list):
            return []
        out: list[dict] = []
        for i, item in enumerate(values):
            ev = cls.normalize_ledger_event(
                item, session_id=session_id, default_seq=i + 1, default_thread_id=default_thread_id
            )
            if ev is not None:
                out.append(ev)
        out.sort(key=lambda e: int(e.get("seq") or 0))
        if len(out) > RAW_LEDGER_MAX_EVENTS:
            out = out[-RAW_LEDGER_MAX_EVENTS:]
        return out

    @staticmethod
    def ledger_max_seq(session: dict) -> int:
        ledger = session.get("raw_ledger") or []
        if not isinstance(ledger, list) or not ledger:
            return 0
        return max(int(e.get("seq") or 0) for e in ledger if isinstance(e, dict))

    @classmethod
    def sync_ledger_cursor(cls, session: dict) -> None:
        sid = str(session.get("session_id") or "")
        aux = cls.normalize_thread_aux_state(sid, session.get("thread_aux_state"))
        aux["session_id"] = sid
        aux["ledger_cursor"] = cls.ledger_max_seq(session)
        session["thread_aux_state"] = aux

    @classmethod
    def append_event(
        cls,
        session: dict,
        *,
        event_type: str,
        observation: str = "",
        payload: dict | None = None,
        external_ref: str = "",
        thread_id: str | None = None,
    ) -> dict:
        ledger = session.setdefault("raw_ledger", [])
        seq = cls.ledger_max_seq(session) + 1
        sid = str(session.get("session_id") or "")
        tid = str(thread_id or session.get("active_thread_id") or "").strip()
        data_payload = payload if isinstance(payload, dict) else {}
        ev = {
            "session_id": sid,
            "thread_id": tid,
            "event_id": f"evt_{secrets.token_hex(12)}",
            "seq": seq,
            "event_type": _trim_text(event_type, 64).upper() or "UNKNOWN",
            "observation": _trim_text(observation, 120),
            "payload": data_payload,
            "external_ref": str(external_ref or "").strip()[:EXTERNAL_REF_MAX],
        }
        ledger.append(ev)
        if len(ledger) > RAW_LEDGER_MAX_EVENTS:
            session["raw_ledger"] = ledger[-RAW_LEDGER_MAX_EVENTS:]
        return ev

    @classmethod
    def _migrate_working_trajectory_to_queries(cls, session: dict) -> None:
        wt = session.get("working_trajectory")
        if not isinstance(wt, list) or not wt:
            return
        if session.get("query_trajectories"):
            return
        sid = str(session.get("session_id") or "")
        tid = str(session.get("active_thread_id") or "").strip()
        ledger = session.get("raw_ledger") or []
        by_id = _ledger_index_by_id(ledger if isinstance(ledger, list) else [])
        qid = f"qry_migrated_{secrets.token_hex(6)}"
        steps: list[dict] = []
        for entry in wt:
            if not isinstance(entry, dict):
                continue
            result_ids = entry.get("result_event_ids") if isinstance(entry.get("result_event_ids"), list) else []
            for eid in result_ids:
                ev = by_id.get(str(eid))
                if not isinstance(ev, dict):
                    continue
                et = str(ev.get("event_type") or "").upper()
                if et not in ("OBSERVATION", "ASSISTANT_OUTPUT"):
                    continue
                payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
                steps.append(
                    {
                        "event_type": et,
                        "action": _trim_text(str(payload.get("_action") or ev.get("observation") or ""), 120),
                        "action_input": payload.get("_action_input") if isinstance(payload.get("_action_input"), dict) else {},
                        "status": _trim_text(str(entry.get("status") or "ok"), 16).lower() or "ok",
                        "pointer": str(ev.get("event_id") or ""),
                    }
                )
        session["query_trajectories"] = [
            {
                "session_id": sid,
                "thread_id": tid,
                "query_id": qid,
                "result_summary": "",
                "steps": steps[-QUERY_TRAJECTORY_MAX_STEPS:],
            }
        ]
        session["active_query_id"] = qid

    @classmethod
    def backfill_legacy(cls, session: dict) -> None:
        wm = session.get("working_memory")
        if not isinstance(wm, list) or not wm:
            return
        if session.get("raw_ledger"):
            return
        sid = str(session.get("session_id") or "")
        tid = str(session.get("active_thread_id") or "").strip()
        qid = f"qry_legacy_{secrets.token_hex(6)}"
        session.setdefault("query_trajectories", [])
        QueryTrajectoryStore.start_query(session, query_id=qid, thread_id=tid, session_id=sid)
        for item in wm:
            payload = item if isinstance(item, dict) else {"legacy": str(item)}
            ev = cls.append_event(
                session,
                event_type="OBSERVATION",
                observation=str(payload.get("tool") or "legacy"),
                payload=payload,
                thread_id=tid,
            )
            QueryTrajectoryStore.append_step(
                session,
                action=str(payload.get("tool") or "legacy"),
                observation_event_id=str(ev.get("event_id") or ""),
            )
        cls.sync_ledger_cursor(session)

    @classmethod
    def migrate_schema(cls, session: dict) -> None:
        sid = str(session.get("session_id") or "")
        tid = str(session.get("active_thread_id") or "").strip()
        if isinstance(session.get("session_state"), dict) and not isinstance(session.get("thread_aux_state"), dict):
            session["thread_aux_state"] = dict(session["session_state"])
        session.pop("session_state", None)

        session["raw_ledger"] = cls.normalize_raw_ledger(sid, session.get("raw_ledger"), default_thread_id=tid)
        for ev in session.get("raw_ledger") or []:
            if isinstance(ev, dict) and not str(ev.get("thread_id") or "").strip() and tid:
                ev["thread_id"] = tid

        qt = session.get("query_trajectories")
        if not isinstance(qt, list) or not qt:
            cls._migrate_working_trajectory_to_queries(session)
        session["query_trajectories"] = QueryTrajectoryStore.normalize_list(
            session.get("query_trajectories"), session_id=sid, thread_id_default=tid
        )
        session.pop("working_trajectory", None)

        session["thread_aux_state"] = cls.normalize_thread_aux_state(sid, session.get("thread_aux_state"))
        cls.backfill_legacy(session)
        cls.sync_ledger_cursor(session)


class QueryTrajectoryStore:
    """按 query_id 维护轨迹；steps 仅保存 action 与 observation_event_id，事实内容以 raw_ledger 为 SSOT。"""

    @staticmethod
    def new_query_id() -> str:
        return f"qry_{secrets.token_hex(8)}"

    @classmethod
    def normalize_step(cls, raw: dict | None) -> dict | None:
        if not isinstance(raw, dict):
            return None
        observation_event_id = str(
            raw.get("observation_event_id") or raw.get("pointer") or raw.get("event_id") or ""
        ).strip()
        if not observation_event_id:
            return None
        step_id = str(raw.get("step_id") or "").strip()
        return {
            "step_id": step_id,
            "action": _trim_text(str(raw.get("action") or ""), 120),
            "observation_event_id": observation_event_id,
        }

    @classmethod
    def normalize_trajectory(cls, raw: dict | None, *, session_id: str, thread_id_default: str) -> dict | None:
        if not isinstance(raw, dict):
            return None
        qid = str(raw.get("query_id") or "").strip()
        if not qid:
            return None
        sid = str(raw.get("session_id") or session_id or "").strip() or session_id
        tid = str(raw.get("thread_id") or thread_id_default or "").strip()
        steps_in = raw.get("steps") if isinstance(raw.get("steps"), list) else []
        steps: list[dict] = []
        for s in steps_in:
            ns = cls.normalize_step(s if isinstance(s, dict) else None)
            if ns is not None:
                steps.append(ns)
        if len(steps) > QUERY_TRAJECTORY_MAX_STEPS:
            steps = steps[-QUERY_TRAJECTORY_MAX_STEPS:]
        return {
            "session_id": sid,
            "thread_id": tid,
            "query_id": qid,
            "query": _trim_text(str(raw.get("query") or raw.get("user_text") or ""), 500),
            "result_summary": _trim_text(str(raw.get("result_summary") or ""), 2000),
            "steps": steps,
        }

    @classmethod
    def normalize_list(cls, values, *, session_id: str, thread_id_default: str) -> list[dict]:
        if isinstance(values, dict):
            items = []
            for key, raw in values.items():
                if not isinstance(raw, dict):
                    continue
                item = dict(raw)
                if not str(item.get("query_id") or "").strip():
                    item["query_id"] = str(key or "").strip()
                items.append(item)
            values = items
        if not isinstance(values, list):
            return []
        out: list[dict] = []
        for item in values:
            t = cls.normalize_trajectory(item, session_id=session_id, thread_id_default=thread_id_default)
            if t is not None:
                out.append(t)
        if len(out) > QUERY_TRAJECTORY_MAX_QUERIES_STORED:
            out = out[-QUERY_TRAJECTORY_MAX_QUERIES_STORED:]
        return out

    @classmethod
    def start_query(cls, session: dict, *, query_id: str, thread_id: str | None = None, session_id: str | None = None) -> dict:
        tid = str(thread_id or session.get("active_thread_id") or "").strip()
        sid = str(session_id or session.get("session_id") or "").strip()
        qt = session.setdefault("query_trajectories", [])
        if not isinstance(qt, list):
            qt = []
            session["query_trajectories"] = qt
        entry = {
            "session_id": sid,
            "thread_id": tid,
            "query_id": query_id,
            "query": "",
            "result_summary": "",
            "steps": [],
        }
        qt.append(entry)
        if len(qt) > QUERY_TRAJECTORY_MAX_QUERIES_STORED:
            session["query_trajectories"] = qt[-QUERY_TRAJECTORY_MAX_QUERIES_STORED:]
        session["active_query_id"] = query_id
        return entry

    @classmethod
    def _active_trajectory(cls, session: dict) -> dict | None:
        qid = str(session.get("active_query_id") or "").strip()
        qt = session.get("query_trajectories")
        if not isinstance(qt, list):
            return None
        if qid:
            for t in reversed(qt):
                if isinstance(t, dict) and str(t.get("query_id") or "") == qid:
                    return t
        for t in reversed(qt):
            if isinstance(t, dict):
                return t
        return None

    @classmethod
    def append_step(
        cls,
        session: dict,
        *,
        action: str,
        observation_event_id: str,
    ) -> None:
        traj = cls._active_trajectory(session)
        if not isinstance(traj, dict):
            return
        steps = traj.setdefault("steps", [])
        if not isinstance(steps, list):
            steps = []
            traj["steps"] = steps
        step_no = len(steps) + 1
        steps.append(
            {
                "step_id": f"step_{step_no}",
                "action": _trim_text(action, 120),
                "observation_event_id": str(observation_event_id or "").strip(),
            }
        )
        if len(steps) > QUERY_TRAJECTORY_MAX_STEPS:
            traj["steps"] = steps[-QUERY_TRAJECTORY_MAX_STEPS:]

    @classmethod
    def set_result_summary(cls, session: dict, *, query_id: str, summary: str) -> None:
        qid = str(query_id or "").strip()
        if not qid:
            return
        qt = session.get("query_trajectories")
        if not isinstance(qt, list):
            return
        for t in qt:
            if isinstance(t, dict) and str(t.get("query_id") or "") == qid:
                t["result_summary"] = _trim_text(summary, 2000)
                return

    @classmethod
    def user_text_for_query(cls, session: dict, *, query_id: str, ledger: list[dict]) -> str:
        qid = str(query_id or "").strip()
        qt = session.get("query_trajectories")
        if isinstance(qt, list):
            for tr in qt:
                if not isinstance(tr, dict):
                    continue
                if str(tr.get("query_id") or "") != qid:
                    continue
                query = _trim_text(str(tr.get("query") or tr.get("user_text") or ""), 500)
                if query:
                    return query
        for ev in ledger:
            if not isinstance(ev, dict):
                continue
            if str(ev.get("event_type") or "").upper() != "USER_INPUT":
                continue
            payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
            if str(payload.get("query_id") or "") != qid:
                continue
            return _trim_text(str(payload.get("text") or ""), 500)
        return ""

    @classmethod
    def resolve_query_steps(cls, session: dict, *, query_id: str, ledger: list[dict]) -> list[dict]:
        qid = str(query_id or "").strip()
        if not qid:
            return []
        qt = session.get("query_trajectories")
        if not isinstance(qt, list):
            return []
        target: dict | None = None
        for tr in qt:
            if isinstance(tr, dict) and str(tr.get("query_id") or "") == qid:
                target = tr
                break
        if not isinstance(target, dict):
            return []

        by_id = _ledger_index_by_id(ledger if isinstance(ledger, list) else [])
        out: list[dict] = []
        for idx, st in enumerate(target.get("steps") if isinstance(target.get("steps"), list) else [], start=1):
            if not isinstance(st, dict):
                continue
            ptr = str(st.get("observation_event_id") or st.get("pointer") or "").strip()
            ev = by_id.get(ptr) if ptr else None
            payload = ev.get("payload") if isinstance(ev, dict) and isinstance(ev.get("payload"), dict) else {}
            out.append(
                {
                    "step_id": str(st.get("step_id") or f"step_{idx}"),
                    "action": _trim_text(str(st.get("action") or ""), 120),
                    "observation_event_id": ptr,
                    "observation": payload,
                    "external_ref": str(ev.get("external_ref") or "") if isinstance(ev, dict) else "",
                }
            )
        return out

    @classmethod
    def build_planner_query_bundle(
        cls,
        session: dict,
        *,
        thread_id: str,
        ledger: list[dict],
        max_queries: int | None = None,
    ) -> list[dict]:
        """当前 thread 全量 query 轨迹；最近 N 条保留完整 steps+payload，较早仅 result_summary。"""
        tid = str(thread_id or "").strip()
        cap = max_queries if max_queries is not None else QUERY_TRAJECTORY_PLANNER_WINDOW
        qt_all = session.get("query_trajectories")
        if not isinstance(qt_all, list):
            return []
        filtered = [x for x in qt_all if isinstance(x, dict) and str(x.get("thread_id") or "") == tid]
        if not filtered:
            filtered = [x for x in qt_all if isinstance(x, dict)]
        full_list = filtered
        rich_from = max(0, len(full_list) - max(1, cap))
        out: list[dict] = []
        for i, tr in enumerate(full_list):
            keep_rich = i >= rich_from
            qid = str(tr.get("query_id") or "")
            base = {
                "session_id": str(tr.get("session_id") or session.get("session_id") or ""),
                "thread_id": str(tr.get("thread_id") or tid),
                "query_id": qid,
                "query": cls.user_text_for_query(session, query_id=qid, ledger=ledger if isinstance(ledger, list) else []),
            }
            if keep_rich:
                rich_steps = cls.resolve_query_steps(
                    session,
                    query_id=qid,
                    ledger=ledger if isinstance(ledger, list) else [],
                )
                for row in rich_steps:
                    if not isinstance(row, dict):
                        continue
                    row["observation"] = _drop_references_from_observation(row.get("observation"))
                out.append({**base, "result_summary": str(tr.get("result_summary") or ""), "steps": rich_steps})
            else:
                out.append(
                    {
                        **base,
                        "result_summary": _trim_text(str(tr.get("result_summary") or ""), 2000),
                        "steps": [],
                    }
                )
        return out


class MemoryProjector:
    """落盘与 working_memory / session_memory 派生。"""

    @staticmethod
    def _guess_goal(step: dict, fallback_text: str) -> str:
        action_input = step.get("action_input") or {}
        for key in ("query", "task_text", "label", "goal"):
            value = str(action_input.get(key) or "").strip()
            if value:
                return _trim_text(value, 220)
        return _trim_text(fallback_text, 220)

    @staticmethod
    def _extract_artifacts(observation: dict) -> list[str]:
        obs = observation if isinstance(observation, dict) else {}
        out: list[str] = []
        if isinstance(obs.get("references"), list) and obs.get("references"):
            out.append(f"references:{len(obs.get('references'))}")
        if isinstance(obs.get("generated_images"), list) and obs.get("generated_images"):
            out.append(f"generated_images:{len(obs.get('generated_images'))}")
        if isinstance(obs.get("annotated_urls"), list) and obs.get("annotated_urls"):
            out.append(f"annotated_images:{len(obs.get('annotated_urls'))}")
        if isinstance(obs.get("evaluation"), dict) and obs.get("evaluation"):
            out.append("evaluation_report")
        return out[:6]

    @staticmethod
    def _extract_points(observation: dict) -> list[dict]:
        obs = observation if isinstance(observation, dict) else {}
        points: list[dict] = []
        summary = _trim_text(str(obs.get("summary") or ""), 600)
        if summary:
            points.append({"type": "summary", "text": summary})
        if isinstance(obs.get("intent_summary"), dict):
            intent_data: dict = {}
            for key in ("task_name", "scene", "target", "camera"):
                val = _trim_text(str(obs["intent_summary"].get(key) or ""), 100)
                if val:
                    intent_data[key] = val
            if intent_data:
                points.append({"type": "intent_summary", "data": intent_data})
        if isinstance(obs.get("evaluation"), dict):
            for key in ("overall_conclusion", "recommendation"):
                val = _trim_text(str(obs["evaluation"].get(key) or ""), 160)
                if val:
                    points.append({"type": "evaluation", "field": key, "text": val})
        return _normalize_dict_list(points, limit=8)

    @staticmethod
    def _observation_text(observation: dict | None) -> str:
        obs = observation if isinstance(observation, dict) else {}
        for key in ("summary", "answer"):
            text = str(obs.get(key) or "").strip()
            if text:
                return text
        try:
            return json.dumps(obs, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(obs)

    @classmethod
    def _trajectory_brief_from_observation(cls, observation: dict | None) -> str:
        return _trim_text(cls._observation_text(observation), 200)

    @classmethod
    def _upsert_rag_numeric_facts(cls, aux: dict, *, observation: dict, action_input: dict) -> None:
        text = cls._observation_text(observation)
        if not text:
            return
        lines = re.split(r"[\n。；;]+", text)
        numeric_lines = []
        for line in lines:
            seg = _trim_text(str(line or "").strip(), 220)
            if not seg:
                continue
            if re.search(r"\d", seg):
                numeric_lines.append(seg)
        if not numeric_lines:
            return
        gkv = aux.get("global_kv_state")
        if not isinstance(gkv, dict):
            gkv = {}
            aux["global_kv_state"] = gkv
        query = _trim_text(str(action_input.get("query") or ""), 160)
        record = {
            "query": query,
            "facts": _normalize_text_list(numeric_lines, limit=12, item_limit=220),
        }
        bucket = gkv.get("rag_numeric_facts")
        if not isinstance(bucket, list):
            bucket = []
        exists = False
        for item in bucket:
            if not isinstance(item, dict):
                continue
            if str(item.get("query") or "") == record["query"]:
                item["facts"] = record["facts"]
                exists = True
                break
        if not exists:
            bucket.append(record)
        gkv["rag_numeric_facts"] = bucket[-50:]

    @classmethod
    def _append_rag_long_excerpt_asset(
        cls,
        aux: dict,
        *,
        obs_event_id: str,
        observation: dict,
    ) -> None:
        text = cls._observation_text(observation)
        if len(text) <= 1000:
            return
        assets = aux.get("asset_registry")
        if not isinstance(assets, list):
            assets = []
            aux["asset_registry"] = assets
        for item in assets:
            if isinstance(item, dict) and str(item.get("id") or "") == obs_event_id and str(item.get("type") or "") == "rag_excerpt":
                return
        assets.append(
            {
                "id": obs_event_id,
                "type": "rag_excerpt",
                "brief": _trim_text(text, 220),
            }
        )

    @classmethod
    def build_working_memory_item(cls, step: dict, fallback_text: str, observation: dict) -> dict:
        action = agent.normalize_agent_action(str(step.get("action") or "").strip())
        action_input = step.get("action_input") or {}
        core_obs = observation if isinstance(observation, dict) else {}
        return {
            "goal": cls._guess_goal(step, fallback_text),
            "tool": action,
            "action_input": action_input if isinstance(action_input, dict) else {},
            "key_points": cls._extract_points(observation),
            "observation_core": {
                "summary": _trim_text(str(core_obs.get("summary") or ""), 600),
                "intent_summary": core_obs.get("intent_summary") if isinstance(core_obs.get("intent_summary"), dict) else {},
                "evaluation": core_obs.get("evaluation") if isinstance(core_obs.get("evaluation"), dict) else {},
            },
            "artifacts": cls._extract_artifacts(observation),
            "next_step_hint": "可以直接向用户给出最终答复。"
            if bool(action_input.get("finish_after_tool", False))
            else "基于本次结果继续判断是否需要下一步工具。",
        }

    @classmethod
    def observation_status(cls, observation: dict | None) -> str:
        obs = observation if isinstance(observation, dict) else {}
        if obs.get("success") is False:
            return "error"
        if str(obs.get("error") or "").strip():
            return "error"
        return "ok"

    @classmethod
    def persist_step(
        cls,
        session: dict,
        *,
        step_index: int,
        text: str,
        step: dict,
        thought: str,
        observation: dict | None,
        run_dir: Path,
    ) -> None:
        action = agent.normalize_agent_action(str(step.get("action") or "").strip())
        action_input = step.get("action_input") if isinstance(step.get("action_input"), dict) else {}
        obs_dump: dict = dict(observation) if isinstance(observation, dict) else {}
        obs_dump["_action"] = action
        obs_dump["_action_input"] = action_input
        obs_dump["_thought"] = thought
        obs_path = run_dir / f"agent_step_{step_index}_observation.json"
        ref_path = ""
        try:
            obs_path.write_text(json.dumps(obs_dump, ensure_ascii=False, indent=2), encoding="utf-8")
            ref_path = str(obs_path.resolve())
        except (OSError, TypeError, ValueError):
            ref_path = ""

        tid = str(session.get("active_thread_id") or "").strip()
        LedgerStore.append_event(
            session,
            event_type="PLAN_DECISION",
            observation=action,
            payload={"action": action, "action_input": action_input, "thought": thought},
            external_ref="",
            thread_id=tid,
        )
        obs_ev = LedgerStore.append_event(
            session,
            event_type="OBSERVATION",
            observation=action,
            payload=obs_dump if isinstance(obs_dump, dict) else {},
            external_ref=ref_path,
            thread_id=tid,
        )
        QueryTrajectoryStore.append_step(
            session,
            action=action,
            observation_event_id=str(obs_ev.get("event_id") or ""),
        )

        sid = str(session.get("session_id") or "")
        aux = LedgerStore.normalize_thread_aux_state(sid, session.get("thread_aux_state"))
        if isinstance(observation, dict) and observation.get("report"):
            aux["asset_registry"].append(
                {"id": obs_ev["event_id"], "type": "report", "brief": _trim_text(str((observation or {}).get("summary") or ""), 220)}
            )
        if action == agent.TOOL_RAG_ANSWER and isinstance(observation, dict):
            cls._upsert_rag_numeric_facts(aux, observation=observation, action_input=action_input)
            cls._append_rag_long_excerpt_asset(
                aux,
                obs_event_id=str(obs_ev.get("event_id") or ""),
                observation=observation,
            )
        if ref_path and (str(ref_path).endswith(".png") or str(ref_path).endswith(".jpg") or str(ref_path).endswith(".jpeg")):
            aux["asset_registry"].append(
                {"id": obs_ev["event_id"], "type": "image", "brief": _trim_text(action, 220)}
            )
        aux["user_focus"] = _trim_text(text, 300)
        session["thread_aux_state"] = aux
        LedgerStore.sync_ledger_cursor(session)

    @classmethod
    def derive_working_memory(cls, session: dict, limit: int) -> list[dict]:
        ledger_list = session.get("raw_ledger") or []
        if not isinstance(ledger_list, list):
            return []
        by_id = _ledger_index_by_id(ledger_list)
        qt = session.get("query_trajectories")
        if not isinstance(qt, list):
            return []
        steps_flat: list[tuple[int, dict]] = []
        for qi, tr in enumerate(qt):
            if not isinstance(tr, dict):
                continue
            for st in tr.get("steps") if isinstance(tr.get("steps"), list) else []:
                if not isinstance(st, dict):
                    continue
                steps_flat.append((qi, st))
        out: list[dict] = []
        for _qi, st in steps_flat[-max(1, limit * 6) :]:
            ptr = str(st.get("observation_event_id") or st.get("pointer") or "").strip()
            ev = by_id.get(ptr)
            if not isinstance(ev, dict):
                continue
            payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
            step = {
                "action": payload.get("_action") or ev.get("observation") or "",
                "action_input": payload.get("_action_input") or {},
            }
            out.append(
                cls.build_working_memory_item(
                    step, str(payload.get("summary") or payload.get("_thought") or ""), payload
                )
            )
        return out[-max(1, limit) :]

    @classmethod
    def update_session_memory(cls, session_memory: dict | None, *, text: str, working_memory: list[dict], final_answer: str, effective_image_path: str) -> dict:
        ctx = session_memory if isinstance(session_memory, dict) else {}
        out = {
            "task_profile": {
                "current_goal": "",
                "active_tools": [],
                "latest_status": "",
                "has_reference_image": bool(effective_image_path),
            },
            "user_focus": {"current_query": _trim_text(text, 240), "query_focus": []},
            "confirmed_facts": {"facts": []},
            "session_summary": {"completed_steps": [], "tool_takeaways": [], "latest_answer": _trim_text(final_answer, 240)},
        }
        if isinstance(ctx.get("task_profile"), dict):
            out["task_profile"].update(ctx["task_profile"])
        if isinstance(ctx.get("user_focus"), dict):
            out["user_focus"].update(ctx["user_focus"])
        if isinstance(ctx.get("confirmed_facts"), dict):
            out["confirmed_facts"].update(ctx["confirmed_facts"])
        if isinstance(ctx.get("session_summary"), dict):
            out["session_summary"].update(ctx["session_summary"])

        latest = working_memory[-1] if working_memory else {}
        latest_goal = _trim_text(str(latest.get("goal") or text), 200)
        latest_tool = _trim_text(str(latest.get("tool") or ""), 80)
        out["task_profile"]["current_goal"] = latest_goal
        tools = list(out["task_profile"].get("active_tools") or [])
        if latest_tool:
            tools.append(latest_tool)
        out["task_profile"]["active_tools"] = _normalize_text_list(tools, limit=8, item_limit=64)
        out["task_profile"]["latest_status"] = _trim_text(final_answer or out["task_profile"].get("latest_status") or "已完成当前轮次。", 220)

        chunks = re.split(r"[，,。；;！!\n]+", str(text or "").strip())
        out["user_focus"]["query_focus"] = _normalize_text_list(chunks, limit=8, item_limit=120)

        facts = list(out["confirmed_facts"].get("facts") or [])
        for item in working_memory[-3:]:
            for p in item.get("key_points") or []:
                if isinstance(p, dict):
                    facts.append(p)
        out["confirmed_facts"]["facts"] = _normalize_dict_list(facts, limit=10)

        completed = list(out["session_summary"].get("completed_steps") or [])
        if latest_tool or latest_goal:
            completed.append(f"{latest_tool or 'tool'}: {latest_goal}")
        out["session_summary"]["completed_steps"] = _normalize_text_list(completed, limit=10, item_limit=180)
        takes = list(out["session_summary"].get("tool_takeaways") or [])
        takes = list(latest.get("key_points") or []) + takes
        out["session_summary"]["tool_takeaways"] = _normalize_dict_list(takes, limit=10)
        out["session_summary"]["latest_answer"] = _trim_text(final_answer, 240)
        return out


class ContextBuilder:
    @classmethod
    def build_prompt_context(cls, session: dict, *, text: str, effective_image_path: str) -> dict:
        tid = str(session.get("active_thread_id") or "").strip()
        ledger = session.get("raw_ledger") if isinstance(session.get("raw_ledger"), list) else []
        queries = QueryTrajectoryStore.build_planner_query_bundle(session, thread_id=tid, ledger=ledger)
        return {
            "schema_version": CONTEXT_SCHEMA_VERSION,
            "session_id": str(session.get("session_id") or ""),
            "active_thread_id": tid,
            "query_trajectories": queries,
        }
