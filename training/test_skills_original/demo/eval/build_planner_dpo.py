#!/usr/bin/env python3
"""Mine planner DPO preference pairs from recorded llm_debug traces.

输入：
- demo/llm_debug/**/*_planner_request.json + 对应 *_planner_response.json
- （可选）demo/sessions/**/*.json，用于序列失败诊断（不直接造训练对）

产出（写入 --out-dir，默认 demo/eval/dpo）：
- planner_dpo_pairs.jsonl      每行一个偏好对 {prompt, chosen, rejected, meta}
- planner_labeled_steps.jsonl  每个 planner 步骤的弱标签（good/bad/neutral + reasons）
- planner_sequence_findings.jsonl  从 sessions 挖出的序列失败（clarify 循环 / re_question 超限 / 空决策），带疑似根因
- planner_dpo_report.json / .md     规则命中与质量汇总

偏好对的两种构造（meta.pair_type）：
- repair：rejected=模型真实产出的坏决策，chosen=规则修复后的合成决策（chosen_synthetic=true，需人工复核）。
- contrastive：chosen=模型真实产出的好决策，rejected=规则合成的典型错误（rejected_synthetic=true，chosen 完全真实，更安全）。

忠实性说明：
- prompt 一律取自 llm_debug 的真实 messages（system+user），不重建。
- sessions 的序列失败仅作诊断清单输出（含 judger/planner 根因区分），因其缺少当时 prompt 快照、
  且部分 clarify 循环根因在 rag_judger 而非 planner，不适合直接作为 planner 训练对。
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LLM_DEBUG_DIR = ROOT / "demo" / "llm_debug"
DEFAULT_SESSIONS_DIR = ROOT / "demo" / "sessions"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "dpo"

# 部署平台串：cuda11.0-trt7.1-int8-T4 之类；或单独的设备/精度标识。
PLATFORM_FULL_RE = re.compile(r"cuda[\d.]+-trt[\d.]+-(?:int8|fp16|fp32|int4)-(?:T4|P4|A2|V100|A100|3090|cpu)", re.I)
PLATFORM_HINT_RE = re.compile(r"(cuda[\d.]+-trt|trt[\d.]+-|-T4\b|-P4\b|-A2\b|int8-|fp16-|fp32-)", re.I)
PERF_KW = ("性能", "耗时", "速度", "latency", "吞吐", "qps", "fps", "时延")
ACC_KW = ("精度", "准确", "accuracy", "acc", "map", "召回", "recall", "precision", "f1")
# 指代词：触发 re_question 实体补全。
COREF_RE = re.compile(r"(这个|这些|该模型|该方案|上述|上面|它的|它在|其在|前面那个|刚才那个|\bit\b|\bthis\b|\bthat\b)", re.I)
# 通用常识/闲聊信号：本不该走 RAG。
GENERIC_RE = re.compile(
    r"(是当季|怎么读|讲个|你是谁|你能做什么|用非技术语言|通俗解释|解释一下.*(区别|概念|原理)|"
    r"什么是\s*(机器学习|深度学习|神经网络|object detection|precision|recall)|举个例子说明)",
    re.I,
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_user_context(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """从 user message 里抽出内嵌的 {query, query_trajectories} JSON。"""
    for msg in messages:
        if str(msg.get("role")) != "user":
            continue
        content = str(msg.get("content") or "")
        brace = content.find("{")
        if brace < 0:
            continue
        snippet = content[brace:]
        try:
            return json.loads(snippet, strict=False)
        except Exception:
            last = snippet.rfind("}")
            if last > 0:
                try:
                    return json.loads(snippet[: last + 1], strict=False)
                except Exception:
                    return {}
    return {}


def parse_decision(raw_response: str) -> dict[str, Any] | None:
    raw = str(raw_response or "").strip()
    if not raw:
        return None
    # strict=False 允许字符串值内含裸换行/制表符（模型常见输出），救回大量“假退化”。
    try:
        obj = json.loads(raw, strict=False)
        return obj if isinstance(obj, dict) else None
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(raw[start : end + 1], strict=False)
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None
    return None


@dataclass
class PlannerStep:
    session_id: str
    run_stamp: str
    step_index: int
    created_at: str
    request_file: Path
    messages: list[dict[str, Any]]
    query: str
    trajectories: list[dict[str, Any]]
    decision: dict[str, Any]
    decision_type: str
    action: str
    action_input: dict[str, Any]
    raw_response: str
    parse_ok: bool = True
    has_response: bool = True


def load_planner_steps(llm_debug_dir: Path) -> list[PlannerStep]:
    steps: list[PlannerStep] = []
    for req_path in sorted(llm_debug_dir.rglob("*_planner_request.json")):
        resp_path = req_path.with_name(req_path.name.replace("_request.json", "_response.json"))
        try:
            req = load_json(req_path)
        except Exception:
            continue
        payload = req.get("payload") if isinstance(req.get("payload"), dict) else {}
        messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
        ctx = parse_user_context(messages)
        query = str(ctx.get("query") or "").strip()
        trajs = ctx.get("query_trajectories") if isinstance(ctx.get("query_trajectories"), list) else []

        raw_response = ""
        if resp_path.is_file():
            try:
                resp = load_json(resp_path)
                raw_response = str((resp.get("payload") or {}).get("raw_response") or "")
            except Exception:
                raw_response = ""
        decision = parse_decision(raw_response) or {}
        parse_ok = bool(decision)

        steps.append(
            PlannerStep(
                session_id=str(req.get("session_id") or req_path.stem),
                run_stamp=str(req.get("run_stamp") or ""),
                step_index=int(req.get("step_index") or 0),
                created_at=str(req.get("created_at") or ""),
                request_file=req_path,
                messages=messages,
                query=query,
                trajectories=trajs,
                decision=decision,
                decision_type=str(decision.get("decision_type") or "").strip(),
                action=str(decision.get("action") or "").strip(),
                action_input=decision.get("action_input") if isinstance(decision.get("action_input"), dict) else {},
                raw_response=raw_response,
                parse_ok=parse_ok,
                has_response=bool(raw_response.strip()),
            )
        )
    return steps


def group_by_session(steps: list[PlannerStep]) -> dict[str, list[PlannerStep]]:
    groups: dict[str, list[PlannerStep]] = {}
    for st in steps:
        groups.setdefault(st.session_id, []).append(st)
    for sid in groups:
        groups[sid].sort(key=lambda s: (s.created_at, s.step_index))
    return groups


# ----------------------------- 工具函数 ----------------------------- #


def detect_eval_type(query: str) -> int:
    q = query.lower()
    if any(k.lower() in q for k in PERF_KW):
        return 1
    if any(k.lower() in q for k in ACC_KW):
        return 0
    return 0


def extract_platform(query: str) -> str:
    m = PLATFORM_FULL_RE.search(query)
    return m.group(0) if m else ""


def prior_queries_of(step: PlannerStep) -> list[str]:
    out: list[str] = []
    for t in step.trajectories:
        if not isinstance(t, dict):
            continue
        q = str(t.get("query") or "").strip()
        if q and q != step.query:
            out.append(q)
    return out


def is_platform_eval_query(query: str) -> bool:
    return bool(PLATFORM_HINT_RE.search(query)) and (
        any(k in query for k in ACC_KW) or any(k in query for k in PERF_KW)
    )


def make_decision_json(decision: dict[str, Any]) -> str:
    return json.dumps(decision, ensure_ascii=False)


@dataclass
class Pair:
    rule_id: str
    pair_type: str  # repair | contrastive
    fail_reason: str
    confidence: float
    chosen: dict[str, Any]
    rejected: dict[str, Any]
    chosen_synthetic: bool
    rejected_synthetic: bool
    step: PlannerStep


# ----------------------------- repair 规则（真实坏决策 -> 合成修复） ----------------------------- #


def rule_platform_misroute(step: PlannerStep) -> Pair | None:
    if step.action != "rag_answer" or not is_platform_eval_query(step.query):
        return None
    platform = extract_platform(step.query)
    chosen_input: dict[str, Any] = {"platform": platform, "eval_type": detect_eval_type(step.query), "finish_after_tool": True}
    mname = re.search(r"[A-Za-z][A-Za-z0-9_./\-]{6,}\.model", step.query)
    if mname:
        chosen_input["model_name"] = mname.group(0)
    chosen = {
        "thought": "用户已给出部署平台标识并询问该模型在该平台上的精度/性能，应使用 adela_cli_eval 而非通用 RAG 问答。",
        "decision_type": "tool",
        "action": "adela_cli_eval",
        "action_input": chosen_input,
    }
    return Pair(
        rule_id="R1_platform_misroute",
        pair_type="repair",
        fail_reason="部署平台精度/性能问题被错误路由到 rag_answer，应为 adela_cli_eval",
        confidence=0.85 if platform else 0.6,
        chosen=chosen,
        rejected=step.decision,
        chosen_synthetic=True,
        rejected_synthetic=False,
        step=step,
    )


def rule_coref_no_rewrite(step: PlannerStep) -> Pair | None:
    if step.action != "rag_answer" or not COREF_RE.search(step.query):
        return None
    priors = prior_queries_of(step)
    if not priors:
        return None
    chosen = {
        "thought": "当前问题含指代且存在历史问题，应先用 re_question 做实体补全再检索，避免指代未消解导致检索不稳。",
        "decision_type": "tool",
        "action": "re_question",
        "action_input": {
            "query": step.query,
            "rewrite_reason": "coref_resolve",
            "retrieval_round": 1,
            "context_hint": priors[-1],
            "finish_after_tool": False,
        },
    }
    return Pair(
        rule_id="R2_coref_no_rewrite",
        pair_type="repair",
        fail_reason="含指代词且有历史轨迹却未先 re_question，直接 rag_answer",
        confidence=0.6,
        chosen=chosen,
        rejected=step.decision,
        chosen_synthetic=True,
        rejected_synthetic=False,
        step=step,
    )


def rule_clarify_loop(step: PlannerStep, session_steps: list[PlannerStep], idx: int) -> Pair | None:
    if step.decision_type != "clarify":
        return None
    prior_clarify = sum(1 for s in session_steps[:idx] if s.decision_type == "clarify")
    if prior_clarify < 1:
        return None
    if prior_clarify >= 2:
        chosen = {
            "thought": "已多次向用户澄清仍未推进，应停止反复追问，基于现有上下文证据用 answerer 给出尽力而为的回答。",
            "decision_type": "tool",
            "action": "answerer",
            "action_input": {"mode": "rag_evidence", "finish_after_tool": True},
        }
        conf = 0.7
    else:
        chosen = {
            "thought": "上一轮已澄清过，不应继续追问，应直接用 rag_answer 检索用户问题。",
            "decision_type": "tool",
            "action": "rag_answer",
            "action_input": {"query": step.query or "", "finish_after_tool": True},
        }
        conf = 0.6
    return Pair(
        rule_id="R3_clarify_loop",
        pair_type="repair",
        fail_reason=f"同一会话内第 {prior_clarify + 1} 次 clarify，疑似澄清死循环未止损",
        confidence=conf,
        chosen=chosen,
        rejected=step.decision,
        chosen_synthetic=True,
        rejected_synthetic=False,
        step=step,
    )


def rule_requestion_overrun(step: PlannerStep, session_steps: list[PlannerStep], idx: int) -> Pair | None:
    if step.action != "re_question":
        return None
    prior_rq = sum(1 for s in session_steps[:idx] if s.action == "re_question")
    rd = step.action_input.get("retrieval_round")
    if not (prior_rq >= 3 or (isinstance(rd, int) and rd > 3)):
        return None
    chosen = {
        "thought": "改写检索已达上限仍未命中，应停止继续 re_question，改用 answerer(rag_evidence) 基于已有弱证据兜底回答。",
        "decision_type": "tool",
        "action": "answerer",
        "action_input": {"mode": "rag_evidence", "finish_after_tool": True},
    }
    return Pair(
        rule_id="R4_requestion_overrun",
        pair_type="repair",
        fail_reason=f"re_question 轮次超限（prior={prior_rq}, round={rd}）仍未兜底",
        confidence=0.7,
        chosen=chosen,
        rejected=step.decision,
        chosen_synthetic=True,
        rejected_synthetic=False,
        step=step,
    )


def rule_empty_decision(step: PlannerStep) -> Pair | None:
    """planner 有响应但输出退化/截断/不可解析，或 action 与 decision_type 全空：无效决策。

    仅当 has_response（确有 LLM 产出）时才视为模型坏决策；纯缺失响应文件的属于数据缺失，跳过。
    """
    if not step.has_response:
        return None
    if step.parse_ok and (step.decision_type or step.action):
        return None
    if not step.query:
        return None
    if GENERIC_RE.search(step.query):
        chosen = {
            "thought": "通用常识问题，应输出合法的 answerer(direct) 决策，而非退化/截断输出。",
            "decision_type": "tool",
            "action": "answerer",
            "action_input": {"mode": "direct", "finish_after_tool": True},
        }
    else:
        chosen = {
            "thought": "企业知识检索类问题，应输出合法的 rag_answer 决策，而非退化/截断输出。",
            "decision_type": "tool",
            "action": "rag_answer",
            "action_input": {"query": step.query, "finish_after_tool": True},
        }
    rejected = {"_raw_degenerate": (step.raw_response[:300] + "...<truncated degenerate output>")}
    return Pair(
        rule_id="R5_degenerate_decision",
        pair_type="repair",
        fail_reason="planner 输出退化/截断（如 thought 陷入重复 token），未产出合法决策；chosen 为占位修复，需复核真实意图",
        confidence=0.45,
        chosen=chosen,
        rejected=rejected,
        chosen_synthetic=True,
        rejected_synthetic=False,
        step=step,
    )


REPAIR_SINGLE = (rule_platform_misroute, rule_coref_no_rewrite, rule_empty_decision)
REPAIR_SEQ = (rule_clarify_loop, rule_requestion_overrun)


# ----------------------------- contrastive 规则（真实好决策 -> 合成反例） ----------------------------- #


def contrastive_platform(step: PlannerStep) -> Pair | None:
    """B1：平台精度/性能问题正确走 adela_cli_eval -> 反例 rag_answer。"""
    if step.action != "adela_cli_eval" or not is_platform_eval_query(step.query):
        return None
    rejected = {
        "thought": "这看起来是企业知识问题，用 rag_answer 检索即可。",
        "decision_type": "tool",
        "action": "rag_answer",
        "action_input": {"query": step.query, "finish_after_tool": True},
    }
    return Pair(
        rule_id="B1_platform_eval_correct",
        pair_type="contrastive",
        fail_reason="对照：平台精度/性能问题应 adela_cli_eval，rag_answer 为典型错误",
        confidence=0.8,
        chosen=step.decision,
        rejected=rejected,
        chosen_synthetic=False,
        rejected_synthetic=True,
        step=step,
    )


def contrastive_coref(step: PlannerStep) -> Pair | None:
    """B2：含指代正确走 re_question -> 反例 直接 rag_answer 不改写。"""
    if step.action != "re_question" or not COREF_RE.search(step.query):
        return None
    if not prior_queries_of(step):
        return None
    rejected = {
        "thought": "直接拿原问题检索即可。",
        "decision_type": "tool",
        "action": "rag_answer",
        "action_input": {"query": step.query, "finish_after_tool": True},
    }
    return Pair(
        rule_id="B2_coref_rewrite_correct",
        pair_type="contrastive",
        fail_reason="对照：含指代应先 re_question，直接 rag_answer 不改写为典型错误",
        confidence=0.7,
        chosen=step.decision,
        rejected=rejected,
        chosen_synthetic=False,
        rejected_synthetic=True,
        step=step,
    )


def contrastive_generic(step: PlannerStep) -> Pair | None:
    """B3：通用常识问题正确走 answerer(direct) -> 反例 rag_answer 过度检索。"""
    if step.action != "answerer":
        return None
    mode = str(step.action_input.get("mode") or "")
    if mode and mode != "direct":
        return None
    if not GENERIC_RE.search(step.query):
        return None
    rejected = {
        "thought": "去知识库检索一下。",
        "decision_type": "tool",
        "action": "rag_answer",
        "action_input": {"query": step.query, "finish_after_tool": True},
    }
    return Pair(
        rule_id="B3_generic_direct_correct",
        pair_type="contrastive",
        fail_reason="对照：通用常识应 answerer(direct)，对其做 RAG 检索为过度检索",
        confidence=0.65,
        chosen=step.decision,
        rejected=rejected,
        chosen_synthetic=False,
        rejected_synthetic=True,
        step=step,
    )


CONTRASTIVE = (contrastive_platform, contrastive_coref, contrastive_generic)


# ----------------------------- 评估主流程 ----------------------------- #


def evaluate_steps(
    groups: dict[str, list[PlannerStep]], include_contrastive: bool
) -> tuple[list[Pair], list[dict[str, Any]]]:
    pairs: list[Pair] = []
    labeled: list[dict[str, Any]] = []
    for sid, session_steps in groups.items():
        for idx, step in enumerate(session_steps):
            reasons: list[str] = []
            repair_pairs: list[Pair] = []
            for fn in REPAIR_SINGLE:
                p = fn(step)
                if p:
                    repair_pairs.append(p)
            for fn in REPAIR_SEQ:
                p = fn(step, session_steps, idx)
                if p:
                    repair_pairs.append(p)

            contrastive_pairs: list[Pair] = []
            if include_contrastive and not repair_pairs:
                for fn in CONTRASTIVE:
                    p = fn(step)
                    if p:
                        contrastive_pairs.append(p)

            step_pairs = repair_pairs + contrastive_pairs
            for p in step_pairs:
                reasons.append(p.rule_id)
            if repair_pairs:
                label = "bad"
            elif not step.has_response:
                label = "skipped_no_response"
            elif not step.parse_ok:
                label = "neutral"
            elif contrastive_pairs:
                label = "good_anchor"
            else:
                label = "good"
            pairs.extend(step_pairs)
            labeled.append(
                {
                    "session_id": sid,
                    "run_stamp": step.run_stamp,
                    "step_index": step.step_index,
                    "query": step.query,
                    "decision_type": step.decision_type,
                    "action": step.action,
                    "label": label,
                    "reasons": reasons,
                    "request_file": str(step.request_file),
                }
            )
    return pairs, labeled


# ----------------------------- sessions 序列诊断（不造训练对） ----------------------------- #


def diagnose_sessions(sessions_dir: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if not sessions_dir.is_dir():
        return findings
    for path in sorted(sessions_dir.rglob("*.json")):
        try:
            data = load_json(path)
        except Exception:
            continue
        threads = data.get("threads")
        if not isinstance(threads, dict):
            continue
        for tid, thread in threads.items():
            if not isinstance(thread, dict):
                continue
            ledger = thread.get("raw_ledger") if isinstance(thread.get("raw_ledger"), list) else []
            # 统计 clarification 来源（区分 rag_judger 与 planner）。
            judger_clarify = 0
            planner_clarify = 0
            for ev in ledger:
                payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
                obs = str(ev.get("observation") or "")
                if obs in ("clarification", "clarify"):
                    src = str((payload.get("pending_clarification") or {}).get("source") or "")
                    if src == "rag_judger":
                        judger_clarify += 1
                    else:
                        planner_clarify += 1
            qmap = thread.get("query_trajectories") if isinstance(thread.get("query_trajectories"), dict) else {}
            for qid, traj in qmap.items():
                if not isinstance(traj, dict):
                    continue
                actions = [str(s.get("action") or "") for s in traj.get("steps", []) if isinstance(s, dict)]
                clarify_n = sum(1 for a in actions if a == "clarify")
                rq_n = sum(1 for a in actions if a == "re_question")
                empty_n = sum(1 for a in actions if a == "")
                issues: list[str] = []
                if clarify_n >= 2:
                    issues.append("clarify_loop")
                if rq_n >= 3:
                    issues.append("requestion_overrun")
                if empty_n >= 1:
                    issues.append("empty_action")
                if not issues:
                    continue
                root_cause = "rag_judger" if (("clarify_loop" in issues) and judger_clarify >= planner_clarify and judger_clarify > 0) else "planner"
                findings.append(
                    {
                        "session_path": str(path),
                        "session_id": str(data.get("session_id") or path.stem),
                        "thread_id": str(tid),
                        "query_id": str(qid),
                        "query": str(traj.get("query") or ""),
                        "action_sequence": actions,
                        "issues": issues,
                        "clarify_count": clarify_n,
                        "requestion_count": rq_n,
                        "empty_action_count": empty_n,
                        "suspected_root_cause": root_cause,
                        "note": "clarify 循环若根因为 rag_judger，应优先调编排/judger 阈值，而非 planner DPO",
                    }
                )
    return findings


# ----------------------------- 输出与报告 ----------------------------- #


def pair_to_record(p: Pair, prompt_style: str) -> dict[str, Any]:
    sys_msg = next((m for m in p.step.messages if str(m.get("role")) == "system"), None)
    usr_msg = next((m for m in p.step.messages if str(m.get("role")) == "user"), None)
    chosen_str = make_decision_json(p.chosen)
    rejected_str = make_decision_json(p.rejected)
    if prompt_style == "messages":
        # TRL conversational 格式：prompt 为消息列表，chosen/rejected 为 assistant 消息列表。
        prompt: Any = [m for m in (sys_msg, usr_msg) if m]
        chosen: Any = [{"role": "assistant", "content": chosen_str}]
        rejected: Any = [{"role": "assistant", "content": rejected_str}]
    else:
        # TRL standard 格式：三者均为字符串。
        prompt = "\n\n".join(str(m.get("content") or "") for m in (sys_msg, usr_msg) if m)
        chosen = chosen_str
        rejected = rejected_str
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "meta": {
            "rule_id": p.rule_id,
            "pair_type": p.pair_type,
            "fail_reason": p.fail_reason,
            "confidence": round(p.confidence, 3),
            "chosen_synthetic": p.chosen_synthetic,
            "rejected_synthetic": p.rejected_synthetic,
            "session_id": p.step.session_id,
            "run_stamp": p.step.run_stamp,
            "step_index": p.step.step_index,
            "query": p.step.query,
            "source_file": str(p.step.request_file),
        },
    }


def build_report(
    steps: list[PlannerStep], pairs: list[Pair], labeled: list[dict[str, Any]], findings: list[dict[str, Any]]
) -> dict[str, Any]:
    rule_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    for p in pairs:
        rule_counts[p.rule_id] = rule_counts.get(p.rule_id, 0) + 1
        type_counts[p.pair_type] = type_counts.get(p.pair_type, 0) + 1
    label_counts: dict[str, int] = {}
    for row in labeled:
        label_counts[row["label"]] = label_counts.get(row["label"], 0) + 1
    action_counts: dict[str, int] = {}
    for st in steps:
        key = st.action or st.decision_type or "unparsed"
        action_counts[key] = action_counts.get(key, 0) + 1
    finding_issue_counts: dict[str, int] = {}
    root_cause_counts: dict[str, int] = {}
    for fd in findings:
        for iss in fd["issues"]:
            finding_issue_counts[iss] = finding_issue_counts.get(iss, 0) + 1
        root_cause_counts[fd["suspected_root_cause"]] = root_cause_counts.get(fd["suspected_root_cause"], 0) + 1
    return {
        "total_planner_steps": len(steps),
        "parse_failed": sum(1 for s in steps if not s.parse_ok),
        "sessions": len({s.session_id for s in steps}),
        "total_pairs": len(pairs),
        "pairs_by_type": dict(sorted(type_counts.items(), key=lambda x: -x[1])),
        "pairs_by_rule": dict(sorted(rule_counts.items(), key=lambda x: -x[1])),
        "labels": label_counts,
        "action_distribution": dict(sorted(action_counts.items(), key=lambda x: -x[1])),
        "session_findings_total": len(findings),
        "session_findings_by_issue": dict(sorted(finding_issue_counts.items(), key=lambda x: -x[1])),
        "session_findings_by_root_cause": dict(sorted(root_cause_counts.items(), key=lambda x: -x[1])),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Planner DPO 偏好对挖掘报告", ""]
    lines.append(f"- planner 步骤总数：{report['total_planner_steps']}（来自 llm_debug 真实 prompt）")
    lines.append(f"- 涉及会话数：{report['sessions']}")
    lines.append(f"- 决策解析失败：{report['parse_failed']}")
    lines.append(f"- 生成偏好对：**{report['total_pairs']}**")
    lines.append("")
    lines.append("## 偏好对按类型")
    lines.append("")
    lines.append("| pair_type | 数量 | 说明 |")
    lines.append("|-----------|------|------|")
    desc = {
        "repair": "rejected=真实坏决策，chosen=合成修复（需复核）",
        "contrastive": "chosen=真实好决策，rejected=合成反例（chosen 真实，更安全）",
    }
    for k, v in report["pairs_by_type"].items():
        lines.append(f"| {k} | {v} | {desc.get(k, '')} |")
    lines.append("")
    lines.append("## 各规则命中数")
    lines.append("")
    lines.append("| rule_id | 偏好对数 |")
    lines.append("|---------|----------|")
    for rid, cnt in report["pairs_by_rule"].items():
        lines.append(f"| {rid} | {cnt} |")
    lines.append("")
    lines.append("## 步骤弱标签分布")
    lines.append("")
    lines.append("| label | 数量 |")
    lines.append("|-------|------|")
    for k, v in report["labels"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## 决策动作分布")
    lines.append("")
    lines.append("| action | 数量 |")
    lines.append("|--------|------|")
    for k, v in report["action_distribution"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## sessions 序列失败诊断（不直接造训练对）")
    lines.append("")
    lines.append(f"- 命中序列失败：{report['session_findings_total']} 条")
    lines.append(f"- 按问题类型：{report['session_findings_by_issue']}")
    lines.append(f"- 按疑似根因：{report['session_findings_by_root_cause']}")
    lines.append("")
    lines.append("> 注：repair 对的 chosen 为合成修复，contrastive 对的 rejected 为合成反例；")
    lines.append("> 建议按 confidence 降序人工抽检后再用于 DPO。clarify 循环若根因为 rag_judger，")
    lines.append("> 应优先调整编排/judger 阈值，而非用 planner DPO 修复。")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine planner DPO preference pairs from llm_debug traces")
    parser.add_argument("--llm-debug-dir", type=Path, default=DEFAULT_LLM_DEBUG_DIR)
    parser.add_argument("--sessions-dir", type=Path, default=DEFAULT_SESSIONS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prompt-style", choices=["messages", "text"], default="messages")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="仅导出 confidence>=该阈值 的偏好对")
    parser.add_argument("--no-contrastive", action="store_true", help="不生成 contrastive 对照对，仅 repair 对")
    parser.add_argument("--no-sessions", action="store_true", help="跳过 sessions 序列诊断")
    args = parser.parse_args()

    if not args.llm_debug_dir.is_dir():
        raise SystemExit(f"llm_debug dir not found: {args.llm_debug_dir}")

    steps = load_planner_steps(args.llm_debug_dir)
    groups = group_by_session(steps)
    pairs, labeled = evaluate_steps(groups, include_contrastive=not args.no_contrastive)
    pairs = [p for p in pairs if p.confidence >= args.min_confidence]
    findings = [] if args.no_sessions else diagnose_sessions(args.sessions_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = args.out_dir / "planner_dpo_pairs.jsonl"
    with pairs_path.open("w", encoding="utf-8") as f:
        for p in sorted(pairs, key=lambda x: -x.confidence):
            f.write(json.dumps(pair_to_record(p, args.prompt_style), ensure_ascii=False) + "\n")

    labeled_path = args.out_dir / "planner_labeled_steps.jsonl"
    with labeled_path.open("w", encoding="utf-8") as f:
        for row in labeled:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    findings_path = args.out_dir / "planner_sequence_findings.jsonl"
    with findings_path.open("w", encoding="utf-8") as f:
        for fd in findings:
            f.write(json.dumps(fd, ensure_ascii=False) + "\n")

    report = build_report(steps, pairs, labeled, findings)
    (args.out_dir / "planner_dpo_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out_dir / "planner_dpo_report.md").write_text(render_markdown(report), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nwrote: {pairs_path}")
    print(f"wrote: {labeled_path}")
    print(f"wrote: {findings_path}")
    print(f"wrote: {args.out_dir / 'planner_dpo_report.json'}")
    print(f"wrote: {args.out_dir / 'planner_dpo_report.md'}")


if __name__ == "__main__":
    main()
