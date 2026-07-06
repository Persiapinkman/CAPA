#!/usr/bin/env python3
"""
生成完整评测套件：
1) 为 session_questions_eval 自动打 suggested_tags
2) 输出 session_cases_eval.json（模板 + 从 sessions 挖掘的多轮 case）
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

from build_from_sessions import collect_questions, write_outputs
from tag_questions import suggest_tags

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SESSIONS_DIR = ROOT / "demo" / "sessions"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent


def apply_tags_to_items(items: list[dict]) -> None:
    for item in items:
        meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        tags, auto_note = suggest_tags(
            item.get("question") or "",
            is_clarification_reply_seen=bool(meta.get("is_clarification_reply_seen")),
            occurrence_count=int(meta.get("occurrence_count") or 1),
        )
        item["suggested_tags"] = tags
        existing_note = str(item.get("notes") or "").strip()
        if auto_note:
            item["notes"] = f"{existing_note}；{auto_note}".strip("；") if existing_note else auto_note


def mine_multi_turn_cases(sessions_dir: Path, *, limit: int = 20) -> list[dict]:
    cases: list[dict] = []
    seen: set[str] = set()

    def add_case(case: dict) -> None:
        key = json.dumps(case.get("turns") or [], ensure_ascii=False)
        if key in seen:
            return
        seen.add(key)
        cases.append(case)

    for path in sorted(sessions_dir.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        sid = str(data.get("session_id") or path.stem)
        threads = data.get("threads") if isinstance(data.get("threads"), dict) else {}
        for tid, tdata in threads.items():
            if not isinstance(tdata, dict):
                continue
            ledger = tdata.get("raw_ledger")
            if not isinstance(ledger, list):
                continue
            prev_text = ""
            prev_is_clar = False
            for ev in ledger:
                if str(ev.get("event_type") or "").upper() != "USER_INPUT":
                    continue
                p = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
                text = str(p.get("effective_text") or p.get("text") or "").strip()
                if not text or "用户补充说明" in text and len(text) > 100:
                    prev_text = ""
                    continue
                is_clar = bool(p.get("is_clarification_reply"))
                if prev_text and not prev_is_clar:
                    t1, t2 = prev_text, text
                    # 跳过纯槽位链（留给 CL 模板）
                    if len(t2) < 40 and re.search(
                        r"cuda\d+\.\d+-trt", t2, re.I
                    ) and "部署" not in t2:
                        prev_text, prev_is_clar = text, is_clar
                        continue
                    tags, _ = suggest_tags(t2, is_clarification_reply_seen=is_clar)
                    case_id = f"MT-MINED-{len(cases)+1:03d}"
                    add_case(
                        {
                            "case_id": case_id,
                            "title": f"会话挖掘：{t1[:24]}… → {t2[:24]}…",
                            "interaction": "multi_turn",
                            "source": {
                                "type": "session_mined",
                                "session_id": sid,
                                "thread_id": str(tid),
                            },
                            "turns": [
                                {"role": "user", "text": t1},
                                {"role": "user", "text": t2},
                            ],
                            "setup": {"same_session": True, "same_thread": True, "image": None},
                            "expect": {
                                "suggested_tags": list(dict.fromkeys(["MT-01"] + tags)),
                                "tool_sequence": [],
                                "tool_sequence_notes": "第二轮依赖 query_trajectories；请按实际标注 tool_sequence",
                                "max_planner_steps": 8,
                                "must_not": [],
                            },
                            "ground_truth": {"per_turn": ["", ""]},
                            "judge": {"mode": "trajectory_plus_answer"},
                        }
                    )
                    if len(cases) >= limit:
                        return cases
                prev_text, prev_is_clar = text, is_clar
    return cases


def template_cases() -> list[dict]:
  """手工标准 case 模板（10~15 条）。"""
  return [
        {
            "case_id": "MT-01-001",
            "title": "RAG 事实问 + 短追问（性能）",
            "interaction": "multi_turn",
            "source": {"type": "template"},
            "turns": [
                {"role": "user", "text": "安全绳检测在adela平台有部署模型吗"},
                {"role": "user", "text": "评测性能怎么样"},
            ],
            "setup": {"same_session": True, "same_thread": True, "image": None},
            "expect": {
                "suggested_tags": ["MT-01", "lookup_fact", "adela_benchmark", "MT-fragment", "PL-1"],
                "tool_sequence": [
                    ["rag_answer", {"finish_after_tool": True}],
                    ["rag_answer", {"finish_after_tool": True}],
                ],
                "tool_sequence_alternatives": [
                    [
                        ["rag_answer", {"finish_after_tool": True}],
                        ["adela_cli_eval", {"finish_after_tool": True}],
                    ]
                ],
                "max_planner_steps": 6,
                "must_not": ["pipeline_eval"],
            },
            "ground_truth": {"per_turn": ["", ""]},
            "judge": {"mode": "trajectory_plus_answer"},
        },
        {
            "case_id": "MT-01-002",
            "title": "指代消解：推荐模型 → 精度",
            "interaction": "multi_turn",
            "source": {"type": "template"},
            "turns": [
                {"role": "user", "text": "安全绳检测有哪些模型"},
                {"role": "user", "text": "这个模型在 T4 上精度怎么样"},
            ],
            "setup": {"same_session": True, "same_thread": True, "image": None},
            "expect": {
                "suggested_tags": ["MT-01", "recommend_model", "coreference", "PL-4", "finish:mixed"],
                "tool_sequence": [
                    ["rag_answer", {"finish_after_tool": False}],
                    ["re_question", {"finish_after_tool": False}],
                    ["rag_answer", {"finish_after_tool": True}],
                ],
                "tool_sequence_alternatives": [
                    [
                        ["rag_answer", {"finish_after_tool": True}],
                        ["adela_cli_eval", {"finish_after_tool": True}],
                    ]
                ],
                "max_planner_steps": 8,
                "must_not": [],
            },
            "ground_truth": {"per_turn": ["", ""]},
            "judge": {"mode": "trajectory_plus_answer"},
        },
        {
            "case_id": "MT-01-003",
            "title": "时间追问：2025 → 2024",
            "interaction": "multi_turn",
            "source": {"type": "template"},
            "turns": [
                {"role": "user", "text": "2025年RD部门总共发布了多少个模型？"},
                {"role": "user", "text": "那2024年呢？"},
            ],
            "setup": {"same_session": True, "same_thread": True, "image": None},
            "expect": {
                "suggested_tags": ["MT-01", "lookup_fact", "PL-1", "finish:true"],
                "tool_sequence": [
                    ["rag_answer", {"finish_after_tool": True}],
                    ["rag_answer", {"finish_after_tool": True}],
                ],
                "max_planner_steps": 4,
                "must_not": ["clarify"],
            },
            "ground_truth": {"per_turn": ["", ""]},
            "judge": {"mode": "trajectory_plus_answer"},
        },
        {
            "case_id": "CL-01-001",
            "title": "Adela 澄清：部署意图 → 平台 → 模型名 → 精度",
            "interaction": "clarification_resume",
            "source": {"type": "template"},
            "turns": [
                {"role": "user", "text": "部署钓鱼人检测模型"},
                {"role": "user", "text": "cuda11.0-trt7.1-int8-T4"},
                {"role": "user", "text": "fishing_det_small"},
                {"role": "user", "text": "精度评测"},
            ],
            "setup": {"same_session": True, "same_thread": True, "image": None},
            "expect": {
                "suggested_tags": ["CL-01", "MT-02", "CL-slot", "adela_benchmark", "PL-1", "finish:true"],
                "tool_sequence": [
                    ["adela_cli_eval", {"finish_after_tool": True}],
                ],
                "tool_sequence_notes": "前几轮应为 clarification；最后一轮 forced adela_cli_eval",
                "max_planner_steps": 2,
                "must_not": ["rag_answer"],
            },
            "ground_truth": {"final_summary": ""},
            "judge": {"mode": "clarification_then_tool"},
        },
        {
            "case_id": "CL-01-002",
            "title": "Adela 模型名歧义澄清",
            "interaction": "clarification_resume",
            "source": {"type": "template"},
            "turns": [
                {
                    "role": "user",
                    "text": "帮我看一下 struct_blackbox_crop_224x224 部署到 cuda11.0-trt7.1-fp16-T4 后的精度怎么样？",
                },
            ],
            "setup": {"same_session": True, "same_thread": True, "image": None},
            "expect": {
                "suggested_tags": ["ST-01", "adela_benchmark", "PL-1", "finish:true"],
                "tool_sequence": [["adela_cli_eval", {"finish_after_tool": True}]],
                "tool_sequence_alternatives": [
                    [
                        ["rag_answer", {"finish_after_tool": True}],
                        ["adela_cli_eval", {"finish_after_tool": True}],
                    ]
                ],
                "max_planner_steps": 4,
                "must_not": [],
            },
            "ground_truth": {"final_summary": ""},
            "judge": {"mode": "trajectory_plus_answer"},
        },
        {
            "case_id": "PL-4-001",
            "title": "RAG 未命中闭环（冷门推荐）",
            "interaction": "single_turn",
            "source": {"type": "template"},
            "turns": [
                {"role": "user", "text": "检测电子元器异常，用什么模型？"},
            ],
            "setup": {"same_session": True, "new_thread": True, "image": None},
            "expect": {
                "suggested_tags": ["ST-01", "recommend_model", "PL-4", "finish:mixed"],
                "tool_sequence": [
                    ["rag_answer", {"finish_after_tool": False}],
                    ["re_question", {"finish_after_tool": False}],
                    ["rag_answer", {"finish_after_tool": False}],
                    ["answerer", {"mode": "rag_evidence", "finish_after_tool": True}],
                ],
                "tool_sequence_notes": "允许状态机强制 re_question；3 轮 miss 后 answerer",
                "max_planner_steps": 10,
                "must_not": [],
            },
            "ground_truth": {"final_answer": ""},
            "judge": {"mode": "trajectory_plus_answer"},
        },
        {
            "case_id": "PL-2-001",
            "title": "生图 → 检测（finish 串联）",
            "interaction": "multi_turn",
            "source": {"type": "template"},
            "turns": [
                {"role": "user", "text": "生成3张钓鱼的图片"},
                {"role": "user", "text": "用qwen模型标注钓鱼的人"},
            ],
            "setup": {"same_session": True, "same_thread": True, "image": None},
            "expect": {
                "suggested_tags": ["MT-01", "vision_generate", "vision_detect", "PL-2", "finish:mixed"],
                "tool_sequence": [
                    ["flux-image-generation", {"finish_after_tool": False}],
                    ["qwen_detection", {"finish_after_tool": True}],
                ],
                "max_planner_steps": 6,
                "must_not": [],
            },
            "ground_truth": {"per_turn": ["", ""]},
            "judge": {"mode": "trajectory_plus_answer"},
        },
        {
            "case_id": "PL-3-001",
            "title": "评测流水线（需参考图）",
            "interaction": "single_turn",
            "source": {"type": "template"},
            "turns": [
                {"role": "user", "text": "钓鱼的人，生成一个精度评估报告"},
            ],
            "setup": {"same_session": True, "new_thread": True, "image": "fixture/fishing_scene.jpg"},
            "expect": {
                "suggested_tags": ["ST-01", "vision_pipeline", "PL-3", "finish:true"],
                "tool_sequence": [["pipeline_eval", {"finish_after_tool": True}]],
                "tool_sequence_alternatives": [
                    [
                        ["flux-image-generation", {"finish_after_tool": False}],
                        ["qwen_detection", {"finish_after_tool": False}],
                        ["pipeline_eval", {"finish_after_tool": True}],
                    ]
                ],
                "max_planner_steps": 8,
                "must_not": [],
            },
            "ground_truth": {"final_answer": ""},
            "judge": {"mode": "trajectory_plus_answer"},
        },
        {
            "case_id": "ST-CL-001",
            "title": "意图歧义应 clarify",
            "interaction": "single_turn",
            "source": {"type": "template"},
            "turns": [{"role": "user", "text": "黑夜检测黑猫"}],
            "setup": {"same_session": True, "new_thread": True, "image": None},
            "expect": {
                "suggested_tags": ["ST-01", "needs_clarify", "PL-1", "finish:true"],
                "decision_type": "clarify",
                "tool_sequence": [],
                "max_planner_steps": 1,
                "must_not": ["rag_answer", "flux-image-generation", "pipeline_eval"],
            },
            "ground_truth": {"clarification_question": ""},
            "judge": {"mode": "clarify_only"},
        },
        {
            "case_id": "ST-01-001",
            "title": "单轮事实查库",
            "interaction": "single_turn",
            "source": {"type": "template", "linked_question_id": "q_0012"},
            "turns": [
                {
                    "role": "user",
                    "text": "城市结构化检测模型 3.0.1是目前最新的支持人体检测的模型么",
                },
            ],
            "setup": {"same_session": True, "new_thread": True, "image": None},
            "expect": {
                "suggested_tags": ["ST-01", "lookup_fact", "PL-1", "finish:true"],
                "tool_sequence": [["rag_answer", {"finish_after_tool": True}]],
                "max_planner_steps": 2,
                "must_not": [],
            },
            "ground_truth": {"final_answer": ""},
            "judge": {"mode": "trajectory_plus_answer"},
        },
        {
            "case_id": "MIG-01-001",
            "title": "迁移顾问：能力边界评估",
            "interaction": "single_turn",
            "source": {"type": "template", "linked_question_id": "q_0111"},
            "turns": [
                {
                    "role": "user",
                    "text": "香港项目要求识别“头盔颜色：黑、白、灰、红、蓝、黄、绿、紫、迷彩”，现有城市结构化人体属性模型是否能直接支持？",
                },
            ],
            "setup": {"same_session": True, "new_thread": True, "image": None},
            "expect": {
                "suggested_tags": ["ST-01", "migration_advisor", "PL-1", "finish:true"],
                "tool_sequence": [],
                "tool_sequence_notes": "期望 migration 抽象+多路检索+固定章节报告；或 rag 后提供迁移追问",
                "max_planner_steps": 6,
                "must_not": [],
            },
            "ground_truth": {"final_answer": ""},
            "judge": {"mode": "trajectory_plus_answer"},
        },
        {
            "case_id": "GEN-01-001",
            "title": "通用知识 / 自我介绍",
            "interaction": "single_turn",
            "source": {"type": "template", "linked_question_id": "q_0055"},
            "turns": [{"role": "user", "text": "介绍一下自己"}],
            "setup": {"same_session": True, "new_thread": True, "image": None},
            "expect": {
                "suggested_tags": ["ST-01", "general_chat", "PL-1", "finish:true"],
                "tool_sequence": [],
                "tool_sequence_alternatives": [[["answerer", {"mode": "direct", "finish_after_tool": True}]]],
                "max_planner_steps": 1,
                "must_not": ["rag_answer"],
            },
            "ground_truth": {"final_answer": ""},
            "judge": {"mode": "answer_only"},
        },
        {
            "case_id": "MT-FRAG-001",
            "title": "无效单轮片段（应用例绑定）",
            "interaction": "fragment",
            "source": {"type": "template", "linked_question_id": "q_0017"},
            "turns": [{"role": "user", "text": "性能评测怎么样"}],
            "setup": {"invalid_standalone": True, "bind_to_case_id": "MT-01-001"},
            "expect": {
                "suggested_tags": ["MT-fragment", "adela_benchmark", "PL-1"],
                "notes": "禁止单独跑；仅作为 MT-01-001 第 2 轮",
            },
            "ground_truth": {},
            "judge": {"mode": "skip_standalone"},
        },
        {
            "case_id": "VIS-01-001",
            "title": "检测缺参应澄清",
            "interaction": "single_turn",
            "source": {"type": "template", "linked_question_id": "q_0033"},
            "turns": [{"role": "user", "text": "检测背书包的人"}],
            "setup": {"same_session": True, "new_thread": True, "image": None},
            "expect": {
                "suggested_tags": ["ST-01", "vision_detect", "needs_clarify", "PL-1"],
                "decision_type": "clarify",
                "tool_sequence_notes": "无图无 label 时应 tool_precondition 或 clarify",
                "max_planner_steps": 2,
                "must_not": ["pipeline_eval"],
            },
            "ground_truth": {"clarification_question": ""},
            "judge": {"mode": "clarify_or_precondition"},
        },
    ]


def write_tagged_questions(items: list[dict], out_dir: Path, sessions_dir: Path, stats: dict) -> None:
    apply_tags_to_items(items)
    dataset = {
        "schema_version": "1.1",
        "description": "单轮题库 + suggested_tags；ground_truth 待补充。",
        "source_dir": str(sessions_dir.resolve()),
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "stats": stats,
        "tag_legend": {
            "interaction": ["ST-01", "MT-01", "MT-fragment", "MT-02", "CL-01", "CL-slot"],
            "intent": [
                "lookup_fact",
                "recommend_model",
                "migration_advisor",
                "adela_benchmark",
                "vision_generate",
                "vision_detect",
                "vision_pipeline",
                "general_chat",
                "needs_clarify",
            ],
            "planner": ["PL-1", "PL-2", "PL-3", "PL-4"],
            "finish": ["finish:true", "finish:false", "finish:mixed"],
        },
        "items": items,
    }
    json_path = out_dir / "session_questions_eval.json"
    json_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = out_dir / "session_questions_eval.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "id",
                "question",
                "ground_truth",
                "tags",
                "suggested_tags",
                "notes",
                "occurrence_count",
                "first_seen_date",
                "is_clarification_reply_seen",
            ]
        )
        for item in items:
            m = item["metadata"]
            w.writerow(
                [
                    item["id"],
                    item["question"],
                    item.get("ground_truth") or "",
                    "|".join(item.get("tags") or []),
                    "|".join(item.get("suggested_tags") or []),
                    item.get("notes") or "",
                    m.get("occurrence_count", 0),
                    m.get("first_seen_date", ""),
                    m.get("is_clarification_reply_seen", False),
                ]
            )


def write_cases(cases: list[dict], out_dir: Path) -> None:
    by_interaction: dict[str, int] = {}
    for c in cases:
        k = str(c.get("interaction") or "unknown")
        by_interaction[k] = by_interaction.get(k, 0) + 1

    payload = {
        "schema_version": "1.0",
        "description": "多轮/澄清/闭环标准 case + 会话挖掘 case；ground_truth 待补充。",
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "stats": {
            "total_cases": len(cases),
            "template_cases": sum(1 for c in cases if (c.get("source") or {}).get("type") == "template"),
            "mined_cases": sum(1 for c in cases if (c.get("source") or {}).get("type") == "session_mined"),
            "by_interaction": by_interaction,
        },
        "cases": cases,
    }
    path = out_dir / "session_cases_eval.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tagged questions + eval cases")
    parser.add_argument("--sessions-dir", type=Path, default=DEFAULT_SESSIONS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--mine-limit", type=int, default=12)
    args = parser.parse_args()

    items, stats = collect_questions(args.sessions_dir)
    write_tagged_questions(items, args.out_dir, args.sessions_dir, stats)

    templates = template_cases()
    mined = mine_multi_turn_cases(args.sessions_dir, limit=args.mine_limit)
    all_cases = templates + mined

    write_cases(all_cases, args.out_dir)

    print(json.dumps({"questions": stats, "cases": len(all_cases), "mined": len(mined)}, ensure_ascii=False))
    print(f"wrote {args.out_dir / 'session_questions_eval.csv'}")
    print(f"wrote {args.out_dir / 'session_cases_eval.json'}")


if __name__ == "__main__":
    main()
