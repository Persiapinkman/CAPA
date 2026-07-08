#!/usr/bin/env python3
"""Build synthetic planner DPO train-seed pairs.

This script creates a reviewable training seed for single-step Planner routing
boundaries. It intentionally excludes tool-internal state-machine issues
(Adela slot clarification) and multi-step composite tasks.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEMO_DIR = ROOT / "demo"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import agent  # noqa: E402

DEFAULT_OUT_DIR = ROOT / "results" / "planner_routing_eval" / "dpo_train_seed"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _planner_messages(case: dict[str, Any]) -> list[dict[str, str]]:
    image_path = ""
    setup = case.get("setup") if isinstance(case.get("setup"), dict) else {}
    if setup.get("has_image"):
        fixture = str(setup.get("image_fixture") or "").strip()
        if fixture:
            path = Path(fixture)
            if not path.is_absolute():
                path = ROOT / path
            if path.is_file():
                image_path = str(path.resolve())
    return [
        {"role": "system", "content": agent.build_agent_system_prompt(max_steps=agent.AGENT_MAX_STEPS)},
        {
            "role": "user",
            "content": agent.build_agent_user_prompt(
                str(case.get("user_query") or ""),
                image_path or None,
                planner_context={
                    "session_id": f"planner_dpo_train_seed_{case.get('case_id')}",
                    "query_trajectories": [],
                },
                step_index=1,
                max_steps=agent.AGENT_MAX_STEPS,
            ),
        },
    ]


def _decision(
    *,
    thought: str,
    action: str,
    action_input: dict[str, Any],
) -> dict[str, Any]:
    return {
        "thought": thought,
        "decision_type": "tool",
        "action": action,
        "action_input": action_input,
        "final_answer": "",
    }


def _pair(
    *,
    case: dict[str, Any],
    error_type: str,
    chosen: dict[str, Any],
    rejected: dict[str, Any],
    rationale: str,
) -> dict[str, Any]:
    return {
        "prompt": _planner_messages(case),
        "chosen": json.dumps(chosen, ensure_ascii=False),
        "rejected": json.dumps(rejected, ensure_ascii=False),
        "meta": {
            "case_id": case["case_id"],
            "title": case["title"],
            "category": case["category"],
            "error_type": error_type,
            "user_query": case["user_query"],
            "pair_type": "synthetic_train_seed",
            "chosen_synthetic": True,
            "rejected_synthetic": True,
            "needs_human_review": True,
            "human_review_status": "todo",
            "rationale": rationale,
        },
    }


def _answerer_pairs() -> list[dict[str, Any]]:
    queries = [
        ("标注框质量建议", "做目标检测数据集时，标注框质量一般要注意哪些问题？"),
        ("IoU 概念解释", "IoU 是什么，目标检测里为什么常用它？"),
        ("PR 曲线解释", "PR 曲线对目标检测评估有什么帮助？"),
        ("召回优先指标", "如果业务更怕漏检，评估指标应该怎么侧重？"),
        ("置信度阈值选择", "检测模型上线时置信度阈值通常怎么选？"),
        ("误报漏报归因", "分析检测误报和漏报时，一般从哪些方面排查？"),
        ("训练集划分", "目标检测训练集、验证集、测试集一般怎么划分？"),
        ("长尾类别处理", "检测类别长尾严重时有哪些常见处理办法？"),
        ("负样本作用", "目标检测任务里为什么需要收集负样本？"),
        ("数据增强风险", "用数据增强提升检测效果时有什么常见风险？"),
        ("开放词表检测", "open-vocabulary detection 和普通检测有什么区别？"),
        ("标注一致性", "多人标注同一批图片时，怎么检查标注一致性？"),
        ("小目标优化", "小目标检测效果差，常见优化方向有哪些？"),
        ("边缘部署权衡", "检测模型部署到边缘设备时，速度和精度怎么权衡？"),
        ("混淆矩阵", "多类别检测结果可以用混淆矩阵分析吗？"),
        ("mAP 解释", "mAP、precision、recall 分别是什么意思？"),
        ("主动学习", "检测任务样本不够时，主动学习可以怎么做？"),
        ("OOV 解释", "视觉模型里的 OOV 问题通常指什么？"),
        ("测试集泄漏", "目标检测评估里测试集泄漏会带来什么问题？"),
        ("类别不均衡", "检测数据类别不均衡时，训练和评估要注意什么？"),
        ("NMS 解释", "NMS 在目标检测后处理里起什么作用？"),
        ("过拟合判断", "怎么判断一个检测模型是不是过拟合了？"),
        ("人工复核抽样", "检测模型上线后，人工抽样复核一般怎么设计？"),
        ("模型漂移", "线上检测模型效果漂移通常有哪些原因？"),
        ("标注边界", "目标被遮挡时，标注框应该标可见区域还是完整区域？"),
        ("难例挖掘", "目标检测里的 hard negative mining 是什么？"),
        ("校准问题", "检测模型的置信度不校准会造成什么影响？"),
        ("评估粒度", "为什么检测模型要按场景和类别分别看指标？"),
        ("阈值分层", "不同类别是否可以设置不同置信度阈值？"),
        ("质检指标", "数据标注质检常用哪些指标？"),
        ("召回精度取舍", "precision 和 recall 冲突时一般怎么取舍？"),
        ("小样本风险", "只用很少样本评估检测模型有什么风险？"),
        ("跨域泛化", "目标检测跨域泛化差通常怎么分析？"),
        ("置信度排序", "为什么有时 mAP 高但业务体验仍然不好？"),
        ("在线监控", "检测模型上线后应该监控哪些质量指标？"),
        ("分辨率影响", "输入分辨率变化会怎样影响检测效果？"),
        ("重叠目标", "密集重叠目标会给检测评估带来什么问题？"),
        ("类别定义", "检测任务前为什么要先定义清楚类别边界？"),
        ("抽样偏差", "测试集抽样偏差会如何影响模型评估？"),
        ("错误分析报告", "一份目标检测错误分析报告通常包含哪些部分？"),
    ]
    out: list[dict[str, Any]] = []
    for idx, (title, query) in enumerate(queries, start=1):
        case = {
            "case_id": f"DPO-ANS-{idx:03d}",
            "title": title,
            "category": "general_answer",
            "user_query": query,
            "setup": {"has_image": False},
        }
        chosen = _decision(
            thought="用户询问通用概念或方法建议，不依赖内部资产库，也没有要求执行工具评测，应直接回答。",
            action=agent.TOOL_ANSWERER,
            action_input={"mode": "direct", "finish_after_tool": True},
        )
        rejected = _decision(
            thought="用户问题涉及目标检测评估方法，可能属于企业规范或业务文档，应优先检索知识库。",
            action=agent.TOOL_RAG_ANSWER,
            action_input={"query": query, "finish_after_tool": True},
        )
        out.append(
            _pair(
                case=case,
                error_type="general_answer_over_rag",
                chosen=chosen,
                rejected=rejected,
                rationale="通用方法/概念问题不应过度路由到 RAG。",
            )
        )
    return out


def _visual_detection_pairs() -> list[dict[str, Any]]:
    fixtures = [
        ("examples/images/trash_truck.jpg", "垃圾车"),
        ("examples/images/person_with_bag.png", "背包"),
        ("examples/images/smoke.jpg", "烟雾"),
        ("examples/images/fisherman.jpg", "垂钓人员"),
        ("examples/images/banner.jpg", "横幅"),
    ]
    templates = [
        "帮我看这张图里有没有{label}。",
        "检测一下当前图片中的{label}。",
        "这张图里是否出现了{label}？直接检测一下。",
        "帮我框出图片里的{label}。",
        "判断这张图片里有没有{label}目标。",
        "只需要检测这张图中的{label}，不用出完整报告。",
        "请看一下上传图里是否有{label}。",
        "对这张图跑一下{label}检测。",
    ]
    out: list[dict[str, Any]] = []
    idx = 1
    for fixture, label in fixtures:
        for template in templates:
            query = template.format(label=label)
            case = {
                "case_id": f"DPO-VIS-{idx:03d}",
                "title": f"单图{label}检测",
                "category": "executable_vision_probe",
                "user_query": query,
                "setup": {"has_image": True, "image_fixture": fixture},
            }
            chosen = _decision(
                thought="用户只要求对当前图片做单步目标检测，不需要生成样本、模型对比或完整评测报告。",
                action=agent.TOOL_QWEN_DETECTION,
                action_input={"label": label, "finish_after_tool": True},
            )
            rejected = _decision(
                thought="用户提出了目标检测需求，适合调用完整视觉评测流水线生成样本、执行检测并形成评估结果。",
                action=agent.TOOL_PIPELINE_EVAL,
                action_input={"task_text": query, "finish_after_tool": True},
            )
            out.append(
                _pair(
                    case=case,
                    error_type="single_image_detection_over_pipeline",
                    chosen=chosen,
                    rejected=rejected,
                    rationale="单图检测不应升级为完整 pipeline_eval。",
                )
            )
            idx += 1
    return out


def _historical_asset_pairs() -> list[dict[str, Any]]:
    queries = [
        ("工程车资产盘点", "我们现在沉淀过工程车或挖掘机检测能力吗？"),
        ("犬只识别资产", "我们有没有犬只识别或宠物入园检测的历史能力？"),
        ("安全帽资产", "目前有没有安全帽佩戴状态识别模型？支持哪些标签？"),
        ("反光衣资产", "历史资产里有没有反光衣识别能力？对应模型是什么？"),
        ("烟火资产", "现有烟火检测模型支持 smoke 和 fire 标签吗？"),
        ("垃圾车资产", "项目库里有没有垃圾车检测能力？历史交付用哪个模型？"),
        ("跌倒资产", "目前有没有人员跌倒检测相关能力？"),
        ("渔船资产", "渔船检测之前交付过哪些版本？"),
        ("裸土覆盖资产", "已有裸土覆盖识别能力吗？历史报告里模型效果如何？"),
        ("安全绳资产", "安全绳模型现在支持哪些标签和版本？"),
        ("口罩资产", "历史资产里有没有口罩佩戴识别模型？"),
        ("占道经营资产", "以前有没有做过占道经营识别？项目名称是什么？"),
        ("电动车进楼资产", "电动车进楼检测有没有已有模型可以查？"),
        ("积水识别资产", "历史交付里有没有道路积水识别能力？"),
        ("工服识别资产", "工服穿戴检测现在有哪些历史能力？"),
        ("烟雾火焰维护", "当前烟雾火焰检测能力最近版本是什么？"),
        ("施工围挡资产", "施工围挡破损识别有没有历史方案？"),
        ("水尺读数资产", "水尺读数识别现有能力做到什么程度？"),
        ("叉车检测资产", "仓库叉车检测是否已有资产？模型名是什么？"),
        ("厨师帽资产", "餐饮后厨厨师帽检测以前做过吗？"),
        ("救生衣资产", "有没有救生衣穿戴检测的历史模型？"),
        ("灭火器资产", "历史项目里有没有灭火器缺失检测能力？"),
        ("井盖资产", "井盖缺失或破损检测有没有已有模型？"),
        ("安全带资产", "车内安全带识别以前是否交付过？"),
        ("车牌遮挡资产", "有没有车牌遮挡检测相关历史能力？"),
        ("蓝牌货车资产", "蓝牌货车识别有历史模型或标签记录吗？"),
        ("道路抛洒物资产", "道路抛洒物检测以前做过哪些项目？"),
        ("垃圾桶满溢资产", "垃圾桶满溢检测有没有已有方案？"),
        ("河道漂浮物资产", "河道漂浮物识别是否有历史能力？"),
        ("烟头识别资产", "烟头识别或垃圾小目标检测有没有资产？"),
        ("工地裸土资产", "工地裸土覆盖检测的历史模型版本有哪些？"),
        ("门店客流资产", "门店客流统计有没有历史算法资产？"),
        ("电梯逆行资产", "扶梯逆行检测以前有没有做过？"),
        ("未戴手套资产", "未戴手套检测有没有历史标签和模型？"),
        ("车辆违停资产", "车辆违停检测有哪些已有模型版本？"),
        ("雨棚破损资产", "雨棚破损识别有没有历史方案？"),
        ("垃圾分类资产", "垃圾分类检测有没有沉淀过模型和标签？"),
        ("人员聚集资产", "人员聚集检测以前有哪些项目交付？"),
        ("摊贩识别资产", "流动摊贩识别有没有历史能力？"),
        ("安全门资产", "安全门未关闭检测有没有已有模型？"),
    ]
    out: list[dict[str, Any]] = []
    for idx, (title, query) in enumerate(queries, start=1):
        case = {
            "case_id": f"DPO-RAG-{idx:03d}",
            "title": title,
            "category": "historical_asset_qa",
            "user_query": query,
            "setup": {"has_image": False},
        }
        chosen = _decision(
            thought="用户在盘点已有能力、历史资产、模型版本或标签事实，应先检索知识库资产。",
            action=agent.TOOL_RAG_ANSWER,
            action_input={"query": query, "finish_after_tool": True},
        )
        rejected = _decision(
            thought="用户在询问现有能力边界和类似能力，适合使用迁移顾问生成能力评估报告。",
            action=agent.TOOL_MIGRATION_ADVISOR,
            action_input={
                "user_query": query,
                "use_image": False,
                "use_visual_probe": False,
                "finish_after_tool": True,
            },
        )
        out.append(
            _pair(
                case=case,
                error_type="asset_inventory_over_migration",
                chosen=chosen,
                rejected=rejected,
                rationale="纯历史资产盘点应走 rag_answer，不应直接生成迁移报告。",
            )
        )
    return out


def build_pairs(limit_per_type: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups = [
        _answerer_pairs()[:limit_per_type],
        _visual_detection_pairs()[:limit_per_type],
        _historical_asset_pairs()[:limit_per_type],
    ]
    pairs = [pair for group in groups for pair in group]
    cases = [
        {
            "case_id": pair["meta"]["case_id"],
            "title": pair["meta"]["title"],
            "category": pair["meta"]["category"],
            "user_query": pair["meta"]["user_query"],
            "setup": {"has_image": "DPO-VIS-" in pair["meta"]["case_id"]},
            "error_type": pair["meta"]["error_type"],
        }
        for pair in pairs
    ]
    return pairs, cases


def write_review(out_dir: Path, pairs: list[dict[str, Any]]) -> None:
    csv_path = out_dir / "planner_dpo_train_seed_review.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "review_status",
            "case_id",
            "category",
            "error_type",
            "chosen_action",
            "rejected_action",
            "user_query",
            "rationale",
            "reviewer_note",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            chosen = json.loads(pair["chosen"])
            rejected = json.loads(pair["rejected"])
            meta = pair["meta"]
            writer.writerow(
                {
                    "review_status": "todo",
                    "case_id": meta["case_id"],
                    "category": meta["category"],
                    "error_type": meta["error_type"],
                    "chosen_action": chosen.get("action"),
                    "rejected_action": rejected.get("action"),
                    "user_query": meta["user_query"],
                    "rationale": meta["rationale"],
                    "reviewer_note": "",
                }
            )

    lines = [
        "# Planner DPO Train Seed Review",
        "",
        "## Summary",
        "",
        f"- Pair count: {len(pairs)}",
        f"- By error_type: `{json.dumps(dict(Counter(p['meta']['error_type'] for p in pairs)), ensure_ascii=False)}`",
        "",
        "## Review Instructions",
        "",
        "- Mark each row as `approve`, `reject`, or `fix` in the CSV.",
        "- Approve only if the chosen routing decision is clearly better for single-step Planner routing.",
        "- Reject state-machine, slot-repair, or multi-step workflow cases; those belong in a different harness.",
        "",
        "## Index",
        "",
        "| # | Case | Error Type | Chosen > Rejected | Status |",
        "|---:|---|---|---|---|",
    ]
    for idx, pair in enumerate(pairs, start=1):
        chosen = json.loads(pair["chosen"])
        rejected = json.loads(pair["rejected"])
        meta = pair["meta"]
        lines.append(
            f"| {idx} | `{meta['case_id']}` | `{meta['error_type']}` | "
            f"`{chosen.get('action')}` > `{rejected.get('action')}` | TODO |"
        )
    lines.extend(["", "## Samples", ""])
    for idx, pair in enumerate(pairs[:12], start=1):
        meta = pair["meta"]
        lines.extend(
            [
                f"### {idx}. {meta['case_id']} - {meta['title']}",
                "",
                f"- Error type: `{meta['error_type']}`",
                f"- User query: {meta['user_query']}",
                f"- Rationale: {meta['rationale']}",
                "",
                "Chosen:",
                "",
                "```json",
                json.dumps(json.loads(pair["chosen"]), ensure_ascii=False, indent=2),
                "```",
                "",
                "Rejected:",
                "",
                "```json",
                json.dumps(json.loads(pair["rejected"]), ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    (out_dir / "planner_dpo_train_seed_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build synthetic planner DPO train seed.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory")
    parser.add_argument("--limit-per-type", type=int, default=40, help="Pairs per error type")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs, cases = build_pairs(limit_per_type=max(1, int(args.limit_per_type or 1)))
    write_jsonl(out_dir / "planner_dpo_train_seed_pairs.jsonl", pairs)
    write_json(out_dir / "planner_dpo_train_seed_cases.json", {"schema_version": "1.0", "cases": cases})
    write_review(out_dir, pairs)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pair_count": len(pairs),
        "by_error_type": dict(Counter(pair["meta"]["error_type"] for pair in pairs)),
        "out_dir": str(out_dir),
        "notes": [
            "Synthetic train seed for Planner single-step routing DPO.",
            "Excludes Adela clarification/state-machine and multi-step workflow cases.",
            "Human review is required before training.",
        ],
    }
    write_json(out_dir / "planner_dpo_train_seed_report.json", report)
    print(
        "Planner DPO train seed:",
        f"pairs={len(pairs)}",
        f"out={out_dir / 'planner_dpo_train_seed_pairs.jsonl'}",
        f"review={out_dir / 'planner_dpo_train_seed_review.md'}",
    )


if __name__ == "__main__":
    main()
