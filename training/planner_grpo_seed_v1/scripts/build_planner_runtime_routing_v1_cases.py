#!/usr/bin/env python3
"""Build entity-isolated cases for Planner decisions that run in the demo."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.planner_grpo_seed_v1.scripts.build_planner_sft_data import (  # noqa: E402
    build_rows,
)


DATASET_ID = "planner_runtime_routing_v1"
CASE_DIR = ROOT / "training/planner_grpo_seed_v1/cases"
STEP_DIR = ROOT / "training/planner_grpo_seed_v1/sft_data_runtime_routing_v1_chatml"
ENTITY_COUNTS = {"train": 30, "dev": 8, "test": 16}

PROJECTS = [
    "海岛光伏巡检",
    "医院物流通道",
    "船舶涂装车间",
    "机场行李分拣区",
    "地铁车辆段",
    "粮食筒仓",
    "校园消防通道",
    "港口冷藏集装箱区",
    "隧道机电设备间",
    "半导体洁净走廊",
    "山区索道站",
    "水产加工车间",
    "汽车电池仓库",
    "市政泵站",
    "高层建筑屋面",
    "铁路客运站台",
    "跨海大桥检修道",
    "医药配送仓",
]

TARGET_FIXTURES = [
    ("手持钓竿的人员", "examples/images/fisherman.jpg"),
    ("肩背包", "examples/images/person_with_bag.png"),
    ("垃圾清运车", "examples/images/trash_truck.jpg"),
    ("可见烟雾", "examples/images/smoke.jpg"),
    ("悬挂横幅", "examples/images/banner.jpg"),
]

FACTS = ["标签边界", "推荐版本", "部署约束", "历史精度", "训练数据规模", "支持平台"]

ACTION_DOMINANT_REWARD = {
    "json_valid": 0.02,
    "decision_type_valid": 0.03,
    "action_match": 0.55,
    "argument_match": 0.25,
    "finish_after_tool": 0.10,
    "no_forbidden_action": 0.05,
    "wrong_action_cap": 0.20,
}

TOOLS = [
    "rag_answer",
    "re_question",
    "answerer",
    "flux-image-generation",
    "qwen_detection",
    "rexomni_detection",
    "pipeline_eval",
    "migration_advisor",
    "adela_cli_eval",
]

SCENARIOS = (
    "private_lookup",
    "general_answer",
    "qwen_probe",
    "rex_probe",
    "pipeline_eval",
    "flux_generation",
    "migration_advisor",
    "adela_eval",
    "clarify_incomplete",
    "memory_end",
    "qwen_probe_then_migration",
    "rex_probe_then_migration",
    "qwen_probe_only_contrast",
)

SPLIT_STYLE = {
    "train": ["请处理", "帮我确认", "需要你核对", "这次想了解"],
    "dev": ["麻烦判断", "请直接处理", "我需要确认"],
    "test": ["协助核验", "这轮请判断", "请按要求完成"],
}


def _entities() -> dict[str, list[dict[str, str]]]:
    items: list[dict[str, str]] = []
    for index in range(sum(ENTITY_COUNTS.values())):
        project = PROJECTS[index % len(PROJECTS)]
        target, fixture = TARGET_FIXTURES[index % len(TARGET_FIXTURES)]
        items.append(
            {
                "project": f"{project}{index + 1:02d}号需求",
                "target": target,
                "fixture": fixture,
                "fact": FACTS[index % len(FACTS)],
                "model": f"KM_runtime_route_{index + 1:03d}_v2.model",
                "platform": (
                    "cuda11.0-trt7.1-fp16-T4"
                    if index % 2 == 0
                    else "cuda12.2-trt9.0-fp16-L4"
                ),
            }
        )
    random.Random(20260713).shuffle(items)
    result: dict[str, list[dict[str, str]]] = {}
    offset = 0
    for split in ("train", "dev", "test"):
        end = offset + ENTITY_COUNTS[split]
        result[split] = items[offset:end]
        offset = end
    return result


ENTITY_SPLITS = _entities()


def _setup(*, fixture: str = "", trajectories: list[dict[str, Any]] | None = None) -> dict:
    return {
        "has_image": bool(fixture),
        "image_fixture": fixture,
        "query_trajectories": trajectories or [],
    }


def _tool(
    action: str,
    *,
    required: dict[str, Any],
    contains: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    return {
        "decision_type": "tool",
        "action": action,
        "required_args": required,
        "arg_contains": contains or {},
    }


def _forbidden(*allowed: str) -> list[str]:
    return [action for action in TOOLS if action not in set(allowed)]


def _render_query(
    split: str,
    scenario: str,
    *,
    style: str,
    project: str,
    target: str,
    fact: str,
    model: str,
    platform: str,
    model_text: str = "",
    metric: str = "",
) -> str:
    templates = {
        "private_lookup": {
            "train": "{style}{project}现有内部模型的{fact}，按公司资料给出依据。",
            "dev": "企业资料里哪份记录说明了{project}的{fact}？请检索后回答。",
            "test": "不要按通用常识推断；从私有知识中查明{project}在{fact}方面的记录。",
        },
        "general_answer": {
            "train": "{style}{project}的一般误报分析步骤，不查内部知识库。",
            "dev": "不访问公司资产时，怎样为{project}系统梳理误报原因？",
            "test": "仅按公开通用方法给出{project}的错误分析框架。",
        },
        "probe": {
            "train": "{style}这张图里是否有{target}，明确用{model_text}快速检测；检测完就结束，不需要完整评测报告。",
            "dev": "上传样例只做一次{model_text}单图探针，目标是{target}；不要扩增数据或写评测报告。",
            "test": "请让{model_text}开放集模型直接检查附件中的{target}，返回框后本轮收口。",
        },
        "probe_only": {
            "train": "{style}这张图里是否有{target}，明确用{model_text}快速检测；只返回本次探针结果，不要继续做迁移方案。",
            "dev": "附件只需要{model_text}检查{target}；即便结果不稳定也先结束，后续能力迁移本轮不讨论。",
            "test": "仅执行一次{model_text}的{target}探针并停止，不要追加现有能力复用或迁移分析。",
        },
        "pipeline_eval": {
            "train": "{style}以这张{target}参考图扩增样本，对比 Qwen 和 Rex-Omni，输出误检漏检与效果评估报告。",
            "dev": "把附件作为种子补生成{target}场景，同时跑 Qwen 与 Rex-Omni，最终汇总模型对比和漏检分析。",
            "test": "需要一条端到端{target}评测流水线：样本扩增、双模型标注、精度结论都要包含。",
        },
        "flux_generation": {
            "train": "{style}生成一张{project}中出现{target}的合成图片，只生成一张，不做检测。",
            "dev": "不要运行识别模型，只创建一幅{project}内含{target}的合成图。",
            "test": "调用生图能力制作单张{target}在{project}中的图片，生成后立即结束。",
        },
        "migration_advisor": {
            "train": "{style}{project}新增{target}需求能否从现有能力迁移，请评估数据量、成本、风险和能力边界。",
            "dev": "针对{project}的{target}新需求，论证已有模型复用边界，并估算补数与工程代价。",
            "test": "为{project}做{target}能力迁移方案，必须包含相似资产、风险、成本和预期效果。",
        },
        "adela_eval": {
            "train": "{style}{model}部署到{platform}后的{metric}，请在 Adela 上实际评测。",
            "dev": "在 Adela 执行{model}到{platform}的部署 benchmark，本次关注{metric}。",
            "test": "平台已定为{platform}，请用 Adela 实测模型{model}的{metric}，不要只查文档。",
        },
        "clarify_incomplete": {
            "train": "关于{project}，{style}评估一下这个模型在目标机器上的效果。",
            "dev": "{project}这边想知道刚才那个模型部署后到底行不行，先帮我跑一下。",
            "test": "给{project}测一下它在那套环境里的表现，具体配置沿用我没说清的那项。",
        },
        "memory_end": {
            "train": "沿用刚才{project}的{fact}结论即可，不要再次调用工具。",
            "dev": "前一轮已有{project}关于{fact}的充分证据，请从当前记忆直接收口。",
            "test": "别重新检索；复用上下文中已经确认的{project}{fact}信息完成回答。",
        },
        "probe_then_migration": {
            "train": "{style}先用{model_text}看这张图有没有{target}；探针完成后无论框数多少，继续给出{project}的低成本迁移方案，不要提前结束。",
            "dev": "任务分两步：附件中的{target}先交给{model_text}做样例探针，随后必须继续分析{project}的能力复用方案。",
            "test": "先取得{model_text}对{target}的单图结果，把它仅作为样例证据；下一步再完成{project}迁移评估。",
        },
    }
    return templates[scenario][split].format(
        style=style,
        project=project,
        target=target,
        fact=fact,
        model=model,
        platform=platform,
        model_text=model_text,
        metric=metric,
    )


def _base(
    *,
    split: str,
    entity_index: int,
    scenario: str,
    query: str,
    setup: dict,
    decisions: list[dict[str, Any]],
    observations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    entity_id = f"rrv1_{split}_entity_{entity_index + 1:03d}"
    return {
        "case_id": f"RRV1-{split.upper()}-{scenario.upper()}-{entity_index + 1:03d}",
        "split": split,
        "entity_id": entity_id,
        "group_id": entity_id,
        "template_id": f"{scenario}_{split}_{entity_index % len(SPLIT_STYLE[split]) + 1}",
        "scenario_id": scenario,
        "category": scenario,
        "user_query": query,
        "setup": setup,
        "expected_decisions": decisions,
        "mock_observations": observations or [],
        "forbidden_actions": _forbidden(
            *[
                str(item.get("action") or "")
                for item in decisions
                if item.get("decision_type") == "tool"
            ]
        ),
        "reward_spec": ACTION_DOMINANT_REWARD,
        "provenance_class": "deidentified_synthetic_from_demo_patterns",
    }


def make_case(
    *, split: str, entity_index: int, entity: dict[str, str], scenario: str
) -> dict[str, Any]:
    style = SPLIT_STYLE[split][entity_index % len(SPLIT_STYLE[split])]
    project = entity["project"]
    target = entity["target"]
    fixture = entity["fixture"]
    fact = entity["fact"]
    model = entity["model"]
    platform = entity["platform"]

    if scenario == "private_lookup":
        query = _render_query(
            split,
            scenario,
            style=style,
            project=project,
            target=target,
            fact=fact,
            model=model,
            platform=platform,
        )
        decisions = [
            _tool(
                "rag_answer",
                required={"finish_after_tool": True},
                contains={"query": [project, fact]},
            )
        ]
        return _base(
            split=split,
            entity_index=entity_index,
            scenario=scenario,
            query=query,
            setup=_setup(),
            decisions=decisions,
        )

    if scenario == "general_answer":
        query = _render_query(
            split,
            scenario,
            style=style,
            project=project,
            target=target,
            fact=fact,
            model=model,
            platform=platform,
        )
        decisions = [
            _tool(
                "answerer",
                required={"mode": "direct", "finish_after_tool": True},
            )
        ]
        return _base(
            split=split,
            entity_index=entity_index,
            scenario=scenario,
            query=query,
            setup=_setup(),
            decisions=decisions,
        )

    if scenario in {"qwen_probe", "rex_probe", "qwen_probe_only_contrast"}:
        action = "rexomni_detection" if scenario == "rex_probe" else "qwen_detection"
        model_text = "Rex-Omni" if action == "rexomni_detection" else "Qwen"
        query = _render_query(
            split,
            "probe_only" if scenario == "qwen_probe_only_contrast" else "probe",
            style=style,
            project=project,
            target=target,
            fact=fact,
            model=model,
            platform=platform,
            model_text=model_text,
        )
        decisions = [
            _tool(
                action,
                required={"finish_after_tool": True},
                contains={"label": [target]},
            )
        ]
        return _base(
            split=split,
            entity_index=entity_index,
            scenario=scenario,
            query=query,
            setup=_setup(fixture=fixture),
            decisions=decisions,
        )

    if scenario == "pipeline_eval":
        query = _render_query(
            split,
            scenario,
            style=style,
            project=project,
            target=target,
            fact=fact,
            model=model,
            platform=platform,
        )
        decisions = [
            _tool(
                "pipeline_eval",
                required={"finish_after_tool": True},
                contains={"task_text": [target, "评估"]},
            )
        ]
        return _base(
            split=split,
            entity_index=entity_index,
            scenario=scenario,
            query=query,
            setup=_setup(fixture=fixture),
            decisions=decisions,
        )

    if scenario == "flux_generation":
        query = _render_query(
            split,
            scenario,
            style=style,
            project=project,
            target=target,
            fact=fact,
            model=model,
            platform=platform,
        )
        decisions = [
            _tool(
                "flux-image-generation",
                required={
                    "source_image_required": False,
                    "num_images": 1,
                    "finish_after_tool": True,
                },
                contains={"task_text": [project, target]},
            )
        ]
        return _base(
            split=split,
            entity_index=entity_index,
            scenario=scenario,
            query=query,
            setup=_setup(),
            decisions=decisions,
        )

    if scenario == "migration_advisor":
        query = _render_query(
            split,
            scenario,
            style=style,
            project=project,
            target=target,
            fact=fact,
            model=model,
            platform=platform,
        )
        decisions = [
            _tool(
                "migration_advisor",
                required={
                    "use_image": False,
                    "use_visual_probe": False,
                    "finish_after_tool": True,
                },
                contains={"user_query": [project, target, "迁移"]},
            )
        ]
        return _base(
            split=split,
            entity_index=entity_index,
            scenario=scenario,
            query=query,
            setup=_setup(),
            decisions=decisions,
        )

    if scenario == "adela_eval":
        eval_type = entity_index % 2
        metric = "精度" if eval_type == 0 else "性能"
        query = _render_query(
            split,
            scenario,
            style=style,
            project=project,
            target=target,
            fact=fact,
            model=model,
            platform=platform,
            metric=metric,
        )
        decisions = [
            _tool(
                "adela_cli_eval",
                required={
                    "model_name": model,
                    "platform": platform,
                    "eval_type": eval_type,
                    "finish_after_tool": True,
                },
            )
        ]
        return _base(
            split=split,
            entity_index=entity_index,
            scenario=scenario,
            query=query,
            setup=_setup(),
            decisions=decisions,
        )

    if scenario == "clarify_incomplete":
        query = _render_query(
            split,
            scenario,
            style=style,
            project=project,
            target=target,
            fact=fact,
            model=model,
            platform=platform,
        )
        decisions = [
            {
                "decision_type": "clarify",
                "required_args": {},
                "arg_contains": {},
            }
        ]
        return _base(
            split=split,
            entity_index=entity_index,
            scenario=scenario,
            query=query,
            setup=_setup(),
            decisions=decisions,
        )

    if scenario == "memory_end":
        query = _render_query(
            split,
            scenario,
            style=style,
            project=project,
            target=target,
            fact=fact,
            model=model,
            platform=platform,
        )
        trajectories = [
            {
                "query_id": f"prior_rrv1_{split}_{entity_index + 1:03d}",
                "query": f"查询{project}的{fact}。",
                "result_summary": f"内部证据已经充分给出{project}的{fact}，可直接复用。",
                "steps": [],
            }
        ]
        decisions = [
            {
                "decision_type": "end",
                "required_args": {"end_reason": "memory_hit"},
                "arg_contains": {},
            }
        ]
        return _base(
            split=split,
            entity_index=entity_index,
            scenario=scenario,
            query=query,
            setup=_setup(trajectories=trajectories),
            decisions=decisions,
        )

    action = (
        "rexomni_detection"
        if scenario == "rex_probe_then_migration"
        else "qwen_detection"
    )
    model_text = "Rex-Omni" if action == "rexomni_detection" else "Qwen"
    query = _render_query(
        split,
        "probe_then_migration",
        style=style,
        project=project,
        target=target,
        fact=fact,
        model=model,
        platform=platform,
        model_text=model_text,
    )
    decisions = [
        _tool(
            action,
            required={"finish_after_tool": False},
            contains={"label": [target]},
        ),
        _tool(
            "migration_advisor",
            required={
                "use_image": True,
                "use_visual_probe": True,
                "finish_after_tool": True,
            },
            contains={"user_query": [project, target, "迁移"]},
        ),
    ]
    observation = {
        "success": True,
        "summary": (
            f"{model_text} 样例探针已完成：返回候选框，但不同表述下结果有波动；"
            "该输出仅是视觉样例证据，不代表已有专用模型或迁移结论。"
        ),
        "num_boxes": entity_index % 3,
        "elapsed_ms": 1800 + entity_index * 37,
    }
    return _base(
        split=split,
        entity_index=entity_index,
        scenario=scenario,
        query=query,
        setup=_setup(fixture=fixture),
        decisions=decisions,
        observations=[{"after_step": 1, "observation": observation}],
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def summarize(cases: list[dict[str, Any]], steps: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cases": len(cases),
        "steps": len(steps),
        "entity_groups": len({row["entity_id"] for row in cases}),
        "categories": dict(sorted(Counter(row["category"] for row in cases).items())),
        "expected_actions": dict(
            sorted(
                Counter(
                    (
                        step.get("decision_type")
                        if step.get("decision_type") in {"clarify", "end"}
                        else step.get("action")
                    )
                    for case in cases
                    for step in case["expected_decisions"]
                ).items()
            )
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, default=CASE_DIR)
    parser.add_argument("--step-dir", type=Path, default=STEP_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report: dict[str, Any] = {
        "dataset_id": DATASET_ID,
        "provenance": (
            "Synthetic and deidentified; scenario frequencies and phrasing classes are "
            "derived from aggregate demo sessions, not copied user turns."
        ),
        "splits": {},
    }
    for split in ("train", "dev", "test"):
        cases = [
            make_case(
                split=split,
                entity_index=entity_index,
                entity=entity,
                scenario=scenario,
            )
            for entity_index, entity in enumerate(ENTITY_SPLITS[split])
            for scenario in SCENARIOS
        ]
        steps = build_rows(
            cases,
            indent=-1,
            prompt_format="qwen_chatml",
            append_im_end=True,
        )
        write_jsonl(args.case_dir / f"{DATASET_ID}_{split}_cases.jsonl", cases)
        write_jsonl(args.step_dir / f"{split}.jsonl", steps)
        report["splits"][split] = summarize(cases, steps)
    args.step_dir.mkdir(parents=True, exist_ok=True)
    (args.step_dir / "metadata.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
