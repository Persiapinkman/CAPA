#!/usr/bin/env python3
"""Build entity- and template-separated stateful retrieval Planner cases."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = ROOT / "training" / "planner_grpo_seed_v1" / "cases"


ENTITY_SPLITS = {
    "train": [
        "安全绳佩戴检测",
        "河道漂浮物识别",
        "后厨厨师帽佩戴检测",
        "垃圾车识别",
        "夜间黑猫检测",
        "烟火检测",
        "工地反光衣识别",
        "输电线异物检测",
        "渔政钓鱼人员识别",
        "商场客流计数",
        "厂区叉车识别",
        "道路抛洒物检测",
        "消防通道占用检测",
        "车间护目镜佩戴检测",
        "河岸非法采砂识别",
        "停车场车位占用检测",
        "站台越线检测",
        "油库明火检测",
        "园区周界入侵检测",
        "传送带煤块堵塞检测",
        "高空抛物识别",
        "水库漂船检测",
        "实验室白大褂穿戴检测",
        "道路施工围挡检测",
    ],
    "dev": [
        "矿区安全帽检测",
        "港口集装箱编号识别",
        "园区违停检测",
        "车间防护手套识别",
        "水面油污识别",
        "机场跑道异物检测",
        "冷库门未关闭检测",
        "桥梁裂缝识别",
    ],
    "test": [
        "仓库托盘检测",
        "隧道积水识别",
        "光伏板缺陷检测",
        "农田秸秆焚烧监测",
        "校园翻墙识别",
        "轨道侵限检测",
        "码头救生衣穿戴检测",
        "果园病果识别",
        "电梯门遮挡检测",
        "园区无人值守岗检测",
        "变电站小动物入侵检测",
        "建筑外墙脱落风险识别",
        "地铁扶梯逆行检测",
        "粮仓虫害识别",
        "海岸垃圾堆积检测",
        "医院口罩佩戴检测",
        "物流包裹破损识别",
        "景区拥堵检测",
        "船舶靠泊状态识别",
        "养殖场动物计数",
        "森林倒木检测",
        "充电桩占位检测",
        "化工区液体泄漏识别",
        "机房柜门开启检测",
    ],
}


FACTS = [
    ("model_version", "模型版本", "CAPA-Route-3.2"),
    ("label_definition", "标签定义", "目标主体完整可见且满足业务区域约束"),
    ("acceptance_threshold", "验收阈值", "召回率不低于 0.90"),
    ("latest_benchmark", "最近一次评测结果", "验证集 F1 为 0.87"),
    ("dataset_size", "数据集规模", "训练集 12,400 张、验证集 1,600 张"),
    ("deployment_platform", "部署平台", "cuda11.8-trt8.6-fp16-T4"),
]


TEMPLATES = {
    "train": {
        "coref": [
            "这个项目的{fact}再从内部资料查一下。",
            "它的{fact}是什么？请接着查公司记录。",
            "上述能力对应的{fact}能在知识库里找到吗？",
        ],
        "miss": [
            "请查{entity}的{fact}；如果第一轮没有命中，做一次小步改写后再查，最多两轮。",
            "从内部资料找{entity}的{fact}。首次检索为空时先改写关键词，再检索一次。",
            "需要{entity}的{fact}，若直接查询失败，不要结束，改写问题后重试一次。",
        ],
        "memory": [
            "刚才关于这个项目的结果已经足够，请直接沿用，不要再次检索。",
            "上一条内部结论可以回答当前问题，直接基于已有结果收口。",
            "不用重复查了，请使用刚得到的{fact}结果回答。",
        ],
        "direct": [
            "请从公司内部资料查询{entity}的{fact}。",
            "知识库里记录的{entity}{fact}是什么？",
            "帮我检索一下{entity}当前的{fact}。",
        ],
        "general": [
            "不查公司资料，通用的{entity}错误分析报告通常应包含哪些部分？",
            "请直接给一个{entity}项目的通用验收清单模板，不需要内部信息。",
            "从通用方法论说明怎样分析{entity}误报，不调用知识库。",
        ],
    },
    "dev": {
        "coref": [
            "前面说的那项能力，其{fact}请补充检索。",
            "该项目对应的{fact}还没回答，请结合上一问继续查。",
        ],
        "miss": [
            "内部查询{entity}{fact}，首轮找不到就先缩小改写范围，再做第二轮检索。",
            "查{entity}的{fact}；第一次无结果时必须改写一次，随后重查并结束。",
        ],
        "memory": [
            "现有摘要已经给出了答案，请不要再调用工具，直接交给回答环节。",
            "沿用上一轮得到的{fact}即可，无需重新搜索。",
        ],
        "direct": [
            "内部文档中{entity}采用什么{fact}？",
            "只检索一次，返回{entity}的{fact}记录。",
        ],
        "general": [
            "不涉及内部事实，解释{entity}项目中 precision 和 recall 应如何取舍。",
            "直接说明{entity}数据标注质检的一般步骤，不要检索。",
        ],
    },
    "test": {
        "coref": [
            "就上一项业务而言，请把它的{fact}从私有资料中补齐。",
            "关于刚提到的能力，继续核实其{fact}，不要丢失实体。",
            "请解析这里的指代，再查该方案的{fact}。",
        ],
        "miss": [
            "检索{entity}的{fact}。若首查未返回证据，先最小化改写，再执行最后一次检索。",
            "需要核对{entity}{fact}；空结果不能直接结束，应改写后重试一轮。",
            "先查{entity}的{fact}，未命中则改写查询并再查，禁止超过两次检索。",
        ],
        "memory": [
            "答案已在上轮结果中，停止规划新工具并使用现有证据。",
            "刚才的摘要足够支持回答，现在直接收口。",
            "请复用已有的{fact}结论，不进行重复检索。",
        ],
        "direct": [
            "从私有知识库确认{entity}所用的{fact}。",
            "查询企业记录：{entity}的{fact}。",
            "请检索内部资产里{entity}对应的{fact}字段。",
        ],
        "general": [
            "仅按通用经验回答：如何为{entity}制定错误分类体系？",
            "无需企业数据，给出{entity}模型上线前的一般检查项。",
            "从公开常识角度解释{entity}数据集为什么要划分训练、验证和测试集。",
        ],
    },
}


BASE_REWARD = {
    "json_valid": 0.05,
    "decision_type_valid": 0.10,
    "action_match": 0.35,
    "argument_match": 0.25,
    "finish_after_tool": 0.15,
    "no_forbidden_action": 0.10,
}
PROCESS_REWARD = {
    **BASE_REWARD,
    "no_premature_stop": 0.15,
    "no_repeated_tool": 0.15,
    "final_tool_finish": 0.10,
}
FORBIDDEN_RETRIEVAL = [
    "answerer",
    "qwen_detection",
    "rexomni_detection",
    "pipeline_eval",
    "migration_advisor",
    "flux-image-generation",
    "adela_cli_eval",
]


def prior_trajectory(case_id: str, query: str, summary: str) -> list[dict[str, Any]]:
    return [
        {
            "query_id": f"prior_{case_id.lower()}",
            "query": query,
            "result_summary": summary,
            "steps": [],
        }
    ]


def setup(trajectories: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "has_image": False,
        "image_fixture": "",
        "query_trajectories": trajectories or [],
    }


def re_question_step(entity: str, fact: str, *, reason: str, round_index: int) -> dict[str, Any]:
    return {
        "decision_type": "tool",
        "action": "re_question",
        "required_args": {
            "rewrite_reason": reason,
            "retrieval_round": round_index,
            "finish_after_tool": False,
        },
        "arg_contains": {"query": [entity, fact], "context_hint": [entity]},
    }


def rag_step(entity: str, fact: str, *, finish: bool) -> dict[str, Any]:
    return {
        "decision_type": "tool",
        "action": "rag_answer",
        "required_args": {"finish_after_tool": finish},
        "arg_contains": {"query": [entity, fact]},
    }


def make_case(
    *,
    split: str,
    entity_index: int,
    entity: str,
    scenario: str,
    template_index: int,
) -> dict[str, Any]:
    fact_id, fact, value = FACTS[entity_index % len(FACTS)]
    template_options = TEMPLATES[split][scenario]
    template = template_options[template_index % len(template_options)]
    query = template.format(entity=entity, fact=fact)
    scenario_code = {
        "coref": "COREF",
        "miss": "MISS",
        "memory": "MEMORY",
        "direct": "DIRECT",
        "general": "GENERAL",
    }[scenario]
    case_id = f"SRV1-{split.upper()}-{scenario_code}-{entity_index + 1:03d}"
    common = {
        "case_id": case_id,
        "split": split,
        "entity_id": f"{split}_entity_{entity_index + 1:03d}",
        "group_id": f"{split}_entity_{entity_index + 1:03d}",
        "template_id": f"{scenario}_{split}_{template_index % len(template_options) + 1}",
        "scenario_id": scenario,
        "fact_id": fact_id,
        "user_query": query,
    }

    if scenario == "coref":
        prior = prior_trajectory(
            case_id,
            f"{entity}是否有历史能力？",
            f"已确认问题指向内部的{entity}项目，但尚未查询{fact}。",
        )
        return {
            **common,
            "category": "coref_rewrite_then_rag",
            "setup": setup(prior),
            "expected_decisions": [
                re_question_step(entity, fact, reason="coref_resolve", round_index=1),
                rag_step(entity, fact, finish=True),
            ],
            "mock_observations": [
                {
                    "after_step": 1,
                    "observation": {
                        "success": True,
                        "summary": f"已将指代改写为：查询{entity}的{fact}。",
                    },
                }
            ],
            "forbidden_actions": FORBIDDEN_RETRIEVAL,
            "reward_spec": PROCESS_REWARD,
        }

    if scenario == "miss":
        return {
            **common,
            "category": "rag_miss_rewrite_then_rag",
            "setup": setup(),
            "expected_decisions": [
                rag_step(entity, fact, finish=False),
                re_question_step(entity, fact, reason="rag_miss", round_index=2),
                rag_step(entity, fact, finish=True),
            ],
            "mock_observations": [
                {
                    "after_step": 1,
                    "observation": {
                        "success": False,
                        "summary": f"首次检索未命中{entity}的{fact}，需要小步改写。",
                    },
                },
                {
                    "after_step": 2,
                    "observation": {
                        "success": True,
                        "summary": f"改写完成：在内部资产中检索{entity}的{fact}字段。",
                    },
                },
            ],
            "forbidden_actions": FORBIDDEN_RETRIEVAL,
            "reward_spec": PROCESS_REWARD,
        }

    if scenario == "memory":
        prior = prior_trajectory(
            case_id,
            f"请查询{entity}的{fact}。",
            f"内部记录显示，{entity}的{fact}为：{value}。证据充分，可直接回答。",
        )
        return {
            **common,
            "category": "memory_hit_end",
            "setup": setup(prior),
            "expected_decisions": [
                {
                    "decision_type": "end",
                    "required_args": {"end_reason": "memory_hit"},
                    "arg_contains": {},
                }
            ],
            "forbidden_actions": [
                "rag_answer",
                "re_question",
                "answerer",
                "qwen_detection",
                "pipeline_eval",
                "migration_advisor",
            ],
            "reward_spec": BASE_REWARD,
        }

    if scenario == "direct":
        return {
            **common,
            "category": "direct_rag_guardrail",
            "setup": setup(),
            "expected_decisions": [rag_step(entity, fact, finish=True)],
            "forbidden_actions": [
                "re_question",
                "answerer",
                "qwen_detection",
                "pipeline_eval",
                "migration_advisor",
            ],
            "reward_spec": BASE_REWARD,
        }

    return {
        **common,
        "category": "general_answer_guardrail",
        "setup": setup(),
        "expected_decisions": [
            {
                "decision_type": "tool",
                "action": "answerer",
                "required_args": {"mode": "direct", "finish_after_tool": True},
                "arg_contains": {},
            }
        ],
        "forbidden_actions": [
            "rag_answer",
            "re_question",
            "qwen_detection",
            "pipeline_eval",
            "migration_advisor",
        ],
        "reward_spec": BASE_REWARD,
    }


def build_split(split: str) -> list[dict[str, Any]]:
    scenarios = ("coref", "miss", "memory", "direct", "general")
    rows: list[dict[str, Any]] = []
    for entity_index, entity in enumerate(ENTITY_SPLITS[split]):
        for scenario_index, scenario in enumerate(scenarios):
            rows.append(
                make_case(
                    split=split,
                    entity_index=entity_index,
                    entity=entity,
                    scenario=scenario,
                    template_index=entity_index + scenario_index,
                )
            )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cases": len(rows),
        "steps": sum(len(row["expected_decisions"]) for row in rows),
        "entities": len({row["entity_id"] for row in rows}),
        "categories": dict(sorted(Counter(row["category"] for row in rows).items())),
        "templates": len({row["template_id"] for row in rows}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    all_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {"dataset_id": "planner_stateful_retrieval_v1", "splits": {}}
    for split in ("train", "dev", "test"):
        rows = build_split(split)
        all_rows.extend(rows)
        write_jsonl(output_dir / f"planner_stateful_retrieval_v1_{split}_cases.jsonl", rows)
        report["splits"][split] = summarize(rows)
    write_jsonl(output_dir / "planner_stateful_retrieval_v1_all_cases.jsonl", all_rows)
    report["all"] = summarize(all_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
