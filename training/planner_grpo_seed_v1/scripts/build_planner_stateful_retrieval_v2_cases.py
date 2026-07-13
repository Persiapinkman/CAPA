#!/usr/bin/env python3
"""Build the expanded, five-step stateful retrieval Planner benchmark."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = ROOT / "training" / "planner_grpo_seed_v1" / "cases"


SITES = [
    "钢铁厂原料区",
    "海上风电场",
    "城市地下管廊",
    "港区散货码头",
    "食品冷链仓库",
    "高速公路服务区",
    "水泥厂装卸区",
    "大型商业综合体",
    "山区输电走廊",
    "污水处理厂",
    "铁路货运编组站",
    "锂电池生产车间",
    "城市生活垃圾站",
    "天然气调压站",
    "精密仪器实验室",
    "沿海防波堤",
    "自动化立体仓库",
    "露天矿运输道路",
    "数据中心机房",
    "农业温室大棚",
    "航空维修机库",
    "大型会展中心",
    "水电站设备层",
    "化工园区罐区",
]

CAPABILITIES = [
    "安全门开启检测",
    "人员跌倒识别",
    "液体泄漏检测",
    "设备表面锈蚀识别",
    "作业区域越界检测",
]

ENTITY_COUNTS = {"train": 72, "dev": 16, "test": 32}


def build_entity_splits() -> dict[str, list[str]]:
    entities = [f"{site}{capability}" for site in SITES for capability in CAPABILITIES]
    random.Random(20260712).shuffle(entities)
    splits: dict[str, list[str]] = {}
    start = 0
    for split in ("train", "dev", "test"):
        end = start + ENTITY_COUNTS[split]
        splits[split] = entities[start:end]
        start = end
    if start != len(entities) or len(set(entities)) != len(entities):
        raise RuntimeError("entity construction must create 120 unique, fully assigned entities")
    return splits


ENTITY_SPLITS = build_entity_splits()

FACTS = [
    ("model_version", "模型版本", "CAPA-State-4.1"),
    ("label_definition", "标签定义", "目标在业务区域内连续可见且边界完整"),
    ("acceptance_threshold", "验收阈值", "关键类别召回率不低于 0.92"),
    ("latest_benchmark", "最近一次评测结果", "独立验证集 F1 为 0.89"),
    ("dataset_size", "数据集规模", "训练集 18,600 张、验证集 2,400 张"),
    ("deployment_platform", "部署平台", "cuda12.2-trt9.0-fp16-L4"),
]

TEMPLATES = {
    "train": {
        "single_miss": [
            "查询{entity}的{fact}；首轮为空时先改写，再检索一次并结束。",
            "从内部资料找{entity}{fact}，第一次没证据就缩小问题后重试一轮。",
            "核对{entity}的{fact}。若直接查询未命中，做一次最小改写并再次查询。",
            "请检索{entity}{fact}，允许一次空结果恢复：改写后完成第二次检索。",
        ],
        "double_miss": [
            "查询{entity}的{fact}；前两轮若为空，每次都先小步改写，第三轮检索后结束。",
            "从内部资料核验{entity}{fact}。最多检索三轮，连续未命中时逐轮收窄关键词。",
            "查找{entity}的{fact}；第一、第二次无证据都不能结束，应分别改写后继续检索。",
            "执行{entity}{fact}的三级恢复检索：每次空结果后改写，第三次查询是最终轮。",
        ],
        "synthesis": [
            "先检索{entity}的{fact}，再依据检索证据解释其业务含义；检索只是中间步骤。",
            "需要内部的{entity}{fact}并给出证据化说明，请先查资料再交给回答环节综合。",
            "查到{entity}的{fact}后还要形成简短解读，不要在检索工具返回时直接结束。",
            "请先取得{entity}{fact}的私有证据，然后基于该证据完成说明。",
        ],
        "coref": [
            "把刚才那个项目的{fact}继续从内部资料查清楚。",
            "该能力对应的{fact}还没有核实，请结合上一问补全实体后查询。",
            "它的{fact}是什么？先解析前文指代，再查企业记录。",
            "请沿用上文项目名称，补查其{fact}。",
        ],
        "memory": [
            "现有摘要已经充分回答{fact}，请直接收口，不要重复检索。",
            "沿用上一轮内部结果即可，无需再次调用工具。",
            "刚得到的{fact}证据足够，请停止规划新工具。",
            "不要重新查询，使用当前记忆中的{fact}结论回答。",
        ],
        "direct": [
            "从公司知识库直接查询{entity}的{fact}。",
            "请检索一次{entity}{fact}并返回结果。",
            "内部资产中记录的{entity}{fact}是什么？",
            "帮我核对企业资料里的{entity}{fact}。",
        ],
        "general": [
            "不查内部资料，说明{entity}项目通常怎样做错误分析。",
            "按通用方法给出{entity}数据质检清单，不使用企业知识库。",
            "直接解释{entity}场景下 precision 与 recall 的一般取舍。",
            "仅用通用知识说明如何验收{entity}项目。",
        ],
    },
    "dev": {
        "single_miss": [
            "内部核查{entity}{fact}；若第一轮没有返回内容，先重写查询再做最后一次检索。",
            "找出{entity}的{fact}，首查落空时用更具体表述重试一次。",
            "查询{entity}{fact}，允许一次改写恢复，但不能在空结果处停止。",
        ],
        "double_miss": [
            "核验{entity}{fact}：前两次检索为空时均需改写，第三次检索后停止。",
            "对{entity}的{fact}做最多三轮内部查询，两次空结果分别触发一次范围收窄。",
            "查{entity}{fact}，连续两轮无证据时逐次改写并完成第三轮查询。",
        ],
        "synthesis": [
            "先取回{entity}{fact}的内部证据，再根据证据做解释，不能在检索后直接结束。",
            "检索{entity}的{fact}只是第一步，随后还需交给回答模块完成归纳。",
            "请查到{entity}{fact}后继续形成证据化说明。",
        ],
        "coref": [
            "前述项目的{fact}请从私有资料中继续核实。",
            "上一问提到的能力，其{fact}还需要补查。",
            "请先还原这里的指代，再检索该项目的{fact}。",
        ],
        "memory": [
            "已有证据足以回答，请复用记忆并结束工具规划。",
            "当前摘要已包含{fact}，不要发起新的检索。",
            "直接使用上一轮{fact}结果收口。",
        ],
        "direct": [
            "查询私有知识库中{entity}对应的{fact}。",
            "内部文档记录的{entity}{fact}是什么？",
            "请直接检索{entity}的{fact}字段。",
        ],
        "general": [
            "无需内部数据，概述{entity}模型的一般上线检查。",
            "仅按通用经验说明{entity}样本应如何分层。",
            "不要检索，解释{entity}误报分析的常见步骤。",
        ],
    },
    "test": {
        "single_miss": [
            "核对企业记录里的{entity}{fact}；首轮空白时先改述问题，再完成一次重查。",
            "检索{entity}的{fact}，一次未命中不能收口，应改写后进行最终查询。",
            "请查{entity}{fact}，若首次没有证据，用更窄查询再试一轮。",
        ],
        "double_miss": [
            "核查{entity}{fact}，前两轮空结果各触发一次改写，最后执行第三轮检索。",
            "对{entity}的{fact}最多查三次；第一、第二次未命中都要收窄表达后继续。",
            "连续恢复检索{entity}{fact}：两次空证据后分别改写，第三次查询即止。",
        ],
        "synthesis": [
            "取得{entity}{fact}的内部记录后，还要依据证据做归纳说明。",
            "先查{entity}{fact}，检索返回不是终点，随后用证据完成解释。",
            "请先获取{entity}{fact}私有证据，再交由回答步骤综合。",
        ],
        "coref": [
            "刚提及的项目，其{fact}请解析指代后从内部资料补齐。",
            "关于上一项能力，继续核实它的{fact}。",
            "保留前文实体，查询该方案对应的{fact}。",
        ],
        "memory": [
            "答案已经存在于当前记忆，请停止调用新工具。",
            "复用刚才的{fact}证据即可，不再搜索。",
            "现有摘要充分，直接结束规划并交给回答环节。",
        ],
        "direct": [
            "从企业私有资料确认{entity}的{fact}。",
            "检索内部资产：{entity}{fact}。",
            "查一下知识库中{entity}对应的{fact}。",
        ],
        "general": [
            "只用通用知识说明{entity}项目的验收原则。",
            "不访问企业资料，给出{entity}错误分类框架。",
            "从一般方法解释怎样复核{entity}数据标注。",
        ],
    },
}

ACTION_DOMINANT_REWARD = {
    "json_valid": 0.02,
    "decision_type_valid": 0.03,
    "action_match": 0.75,
    "argument_match": 0.10,
    "finish_after_tool": 0.05,
    "no_forbidden_action": 0.05,
    "wrong_action_cap": 0.20,
}

UNRELATED_ACTIONS = [
    "qwen_detection",
    "rexomni_detection",
    "pipeline_eval",
    "migration_advisor",
    "flux-image-generation",
    "adela_cli_eval",
]


def setup(trajectories: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"has_image": False, "image_fixture": "", "query_trajectories": trajectories or []}


def prior_trajectory(case_id: str, query: str, summary: str) -> list[dict[str, Any]]:
    return [
        {
            "query_id": f"prior_{case_id.lower()}",
            "query": query,
            "result_summary": summary,
            "steps": [],
        }
    ]


def rag_step(entity: str, fact: str, *, finish: bool) -> dict[str, Any]:
    return {
        "decision_type": "tool",
        "action": "rag_answer",
        "required_args": {"finish_after_tool": finish},
        "arg_contains": {"query": [entity, fact]},
    }


def rewrite_step(entity: str, fact: str, *, reason: str, round_index: int) -> dict[str, Any]:
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


def answerer_step(*, mode: str) -> dict[str, Any]:
    return {
        "decision_type": "tool",
        "action": "answerer",
        "required_args": {"mode": mode, "finish_after_tool": True},
        "arg_contains": {},
    }


def observation(after_step: int, *, success: bool, summary: str) -> dict[str, Any]:
    return {"after_step": after_step, "observation": {"success": success, "summary": summary}}


def make_case(
    *, split: str, entity_index: int, entity: str, scenario: str, template_index: int
) -> dict[str, Any]:
    fact_id, fact, value = FACTS[entity_index % len(FACTS)]
    templates = TEMPLATES[split][scenario]
    query = templates[template_index % len(templates)].format(entity=entity, fact=fact)
    code = {
        "single_miss": "MISS1",
        "double_miss": "MISS2",
        "synthesis": "SYNTH",
        "coref": "COREF",
        "memory": "MEMORY",
        "direct": "DIRECT",
        "general": "GENERAL",
    }[scenario]
    case_id = f"SRV2-{split.upper()}-{code}-{entity_index + 1:03d}"
    common = {
        "case_id": case_id,
        "split": split,
        "entity_id": f"srv2_{split}_entity_{entity_index + 1:03d}",
        "group_id": f"srv2_{split}_entity_{entity_index + 1:03d}",
        "template_id": f"{scenario}_{split}_{template_index % len(templates) + 1}",
        "scenario_id": scenario,
        "fact_id": fact_id,
        "user_query": query,
        "reward_spec": ACTION_DOMINANT_REWARD,
    }

    if scenario == "single_miss":
        return {
            **common,
            "category": "rag_single_miss_recovery",
            "setup": setup(),
            "expected_decisions": [
                rag_step(entity, fact, finish=False),
                rewrite_step(entity, fact, reason="rag_miss", round_index=2),
                rag_step(entity, fact, finish=True),
            ],
            "mock_observations": [
                observation(1, success=False, summary=f"首次检索未命中{entity}的{fact}，需要收窄查询。"),
                observation(2, success=True, summary=f"查询已改写为更具体的{entity}{fact}字段。"),
            ],
            "forbidden_actions": ["answerer", *UNRELATED_ACTIONS],
        }

    if scenario == "double_miss":
        return {
            **common,
            "category": "rag_double_miss_recovery",
            "setup": setup(),
            "expected_decisions": [
                rag_step(entity, fact, finish=False),
                rewrite_step(entity, fact, reason="rag_miss", round_index=2),
                rag_step(entity, fact, finish=False),
                rewrite_step(entity, fact, reason="narrow_scope", round_index=3),
                rag_step(entity, fact, finish=True),
            ],
            "mock_observations": [
                observation(1, success=False, summary=f"第一轮检索未命中{entity}的{fact}。"),
                observation(2, success=True, summary=f"已完成第一轮收窄改写，准备第二次检索{entity}{fact}。"),
                observation(3, success=False, summary=f"第二轮仍无{entity}{fact}证据，允许最后一次改写。"),
                observation(4, success=True, summary=f"最终查询已限定到{entity}的{fact}字段。"),
            ],
            "forbidden_actions": ["answerer", *UNRELATED_ACTIONS],
        }

    if scenario == "synthesis":
        return {
            **common,
            "category": "rag_hit_then_synthesize",
            "setup": setup(),
            "expected_decisions": [
                rag_step(entity, fact, finish=False),
                answerer_step(mode="rag_evidence"),
            ],
            "mock_observations": [
                observation(1, success=True, summary=f"内部证据显示{entity}的{fact}为：{value}。需继续综合解释。")
            ],
            "forbidden_actions": ["re_question", *UNRELATED_ACTIONS],
        }

    if scenario == "coref":
        prior = prior_trajectory(
            case_id,
            f"{entity}是否已有可复用能力？",
            f"上一问已确定实体为{entity}，但尚未核验{fact}。",
        )
        return {
            **common,
            "category": "coref_rewrite_then_rag",
            "setup": setup(prior),
            "expected_decisions": [
                rewrite_step(entity, fact, reason="coref_resolve", round_index=1),
                rag_step(entity, fact, finish=True),
            ],
            "mock_observations": [
                observation(1, success=True, summary=f"指代已补全为：查询{entity}的{fact}。")
            ],
            "forbidden_actions": ["answerer", *UNRELATED_ACTIONS],
        }

    if scenario == "memory":
        prior = prior_trajectory(
            case_id,
            f"请查询{entity}的{fact}。",
            f"内部证据显示{entity}的{fact}为：{value}。现有证据充分。",
        )
        return {
            **common,
            "category": "memory_hit_end_guardrail",
            "setup": setup(prior),
            "expected_decisions": [
                {
                    "decision_type": "end",
                    "required_args": {"end_reason": "memory_hit"},
                    "arg_contains": {},
                }
            ],
            "mock_observations": [],
            "forbidden_actions": ["rag_answer", "re_question", "answerer", *UNRELATED_ACTIONS],
        }

    if scenario == "direct":
        return {
            **common,
            "category": "direct_rag_guardrail",
            "setup": setup(),
            "expected_decisions": [rag_step(entity, fact, finish=True)],
            "mock_observations": [],
            "forbidden_actions": ["re_question", "answerer", *UNRELATED_ACTIONS],
        }

    return {
        **common,
        "category": "general_answer_guardrail",
        "setup": setup(),
        "expected_decisions": [answerer_step(mode="direct")],
        "mock_observations": [],
        "forbidden_actions": ["rag_answer", "re_question", *UNRELATED_ACTIONS],
    }


def build_split(split: str) -> list[dict[str, Any]]:
    scenarios = ("single_miss", "double_miss", "synthesis", "coref", "memory", "direct", "general")
    return [
        make_case(
            split=split,
            entity_index=entity_index,
            entity=entity,
            scenario=scenario,
            template_index=entity_index + scenario_index,
        )
        for entity_index, entity in enumerate(ENTITY_SPLITS[split])
        for scenario_index, scenario in enumerate(scenarios)
    ]


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
        "max_steps_per_case": max(len(row["expected_decisions"]) for row in rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    report: dict[str, Any] = {"dataset_id": "planner_stateful_retrieval_v2", "splits": {}}
    all_rows: list[dict[str, Any]] = []
    for split in ("train", "dev", "test"):
        rows = build_split(split)
        all_rows.extend(rows)
        write_jsonl(output_dir / f"planner_stateful_retrieval_v2_{split}_cases.jsonl", rows)
        report["splits"][split] = summarize(rows)
    write_jsonl(output_dir / "planner_stateful_retrieval_v2_all_cases.jsonl", all_rows)
    report["all"] = summarize(all_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
