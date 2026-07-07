#!/usr/bin/env python3
"""Build deterministic expanded Planner GRPO cases from local templates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_REWARD_SPEC = {
    "json_valid": 0.10,
    "decision_type_valid": 0.10,
    "action_match": 0.35,
    "argument_match": 0.25,
    "finish_after_tool": 0.10,
    "no_forbidden_action": 0.10,
}

IMAGE_FIXTURES = {
    "fisherman": "examples/images/fisherman.jpg",
    "person_with_bag": "examples/images/person_with_bag.png",
    "trash_truck": "examples/images/trash_truck.jpg",
    "smoke": "examples/images/smoke.jpg",
    "banner": "examples/images/banner.jpg",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: row must be object")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def base_case(
    *,
    case_id: str,
    category: str,
    user_query: str,
    has_image: bool,
    image_fixture: str = "",
    expected_decisions: list[dict[str, Any]],
    forbidden_actions: list[str],
    mock_observations: list[dict[str, Any]] | None = None,
    reward_spec: dict[str, float] | None = None,
    query_trajectories: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": case_id,
        "category": category,
        "user_query": user_query,
        "setup": {
            "has_image": has_image,
            "image_fixture": image_fixture,
            "query_trajectories": query_trajectories or [],
        },
        "expected_decisions": expected_decisions,
        "forbidden_actions": forbidden_actions,
        "reward_spec": dict(reward_spec) if reward_spec else dict(DEFAULT_REWARD_SPEC),
    }
    if mock_observations:
        row["mock_observations"] = mock_observations
    return row


def tool_step(
    *,
    action: str,
    finish_after_tool: bool,
    required_args: dict[str, Any] | None = None,
    arg_contains: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    args = dict(required_args or {})
    args["finish_after_tool"] = finish_after_tool
    return {
        "step": 1,
        "action": action,
        "required_args": args,
        "arg_contains": arg_contains or {},
    }


def two_step_probe_then_migration(
    *,
    label_tokens: list[str],
    migration_tokens: list[str],
) -> list[dict[str, Any]]:
    return [
        {
            "step": 1,
            "action": "qwen_detection",
            "required_args": {"finish_after_tool": False},
            "arg_contains": {"label": label_tokens},
        },
        {
            "step": 2,
            "action": "migration_advisor",
            "required_args": {
                "use_image": True,
                "use_visual_probe": True,
                "finish_after_tool": True,
            },
            "arg_contains": {"user_query": migration_tokens},
        },
    ]


def build_single_image_probe_cases(start_index: int, count: int) -> list[dict[str, Any]]:
    labels = [
        ("钓鱼的人", ["钓鱼", "人"], IMAGE_FIXTURES["fisherman"]),
        ("背包", ["背包"], IMAGE_FIXTURES["person_with_bag"]),
        ("垃圾车", ["垃圾车"], IMAGE_FIXTURES["trash_truck"]),
        ("烟雾", ["烟雾"], IMAGE_FIXTURES["smoke"]),
        ("横幅", ["横幅"], IMAGE_FIXTURES["banner"]),
        ("安全帽", ["安全帽"], IMAGE_FIXTURES["person_with_bag"]),
        ("厨师帽", ["厨师帽"], IMAGE_FIXTURES["person_with_bag"]),
        ("反光条", ["反光条"], IMAGE_FIXTURES["person_with_bag"]),
    ]
    phrasings = [
        "这张图里有没有{label}？用已有模型快速试一下。",
        "帮我看这张图是否出现{label}，直接检测就行，不要跑完整评测。",
        "客户只想先确认样例图里有没有{label}，用开放集模型试一下。",
        "先别做报告，帮我直接检测这张图里的{label}。",
        "这张参考图里疑似有{label}，用现有视觉模型验证一下。",
        "我只需要一个快速判断：图里有没有{label}？",
        "这张图先跑个轻量探针，看看有没有{label}。",
        "不要生成样本，也不要对比模型，直接检查图里是否有{label}。",
        "客户在等一个初判，帮我用现有检测能力看有没有{label}。",
        "这张样例图只需要判断{label}是否存在，别走完整流水线。",
    ]
    rows: list[dict[str, Any]] = []
    for i in range(count):
        label, tokens, fixture = labels[i % len(labels)]
        template = phrasings[(i // len(labels)) % len(phrasings)]
        rows.append(
            base_case(
                case_id=f"GRPO-EXP-VIS-PROBE-{start_index + i:03d}",
                category="single_image_probe",
                user_query=template.format(label=label),
                has_image=True,
                image_fixture=fixture,
                expected_decisions=[
                    tool_step(
                        action="qwen_detection",
                        finish_after_tool=True,
                        arg_contains={"label": tokens},
                    )
                ],
                forbidden_actions=["pipeline_eval", "migration_advisor", "rag_answer", "answerer"],
            )
        )
    return rows


def build_probe_then_migration_cases(start_index: int, count: int) -> list[dict[str, Any]]:
    tasks = [
        ("钓鱼的人", ["钓鱼", "人"], "钓鱼人员检测", ["钓鱼", "低成本验证"], IMAGE_FIXTURES["fisherman"]),
        ("厨师帽", ["厨师帽"], "后厨帽子佩戴检测", ["后厨", "帽子", "低成本验证"], IMAGE_FIXTURES["person_with_bag"]),
        ("反光条", ["反光条"], "工服反光条检测", ["反光条", "迁移", "低成本验证"], IMAGE_FIXTURES["person_with_bag"]),
        ("烟雾", ["烟雾"], "烟雾异常检测", ["烟雾", "可行性", "低成本"], IMAGE_FIXTURES["smoke"]),
        ("横幅", ["横幅"], "横幅违规悬挂检测", ["横幅", "迁移", "验证方案"], IMAGE_FIXTURES["banner"]),
        ("垃圾车", ["垃圾车"], "垃圾车识别", ["垃圾车", "已有能力", "验证"], IMAGE_FIXTURES["trash_truck"]),
    ]
    phrasings = [
        "这张图里有没有{label}？先用已有模型试一下。如果不确定，再给客户一个{business}的低成本验证方案。",
        "客户给了这张样例图，先检测有没有{label}；如果结果不稳，再判断{business}能不能迁移。",
        "请先对图里的{label}做快速探针，探针结果不确定的话，再给一个{business}的可行性方案。",
        "先别直接做完整评测，先看这张图有没有{label}，再决定{business}是否值得低成本验证。",
        "这张图作为样例，先用开放集模型试{label}；如果信心不足，再输出{business}迁移建议。",
        "先用这张图验证{label}能不能被现有模型看到；若不稳定，再给{business}的轻量验证路径。",
        "我想先做一个小探针：检测{label}。如果结果模糊，再分析{business}的复用空间。",
        "这不是完整评测，先检查{label}；如果检测没把握，再转成{business}迁移评估。",
        "先跑单图检测看{label}，再根据结果判断是否需要给客户{business}的低成本方案。",
        "请先用现有开放集能力试{label}，若样例不确定，再给{business}的风险和验证建议。",
    ]
    rows: list[dict[str, Any]] = []
    for i in range(count):
        label, label_tokens, business, mig_tokens, fixture = tasks[i % len(tasks)]
        template = phrasings[(i // len(tasks)) % len(phrasings)]
        rows.append(
            base_case(
                case_id=f"GRPO-EXP-PROBE-MIG-{start_index + i:03d}",
                category="probe_then_migration",
                user_query=template.format(label=label, business=business),
                has_image=True,
                image_fixture=fixture,
                expected_decisions=two_step_probe_then_migration(
                    label_tokens=label_tokens,
                    migration_tokens=mig_tokens,
                ),
                mock_observations=[
                    {
                        "after_step": 1,
                        "observation": {
                            "success": True,
                            "summary": "开放集检测结果不稳定，建议结合已有资产做低成本可行性判断。",
                        },
                    }
                ],
                forbidden_actions=["pipeline_eval", "answerer"],
            )
        )
    return rows


def build_migration_with_image_cases(start_index: int, count: int) -> list[dict[str, Any]]:
    tasks = [
        ("后厨帽子佩戴检测", ["后厨", "帽子佩戴", "已有能力"], IMAGE_FIXTURES["person_with_bag"]),
        ("工服反光条检测", ["反光条", "迁移", "补数据"], IMAGE_FIXTURES["person_with_bag"]),
        ("钓鱼人员识别", ["钓鱼", "已有能力", "风险"], IMAGE_FIXTURES["fisherman"]),
        ("烟雾异常检测", ["烟雾", "迁移", "验证"], IMAGE_FIXTURES["smoke"]),
        ("横幅违规悬挂检测", ["横幅", "复用", "风险"], IMAGE_FIXTURES["banner"]),
        ("垃圾车识别", ["垃圾车", "已有能力", "验证"], IMAGE_FIXTURES["trash_truck"]),
    ]
    phrasings = [
        "这张图是客户给的样例，他们想做{business}，帮我判断已有能力能不能支持，并给低成本验证方案。",
        "客户拿这张样例图问{business}能不能做，已有资产能复用多少？风险和补数据量也说一下。",
        "基于这张客户样例，评估{business}是否可以从现有能力迁移，必要时做轻量视觉探针。",
        "这张图代表客户场景，请给出{business}的迁移可行性、数据要求和低成本验证路径。",
        "不用马上跑完整评测，先基于样例图判断{business}有没有可复用资产和工程风险。",
        "客户给了样例图但还没立项，先判断{business}是否有历史能力可借用，并给验证建议。",
        "围绕这张样例图，帮我做{business}的能力边界判断，不要直接变成检测报告。",
        "请把这张图当作业务样例，分析{business}能否迁移、缺口在哪里、怎么低成本试。",
        "客户想知道这类图能不能做{business}，请结合样例图给资产复用和风险判断。",
    ]
    rows: list[dict[str, Any]] = []
    for i in range(count):
        business, tokens, fixture = tasks[i % len(tasks)]
        template = phrasings[(i // len(tasks)) % len(phrasings)]
        rows.append(
            base_case(
                case_id=f"GRPO-EXP-MIG-IMG-{start_index + i:03d}",
                category="migration_feasibility_with_image",
                user_query=template.format(business=business),
                has_image=True,
                image_fixture=fixture,
                expected_decisions=[
                    tool_step(
                        action="migration_advisor",
                        finish_after_tool=True,
                        required_args={"use_image": True, "use_visual_probe": True},
                        arg_contains={"user_query": tokens},
                    )
                ],
                forbidden_actions=["qwen_detection", "pipeline_eval", "answerer"],
            )
        )
    return rows


def build_migration_text_cases(start_index: int, count: int) -> list[dict[str, Any]]:
    tasks = [
        ("后厨帽子佩戴检测", ["后厨", "帽子", "低成本"]),
        ("河道漂浮物识别", ["河道", "漂浮物", "补数据"]),
        ("工服反光条检测", ["反光条", "迁移", "风险"]),
        ("夜间黑猫检测", ["黑猫", "夜间", "验证"]),
        ("安全绳挂接状态识别", ["安全绳", "迁移", "数据"]),
    ]
    phrasings = [
        "客户想做{business}，已有资产能不能支持？不确定的话给低成本验证方案。",
        "新需求是{business}，请评估能否由现有能力迁移，风险和补数据量是什么。",
        "{business}有没有可复用模型？如果不能直接支持，给一个分阶段验证方案。",
        "客户在问{business}能不能做，帮我从历史能力、数据要求和工程风险角度判断。",
        "我们准备评估{business}，先别跑检测，帮我判断迁移可行性和低成本验证路径。",
        "{business}这个需求如果要从现有资产迁移，大概有哪些风险和数据缺口？",
        "请给{business}做一个能力边界判断：能复用什么，不能复用时怎么验证。",
    ]
    rows: list[dict[str, Any]] = []
    for i in range(count):
        business, tokens = tasks[i % len(tasks)]
        template = phrasings[(i // len(tasks)) % len(phrasings)]
        rows.append(
            base_case(
                case_id=f"GRPO-EXP-MIG-TXT-{start_index + i:03d}",
                category="migration_feasibility",
                user_query=template.format(business=business),
                has_image=False,
                expected_decisions=[
                    tool_step(
                        action="migration_advisor",
                        finish_after_tool=True,
                        required_args={"use_image": False, "use_visual_probe": False},
                        arg_contains={"user_query": tokens},
                    )
                ],
                forbidden_actions=["qwen_detection", "pipeline_eval", "answerer"],
            )
        )
    return rows


def build_boundary_cases(start_index: int, count: int) -> list[dict[str, Any]]:
    specs = [
        (
            "historical_asset_qa",
            "安全绳检测历史上有哪些模型版本？按内部资料查一下。",
            "rag_answer",
            {"query": ["安全绳", "模型"]},
            ["migration_advisor", "pipeline_eval", "answerer"],
        ),
        (
            "historical_asset_qa",
            "河道漂浮物识别有没有历史能力或项目记录？",
            "rag_answer",
            {"query": ["河道", "漂浮物"]},
            ["migration_advisor", "pipeline_eval", "answerer"],
        ),
        (
            "general_answer",
            "目标检测错误分析报告一般怎么写？给我一个通用结构。",
            "answerer",
            {},
            ["rag_answer", "pipeline_eval", "migration_advisor"],
        ),
        (
            "general_answer",
            "用通俗语言解释一下目标检测 mAP 是什么，不要查内部文档。",
            "answerer",
            {},
            ["rag_answer", "pipeline_eval", "migration_advisor"],
        ),
        (
            "full_detection_eval",
            "用这张图扩增样本并做开放集检测效果对比，输出评估报告。",
            "pipeline_eval",
            {"task_text": ["检测"]},
            ["qwen_detection", "migration_advisor", "answerer"],
        ),
        (
            "full_detection_eval",
            "请基于参考图生成类似样本，并比较两个开放集模型检测效果。",
            "pipeline_eval",
            {"task_text": ["检测"]},
            ["qwen_detection", "migration_advisor", "answerer"],
        ),
        (
            "historical_asset_qa",
            "安全帽佩戴状态识别历史上支持过哪些标签？",
            "rag_answer",
            {"query": ["安全帽", "标签"]},
            ["migration_advisor", "pipeline_eval", "answerer"],
        ),
        (
            "historical_asset_qa",
            "电动车检测模型的推荐阈值和适用场景是什么？",
            "rag_answer",
            {"query": ["电动车", "阈值"]},
            ["migration_advisor", "pipeline_eval", "answerer"],
        ),
        (
            "general_answer",
            "解释一下开放集检测和闭集检测的区别，用项目经理能听懂的话。",
            "answerer",
            {},
            ["rag_answer", "pipeline_eval", "migration_advisor"],
        ),
        (
            "general_answer",
            "目标检测评估里为什么要看误检和漏检？给一个通用说明。",
            "answerer",
            {},
            ["rag_answer", "pipeline_eval", "migration_advisor"],
        ),
    ]
    rows: list[dict[str, Any]] = []
    for i in range(count):
        category, query, action, arg_contains, forbidden = specs[i % len(specs)]
        has_image = category == "full_detection_eval"
        rows.append(
            base_case(
                case_id=f"GRPO-EXP-BOUNDARY-{start_index + i:03d}",
                category=category,
                user_query=query,
                has_image=has_image,
                image_fixture=IMAGE_FIXTURES["person_with_bag"] if has_image else "",
                expected_decisions=[
                    tool_step(
                        action=action,
                        finish_after_tool=True,
                        required_args={"mode": "direct"} if action == "answerer" else {},
                        arg_contains=arg_contains,
                    )
                ],
                forbidden_actions=forbidden,
            )
        )
    return rows


# high-value GRPO reward specs. 只保留“工程/SFT/few-shot 都难压稳、又没有工程兜底”的两类：
# 1) Planner 单话题内的 probe->migration 多步软转移（核心）；2) 语义歧义 clarify（次优）。
CLARIFY_REWARD_SPEC = {
    # clarify 正例聚焦“该问就问”，主要考核 decision_type 与 clarify 动作，不强约束澄清问题文本。
    "json_valid": 0.10,
    "decision_type_valid": 0.30,
    "action_match": 0.50,
    "argument_match": 0.00,
    "finish_after_tool": 0.00,
    "no_forbidden_action": 0.10,
}

PROBE_MIGRATION_STRICT_REWARD_SPEC = {
    # 探针->迁移 转移：默认 6 维 + 过早收口过程奖励，直接惩罚“探针后立即收口”的残余失败偏置。
    **DEFAULT_REWARD_SPEC,
    "no_premature_stop": 0.30,
}


def build_clarify_cases(start_index: int, count: int) -> list[dict[str, Any]]:
    """语义歧义正例：意图存在多种高概率解释且导向不同工具路径，必须先 clarify。"""
    # (query, has_image, fixture_key)：query 本身多义，任何直接动作都是过早猜测。
    ambiguous = [
        ("黑夜检测黑猫，帮我弄一下。", False, ""),
        ("安全帽这个东西帮我做一下。", False, ""),
        ("河道漂浮物能不能整一个？", False, ""),
        ("这张图，烟雾，处理一下。", True, "smoke"),
        ("横幅，你看着办。", True, "banner"),
        ("钓鱼这个搞一下。", True, "fisherman"),
        ("垃圾车弄个方案出来。", False, ""),
        ("反光条这块儿你处理下。", True, "person_with_bag"),
    ]
    forbidden = [
        "qwen_detection",
        "rexomni_detection",
        "flux-image-generation",
        "pipeline_eval",
        "migration_advisor",
        "rag_answer",
        "answerer",
    ]
    rows: list[dict[str, Any]] = []
    for i in range(count):
        query, has_image, fixture_key = ambiguous[i % len(ambiguous)]
        rows.append(
            base_case(
                case_id=f"GRPO-EXP-CLARIFY-{start_index + i:03d}",
                category="clarify_intent_ambiguity",
                user_query=query,
                has_image=has_image,
                image_fixture=IMAGE_FIXTURES[fixture_key] if fixture_key else "",
                expected_decisions=[
                    {"step": 1, "decision_type": "clarify", "action": "clarify", "arg_contains": {}}
                ],
                forbidden_actions=forbidden,
                reward_spec=CLARIFY_REWARD_SPEC,
            )
        )
    return rows


def build_probe_only_contrastive_cases(start_index: int, count: int) -> list[dict[str, Any]]:
    """探针->迁移 的 B 对照组：句式提到业务/迁移词看似要转移，但意图明确“确认完就结束”，
    应单步 detection 且 finish_after_tool=true，禁止转 migration。与 build_probe_migration_strict_cases
    共享检测目标，构成 A(转移)/B(不转移) 对照对，逼模型学 finish_after_tool 的语境敏感性。"""
    tasks = [
        ("钓鱼的人", ["钓鱼", "人"], "钓鱼人员检测", "fisherman"),
        ("厨师帽", ["厨师帽"], "后厨帽子佩戴检测", "person_with_bag"),
        ("反光条", ["反光条"], "工服反光条检测", "person_with_bag"),
        ("烟雾", ["烟雾"], "烟雾异常检测", "smoke"),
        ("横幅", ["横幅"], "横幅违规悬挂检测", "banner"),
        ("垃圾车", ["垃圾车"], "垃圾车识别", "trash_truck"),
    ]
    phrasings = [
        "先确认这张图里有没有{label}就行，{business}的迁移可行性我自己判断，你不用分析。",
        "只要快速看下有没有{label}，看完就结束，别再做{business}的迁移评估。",
        "这张图我只需要{label}的探针结果，{business}后续方案先不用你给。",
        "帮我单纯检测下{label}，确认完直接收口，不要接着分析{business}能不能迁移。",
        "就跑一次{label}检测告诉我有没有，{business}的事这轮不涉及。",
    ]
    rows: list[dict[str, Any]] = []
    for i in range(count):
        label, label_tokens, business, fixture_key = tasks[i % len(tasks)]
        template = phrasings[(i // len(tasks)) % len(phrasings)]
        rows.append(
            base_case(
                case_id=f"GRPO-EXP-PROBE-ONLY-{start_index + i:03d}",
                category="probe_only_contrastive",
                user_query=template.format(label=label, business=business),
                has_image=True,
                image_fixture=IMAGE_FIXTURES[fixture_key],
                expected_decisions=[
                    tool_step(
                        action="qwen_detection",
                        finish_after_tool=True,
                        arg_contains={"label": label_tokens},
                    )
                ],
                forbidden_actions=["migration_advisor", "pipeline_eval", "answerer"],
            )
        )
    return rows


def build_probe_migration_strict_cases(start_index: int, count: int) -> list[dict[str, Any]]:
    """探针->迁移 强化：query 明确隐含“探针不确定后再判断迁移”，
    step1 detection 必须 finish_after_tool=false，探针后不得过早收口（no_premature_stop 生效）。"""
    tasks = [
        ("钓鱼的人", ["钓鱼", "人"], "钓鱼人员检测", ["钓鱼", "低成本验证"], "fisherman"),
        ("厨师帽", ["厨师帽"], "后厨帽子佩戴检测", ["后厨", "帽子", "低成本验证"], "person_with_bag"),
        ("反光条", ["反光条"], "工服反光条检测", ["反光条", "迁移", "低成本验证"], "person_with_bag"),
        ("烟雾", ["烟雾"], "烟雾异常检测", ["烟雾", "可行性", "低成本"], "smoke"),
        ("横幅", ["横幅"], "横幅违规悬挂检测", ["横幅", "迁移", "验证方案"], "banner"),
        ("垃圾车", ["垃圾车"], "垃圾车识别", ["垃圾车", "已有能力", "验证"], "trash_truck"),
    ]
    phrasings = [
        "先用已有模型探一下这张图有没有{label}；探针不确定的话，再给{business}的低成本迁移验证方案。",
        "第一步只做轻量探针看{label}，如果结果不稳，第二步再判断{business}能否迁移并给验证路径。",
        "先别收口：先检测{label}，探针信心不足时接着输出{business}的迁移可行性与风险建议。",
        "先跑单图检测看有没有{label}；要是模型没把握，再评估{business}的资产复用和补数据量。",
        "这张图先探{label}，探针结果模糊就继续分析{business}能不能从现有能力迁移。",
    ]
    rows: list[dict[str, Any]] = []
    for i in range(count):
        label, label_tokens, business, mig_tokens, fixture_key = tasks[i % len(tasks)]
        template = phrasings[(i // len(tasks)) % len(phrasings)]
        rows.append(
            base_case(
                case_id=f"GRPO-EXP-PROBE-MIG-STRICT-{start_index + i:03d}",
                category="probe_then_migration_strict",
                user_query=template.format(label=label, business=business),
                has_image=True,
                image_fixture=IMAGE_FIXTURES[fixture_key],
                expected_decisions=two_step_probe_then_migration(
                    label_tokens=label_tokens,
                    migration_tokens=mig_tokens,
                ),
                mock_observations=[
                    {
                        "after_step": 1,
                        "observation": {
                            "success": True,
                            "summary": "开放集探针结果不稳定/置信度不足，建议结合已有资产做低成本可行性判断。",
                        },
                    }
                ],
                forbidden_actions=["pipeline_eval", "answerer"],
                reward_spec=PROBE_MIGRATION_STRICT_REWARD_SPEC,
            )
        )
    return rows


def build_expanded_cases() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(build_single_image_probe_cases(1, 70))
    rows.extend(build_probe_then_migration_cases(1, 60))
    rows.extend(build_migration_with_image_cases(1, 50))
    rows.extend(build_migration_text_cases(1, 35))
    rows.extend(build_boundary_cases(1, 30))
    # 高价值 GRPO 强化：集中在 Planner 单话题内 probe->migration 多步软转移（A/B 对照）与语义歧义 clarify。
    rows.extend(build_clarify_cases(1, 16))
    rows.extend(build_probe_migration_strict_cases(1, 30))  # A：探针后转 migration + no_premature_stop
    rows.extend(build_probe_only_contrastive_cases(1, 30))  # B：迷惑句式纯探针，finish=true 且禁 migration
    return rows


def dedupe_by_query(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = str(row.get("user_query") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def build_report(*, seed_rows: list[dict[str, Any]], expanded_rows: list[dict[str, Any]], train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "seed_count": len(seed_rows),
        "expanded_count": len(expanded_rows),
        "train_count": len(train_rows),
        "expanded_by_category": dict(Counter(str(row.get("category") or "") for row in expanded_rows)),
        "train_by_category": dict(Counter(str(row.get("category") or "") for row in train_rows)),
        "train_by_first_expected_action": dict(
            Counter(
                str((row.get("expected_decisions") or [{}])[0].get("action") or "")
                for row in train_rows
                if isinstance(row.get("expected_decisions"), list) and row.get("expected_decisions")
            )
        ),
        "train_by_expected_decision_action": dict(
            Counter(
                str(step.get("action") or "")
                for row in train_rows
                for step in (row.get("expected_decisions") if isinstance(row.get("expected_decisions"), list) else [])
                if isinstance(step, dict)
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build expanded Planner GRPO case files.")
    parser.add_argument(
        "--seed",
        type=Path,
        default=Path("training/planner_grpo_seed_v1/cases/planner_grpo_seed_cases.jsonl"),
        help="Seed cases JSONL",
    )
    parser.add_argument(
        "--expanded-out",
        type=Path,
        default=Path("training/planner_grpo_seed_v1/cases/planner_grpo_expanded_cases.jsonl"),
        help="Expanded generated cases JSONL",
    )
    parser.add_argument(
        "--train-out",
        type=Path,
        default=Path("training/planner_grpo_seed_v1/cases/planner_grpo_train_cases.jsonl"),
        help="Seed + expanded training cases JSONL",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("training/planner_grpo_seed_v1/reports/planner_grpo_expanded_data_report.json"),
        help="Data distribution report",
    )
    args = parser.parse_args()

    seed_rows = load_jsonl(args.seed)
    expanded_rows = dedupe_by_query(build_expanded_cases())
    train_rows = seed_rows + expanded_rows

    write_jsonl(args.expanded_out, expanded_rows)
    write_jsonl(args.train_out, train_rows)
    report = build_report(seed_rows=seed_rows, expanded_rows=expanded_rows, train_rows=train_rows)
    write_json(args.report_out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
