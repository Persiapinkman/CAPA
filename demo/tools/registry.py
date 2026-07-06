"""
Tool registry for the demo agent.

主要功能：
- 集中定义工具常量、动作名与 `final_answer` 特殊动作。
- 维护 Planner 可见的工具 JSON Schema（含 `finish_after_tool` 约束）。
- 提供工具合法性判断与 schema 读取入口。

主要模块：
- 常量区：`TOOL_*` / `ACTION_FINAL_ANSWER`
- Schema 区：`TOOL_SCHEMAS`
- 工具函数：`get_tool_schemas()` / `is_valid_tool_action()`
"""

from __future__ import annotations

from tools import schemas as schema_defs

TOOL_RAG_ANSWER = "rag_answer"
TOOL_RE_QUESTION = "re_question"
TOOL_ANSWERER = "answerer"
TOOL_FLUX_IMAGE_GENERATION = "flux-image-generation"
TOOL_QWEN_DETECTION = "qwen_detection"
TOOL_REXOMNI_DETECTION = "rexomni_detection"
TOOL_PIPELINE_EVAL = "pipeline_eval"
TOOL_MIGRATION_ADVISOR = "migration_advisor"
TOOL_ADELA_CLI_EVAL = "adela_cli_eval"
ACTION_FINAL_ANSWER = "final_answer"
ACTION_RAG_ANSWER = TOOL_RAG_ANSWER
ACTION_RE_QUESTION = TOOL_RE_QUESTION
ACTION_ANSWERER = TOOL_ANSWERER
ACTION_FLUX_IMAGE_GENERATION = TOOL_FLUX_IMAGE_GENERATION
ACTION_QWEN_OPEN_SET_DETECTION = "qwen-vlm-open-set-delection"
ACTION_REXOMNI_OPEN_SET_DETECTION = "rexomni-open-set-detection"
ACTION_TARGET_DETECTION_EVALUATION = "target-detection-evaluation"
ACTION_MIGRATION_ADVISOR = TOOL_MIGRATION_ADVISOR
ACTION_ADELA_CLI_EVAL = TOOL_ADELA_CLI_EVAL

TOOL_DECLARATIONS = [
    {
        "name": TOOL_RAG_ANSWER,
        "legacy_action": ACTION_RAG_ANSWER,
        "executor_branch": "rag",
        "flow": "rag",
        "requires_image": False,
        "schema": {
            "description": (
                "适合用**知识库 / RAG 检索**回答的问题：如公司内部私有业务知识、项目文档、模型版本、标签含义、safety_rope 等业务与配置类问答；"
                "不适用于已在问题中写明 Adela 类部署平台标识（如 cuda*-trt*-*）并追问该模型在该平台上的精度、性能或 benchmark 数值的场景，此类应使用 adela_cli_eval。"
            ),
            "parameters": {
                **schema_defs.RAG_ANSWER_PARAMETERS,
            },
        },
    },
    {
        "name": TOOL_RE_QUESTION,
        "legacy_action": ACTION_RE_QUESTION,
        "executor_branch": "re_question",
        "flow": "re_question",
        "requires_image": False,
        "schema": {
            "description": (
                "在 RAG 未命中时，针对当前问题进行小步改写后继续检索。"
                "当问题含有指代（如“这个模型/它/上述方案”）或表述过短导致检索不稳定时，"
                "先调用 re_question 做实体补全与最小改写，再交给 rag_answer。"
                "示例：历史问题“安全绳检测用什么模型？”，当前问题“这个模型的精度如何？”"
                " -> 改写为“安全绳检测模型的精度如何”。"
            ),
            "parameters": {
                **schema_defs.RE_QUESTION_PARAMETERS,
            },
        },
    },
    {
        "name": TOOL_ANSWERER,
        "legacy_action": ACTION_ANSWERER,
        "executor_branch": "answerer",
        "flow": "direct_answer",
        "requires_image": False,
        "schema": {
            "description": "无需查阅公司内部文档的问题，比如：苹果是当季水果吗",
            "parameters": {
                **schema_defs.ANSWERER_PARAMETERS,
            },
        },
    },
    {
        "name": TOOL_FLUX_IMAGE_GENERATION,
        "legacy_action": ACTION_FLUX_IMAGE_GENERATION,
        "executor_branch": "flux",
        "flow": "flux",
        "requires_image": False,
        "schema": {
            "description": "图像生成工具。可纯文本生成，也可结合参考图进行变化生成。",
            "parameters": {
                **schema_defs.FLUX_PARAMETERS,
            },
        },
    },
    {
        "name": TOOL_QWEN_DETECTION,
        "legacy_action": ACTION_QWEN_OPEN_SET_DETECTION,
        "executor_branch": "qwen_detection",
        "flow": "qwen_detect",
        "requires_image": True,
        "schema": {
            "description": "适用于检测目标的任务 Qwen。",
            "parameters": {
                **schema_defs.OPEN_SET_DETECTION_PARAMETERS,
            },
        },
    },
    {
        "name": TOOL_REXOMNI_DETECTION,
        "legacy_action": ACTION_REXOMNI_OPEN_SET_DETECTION,
        "executor_branch": "rexomni_detection",
        "flow": "rexomni_detect",
        "requires_image": True,
        "schema": {
            "description": "适用于检测目标的任务 Rex-Omni",
            "parameters": {
                **schema_defs.OPEN_SET_DETECTION_PARAMETERS,
            },
        },
    },
    {
        "name": TOOL_PIPELINE_EVAL,
        "legacy_action": ACTION_TARGET_DETECTION_EVALUATION,
        "executor_branch": "pipeline",
        "flow": "pipeline",
        "requires_image": True,
        "schema": {
            "description": "完整目标检测评测流水线：图片生成 - 分别使用 Qwen 和 Rex-Omni 进行标注 - 评估模型检测效果",
            "parameters": {
                **schema_defs.PIPELINE_EVAL_PARAMETERS,
            },
        },
    },
    {
        "name": TOOL_MIGRATION_ADVISOR,
        "legacy_action": ACTION_MIGRATION_ADVISOR,
        "executor_branch": "migration_advisor",
        "flow": "migration_advisor",
        "requires_image": False,
        "schema": {
            "description": (
                "迁移/能力边界报告助手。适用于用户询问新需求能否由现有模型或历史能力迁移、能力边界在哪里、"
                "需要补多少数据、工程/成本/风险如何，以及“不能直接支持怎么办”的问题。"
                "该工具内部会按字段检索历史资产与相似模型，并在有样例图时可执行轻量视觉探针，最后输出固定结构的迁移评估报告。"
                "不要把这类问题拆成普通 rag_answer 或 pipeline_eval；pipeline_eval 只用于用户明确要求生成样本并做目标检测评测。"
            ),
            "parameters": {
                **schema_defs.MIGRATION_ADVISOR_PARAMETERS,
            },
        },
    },
    {
        "name": TOOL_ADELA_CLI_EVAL,
        "legacy_action": ACTION_ADELA_CLI_EVAL,
        "executor_branch": "adela_cli",
        "flow": "adela_eval",
        "requires_image": False,
        "schema": {
            "description": (
                "用于在 Adela 上对指定模型与部署平台执行部署与 benchmark 评测。"
                "典型场景：已知或可从知识库解析的模型名/rawmodel_id，搭配具体平台串（如 cuda11.0-trt7.1-fp32-T4），"
                "询问该模型在该平台上的精度（eval_type=0）或性能（eval_type=1）。"
                "仅填 model_name 时，系统会在工具内先经 RAG 解析出 rawmodel_id，再调用 Adela CLI，不要改用 rag_answer 单独回答此类问题。"
            ),
            "parameters": {
                **schema_defs.ADELA_CLI_EVAL_PARAMETERS,
            },
        },
    },
]

TOOL_NAME_TO_DECLARATION = {item["name"]: item for item in TOOL_DECLARATIONS}
LEGACY_ACTION_TO_DECLARATION = {
    item["legacy_action"]: item for item in TOOL_DECLARATIONS
}
TOOL_NAME_TO_LEGACY_ACTION = {
    item["name"]: item["legacy_action"] for item in TOOL_DECLARATIONS
}
LEGACY_ACTION_TO_TOOL_NAME = {
    item["legacy_action"]: item["name"] for item in TOOL_DECLARATIONS
}
TOOL_NAME_TO_FLOW = {item["name"]: item["flow"] for item in TOOL_DECLARATIONS}
TOOL_NAME_TO_EXECUTOR_BRANCH = {
    item["name"]: item["executor_branch"] for item in TOOL_DECLARATIONS
}
TOOL_NAME_REQUIRES_IMAGE = {
    item["name"]: bool(item["requires_image"]) for item in TOOL_DECLARATIONS
}


def get_tool_schemas() -> list[dict]:
    return [
        {
            "name": item["name"],
            "description": item["schema"]["description"],
            "parameters": item["schema"]["parameters"],
        }
        for item in TOOL_DECLARATIONS
    ]


def get_declared_tool_names() -> list[str]:
    return [item["name"] for item in TOOL_DECLARATIONS]


def normalize_tool_action(action: str) -> str:
    raw = str(action or "").strip()
    if raw in TOOL_NAME_TO_DECLARATION:
        return raw
    return LEGACY_ACTION_TO_TOOL_NAME.get(raw, raw)


def to_legacy_action(action: str) -> str:
    normalized = normalize_tool_action(action)
    return TOOL_NAME_TO_LEGACY_ACTION.get(normalized, str(action or "").strip())


def flow_for_action(action: str) -> str:
    normalized = normalize_tool_action(action)
    return TOOL_NAME_TO_FLOW.get(normalized, "rag")


def executor_branch_for_action(action: str) -> str:
    normalized = normalize_tool_action(action)
    return TOOL_NAME_TO_EXECUTOR_BRANCH.get(normalized, "")


def action_requires_image(action: str) -> bool:
    normalized = normalize_tool_action(action)
    return bool(TOOL_NAME_REQUIRES_IMAGE.get(normalized, False))


def is_valid_tool_action(action: str) -> bool:
    valid = set(get_declared_tool_names())
    valid.add(ACTION_FINAL_ANSWER)
    return str(action or "").strip() in valid
