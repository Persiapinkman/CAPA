from __future__ import annotations

"""
Centralized JSON schemas for demo prompts/tools.

主要功能：
- 集中维护 Planner / Answerer / Rewrite 的 response schema。
- 集中维护工具参数 schema（供 registry 组装工具声明）。
"""


def build_agent_step_response_format(valid_actions: list[str]) -> dict:
    tool_actions = [item for item in valid_actions if item != "final_answer"]
    tool_branch_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "thought": {
                "type": "string",
                "description": "当前回合的分析与计划，中文，简洁但要说明为什么选择该动作。",
            },
            "decision_type": {
                "type": "string",
                "const": "tool",
                "description": "本轮决策类型：tool(执行动作)。",
            },
            "action": {
                "type": "string",
                "enum": tool_actions,
                "description": "本轮要执行的工具动作。",
            },
            "action_input": {
                "type": "object",
                "description": "对应工具入参；tool 模式下必须提供。",
            },
            "final_answer": {
                "type": "string",
                "description": "tool 模式下通常为空字符串，可省略。",
            },
        },
        "required": ["thought", "decision_type", "action", "action_input"],
    }
    end_branch_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "thought": {
                "type": "string",
                "description": "当前回合的分析与计划，中文，简洁但要说明为什么结束。",
            },
            "decision_type": {
                "type": "string",
                "const": "end",
                "description": "本轮决策类型：end(结束该轮次的任务)。",
            },
            "end_reason": {
                "type": "string",
                "enum": ["recheck_done", "memory_hit"],
                "description": "结束原因：recheck_done / memory_hit。",
            },
            "final_answer": {
                "type": "string",
                "description": "当 end 且无需再调用 Answerer 时，可直接填写最终答复；否则通常为空字符串。",
            },
        },
        "required": ["thought", "decision_type", "end_reason", "final_answer"],
    }
    clarify_branch_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "thought": {
                "type": "string",
                "description": "当前回合的分析与计划，中文，简洁说明为何需要向用户澄清。",
            },
            "decision_type": {
                "type": "string",
                "const": "clarify",
                "description": "本轮决策类型：clarify(向用户追问澄清)。",
            },
            "clarification_question": {
                "type": "string",
                "description": "直接发给用户的澄清问题，中文，单轮可回答。",
            },
        },
        "required": ["thought", "decision_type", "clarification_question"],
    }
    agent_step_schema = {
        "oneOf": [tool_branch_schema, end_branch_schema, clarify_branch_schema],
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "agent_step_decision",
            "strict": True,
            "schema": agent_step_schema,
        },
    }


ANSWERER_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "user_query": {"type": "string"},
        "evidence": {
            "type": ["object", "null"],
            "properties": {
                "retrieved_chunks": {"type": "array"},
                "query_trajectories": {"type": "array"},
            },
        },
    },
    "required": ["user_query"],
}

ANSWERER_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "final_answer": {"type": "string"},
    },
    "required": ["final_answer"],
}

ANSWERER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "answerer_output",
        "schema": ANSWERER_OUTPUT_SCHEMA,
    },
}

REWRITE_QUERY_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "rewrite_query_output",
        "schema": {
            "type": "object",
            "properties": {
                "rewritten_query": {
                    "type": "string",
                    "description": "改写后的检索 query，单行文本。",
                }
            },
            "required": ["rewritten_query"],
        },
    },
}

QUERY_TRAJECTORY_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "result_summary": {"type": "string"},
    },
    "required": ["result_summary"],
    "additionalProperties": False,
}

QUERY_TRAJECTORY_SUMMARY_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "query_traj_summary",
        "schema": QUERY_TRAJECTORY_SUMMARY_SCHEMA,
    },
}

ANSWER_RESOLUTION_JUDGER_SCHEMA = {
    "type": "object",
    "properties": {
        "resolved": {"type": "boolean"},
        "reason": {"type": "string"},
        "clarification_question": {"type": "string"},
    },
    "required": ["resolved", "reason", "clarification_question"],
    "additionalProperties": False,
}

ANSWER_RESOLUTION_JUDGER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer_resolution_judger",
        "schema": ANSWER_RESOLUTION_JUDGER_SCHEMA,
    },
}

FINISH_AFTER_TOOL_FIELD = {
    "type": "boolean",
    "description": (
        "该工具执行完成后是否可以直接结束本轮请求。"
        "若工具结果只是中间状态、后续还要继续规划其它工具，则填 false；"
        "若该工具结果就是用户要的最终结果，则填 true。"
    ),
}

RAG_ANSWER_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "完全使用“用户原始输入”",
        },
        "finish_after_tool": FINISH_AFTER_TOOL_FIELD,
    },
    "required": ["query"],
}

RE_QUESTION_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "待改写的原始查询（通常为用户问题或上一轮查询）。",
        },
        "rewrite_reason": {
            "type": "string",
            "description": "改写原因（如 rag_miss / coref_resolve / narrow_scope）。",
        },
        "retrieval_round": {
            "type": "integer",
            "minimum": 1,
            "maximum": 3,
            "description": "当前检索轮次（1~3）。",
        },
        "context_hint": {
            "type": "string",
            "description": "可选上下文提示，用于指代消解（例如上一问中的实体）。",
        },
        "finish_after_tool": FINISH_AFTER_TOOL_FIELD,
    },
    "required": ["query", "rewrite_reason", "retrieval_round"],
}

ANSWERER_PARAMETERS = {
    "type": "object",
    "properties": {
        "mode": {
            "type": "string",
            "enum": ["direct", "rag_evidence", "memoryquery_trajectories"],
            "description": (
                "direct=通用知识直接回答；"
                "rag_evidence=主要依据 evidence.retrieved_chunks（可含 RAG 未命中时的弱证据）；"
                "memoryquery_trajectories=主要依据 evidence.query_trajectories。"
            ),
        },
        "finish_after_tool": FINISH_AFTER_TOOL_FIELD,
    },
    "required": [],
}

FLUX_PARAMETERS = {
    "type": "object",
    "properties": {
        "task_text": {"type": "string"},
        "source_image_required": {"type": "boolean"},
        "num_images": {"type": "integer", "minimum": 1, "maximum": 5},
        "finish_after_tool": FINISH_AFTER_TOOL_FIELD,
    },
    "required": ["task_text", "source_image_required", "num_images"],
}

OPEN_SET_DETECTION_PARAMETERS = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "finish_after_tool": FINISH_AFTER_TOOL_FIELD,
    },
    "required": ["label"],
}

PIPELINE_EVAL_PARAMETERS = {
    "type": "object",
    "properties": {
        "task_text": {"type": "string"},
        "finish_after_tool": FINISH_AFTER_TOOL_FIELD,
    },
    "required": ["task_text"],
}

MIGRATION_ADVISOR_PARAMETERS = {
    "type": "object",
    "properties": {
        "user_query": {
            "type": "string",
            "description": "用户的原始迁移/能力边界需求，尽量完整保留。",
        },
        "use_image": {
            "type": "boolean",
            "description": "当前是否需要把上传图片作为迁移顾问的样例证据；有样例图且问题提到“这张图/样例图/图片里的目标”时填 true。",
        },
        "use_visual_probe": {
            "type": "boolean",
            "description": "是否允许迁移顾问内部对样例图运行轻量视觉探针；有图且用户要判断可行性/能力边界时通常为 true。",
        },
        "finish_after_tool": FINISH_AFTER_TOOL_FIELD,
    },
    "required": ["user_query", "use_image", "use_visual_probe"],
}

ADELA_CLI_EVAL_PARAMETERS = {
    "type": "object",
    "properties": {
        "model_name": {
            "type": "string",
            "description": "Adela 上的模型名称。若未提供 rawmodel_id，可先提供该字段让系统通过 RAG 解析模型 ID。禁止自己编造",
        },
        "rawmodel_id": {
            "type": "integer",
            "description": "Adela 原模型 ID。",
        },
        "platform": {
            "type": "string",
            "description": "目标部署平台，例如 cuda11.0-trt7.1-int8-T4。在用户未提供该类信息的时候，禁止编造",
        },
        "eval_type": {
            "type": "integer",
            "enum": [0, 1],
            "description": "0=精度评测，1=性能评测。在用户未提供该类信息的时候，禁止编造",
        },
        "finish_after_tool": FINISH_AFTER_TOOL_FIELD,
    },
    "required": ["platform", "eval_type"],
}
