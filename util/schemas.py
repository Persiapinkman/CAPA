# schemas.py

intent_response_schema = {
    "type": "object",
    "properties": {
        "task_type": {
            "type": "string",
            "enum": ["detection", "classification"]
        },
        "task_name": {"type": "string"},
        "target_label": {"type": "string"},
        "target_keywords": {"type": "array", "items": {"type": "string"}},
        "scene": {"type": "string"},
        "target": {"type": "string"},
        "camera": {"type": "string"},
        "expand_scene": {"type": "string"},
        "requirement_background": {"type": "string"},
        "solution": {"type": "string"},
        "annotation_spec": {"type": "string"},
        "prompts": {"type": "array", "items": {"type": "string"}}
    },
    "required": [
        "task_type",
        "task_name",
        "target_label",
        "target_keywords",
        "scene",
        "target",
        "camera",
        "expand_scene",
        "requirement_background",
        "solution",
        "annotation_spec",
        "prompts",
    ]
}

intent_response_format = dict(
    type="json_schema",
    json_schema=dict(
        name="intent_schema",
        schema=intent_response_schema
    )
)


detection_response_schema = {
    "type": "object",
    "properties": {
        "bboxes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "bbox": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4
                    },
                    "score": {"type": "number"}
                },
                "required": ["label", "bbox", "score"]
            }
        }
    },
    "required": ["bboxes"]
}

detection_response_format = dict(
    type="json_schema",
    json_schema=dict(
        name="detection_schema",
        schema=detection_response_schema
    )
)

# 批量检测：一次请求多张图，返回 results[i] 对应第 i 张图
detection_batch_response_schema = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "bboxes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "bbox": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 4,
                                    "maxItems": 4,
                                },
                                "score": {"type": "number"},
                            },
                            "required": ["label", "bbox", "score"],
                        },
                    },
                },
                "required": ["bboxes"],
            },
        },
    },
    "required": ["results"],
}

detection_batch_response_format = dict(
    type="json_schema",
    json_schema=dict(
        name="detection_batch_schema",
        schema=detection_batch_response_schema,
    )
)

# 扩写 prompts：输出结构化 (scene/target/camera) 列表
prompts_expand_response_schema = {
    "type": "object",
    "properties": {
        "expand_descriptions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scene": {"type": "string"},
                    "target": {"type": "string"},
                    "camera": {"type": "string"},
                },
                "required": ["scene", "target", "camera"],
            },
            "minItems": 10,
        }
    },
    "required": ["expand_descriptions"],
}

prompts_expand_response_format = dict(
    type="json_schema",
    json_schema=dict(
        name="prompts_expand_schema",
        schema=prompts_expand_response_schema,
    ),
)

# 与 new_code 中 image_generate_response_format 对齐（扩写 prompts 输出）
image_generate_response_schema = prompts_expand_response_schema

image_generate_response_format = dict(
    type="json_schema",
    json_schema=dict(
        name="image_generate_schema",
        schema=image_generate_response_schema,
    ),
)


evaluation_summary_response_schema = {
    "type": "object",
    "properties": {
        "per_image_evaluation": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "image_idx": {"type": "integer"},
                    "image": {"type": "string"},
                    "source": {"type": "string"},
                    "qwen3-vl-8b": {
                        "type": "object",
                        "properties": {
                            "accuracy": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["accuracy", "reason"],
                    },
                    "rex-omni": {
                        "type": "object",
                        "properties": {
                            "accuracy": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["accuracy", "reason"],
                    },
                },
                "required": ["image_idx", "image", "source", "qwen3-vl-8b", "rex-omni"],
            },
        },
        "overall_conclusion": {"type": "string"},
        "model_results": {
            "type": "object",
            "properties": {
                "qwen3-vl-8b": {
                    "type": "object",
                    "properties": {
                        "accuracy": {"type": "string"},
                        "reason": {"type": "string"}
                    },
                    "required": ["accuracy", "reason"]
                },
                "rex-omni": {
                    "type": "object",
                    "properties": {
                        "accuracy": {"type": "string"},
                        "reason": {"type": "string"}
                    },
                    "required": ["accuracy", "reason"]
                }
            },
            "required": ["qwen3-vl-8b", "rex-omni"]
        },
        "recommendation": {
            "type": "string",
            "enum": ["qwen3-vl-8b", "rex-omni"]
            }
    },
    "required": [
        "per_image_evaluation",
        "overall_conclusion",
        "model_results",
        "recommendation"
    ]
}

evaluation_summary_response_format = dict(
    type="json_schema",
    json_schema=dict(
        name="evaluation_summary_schema",
        schema=evaluation_summary_response_schema,
    ),
)

solution_report_response_schema = {
    "type": "object",
    "properties": {
        "background_and_goals": {
            "type": "string",
            "description": "业务/技术背景与要达成的目标",
        },
        "model_training_plan": {
            "type": "string",
            "description": "模型训练方案：数据准备、基座选型、训练策略、算力与周期粗估等",
        },
        "annotation_data_format": {
            "type": "string",
            "description": "标注数据格式（如 COCO/YOLO，字段说明，示例结构）",
        },
        "annotation_howto": {
            "type": "string",
            "description": "如何标注：流程、工具建议、质检、难例与边界规则",
        },
        "data_volume_requirements": {
            "type": "string",
            "description": "数据量要求：训练/验证划分、场景覆盖、增量建议",
        },
        "evaluation_metrics": {
            "type": "string",
            "description": "离线/在线评估指标与达标建议",
        },
        "deployment_plan": {
            "type": "string",
            "description": "部署方案：推理服务、硬件、扩容、监控与回滚",
        },
        "performance_estimate": {
            "type": "string",
            "description": "性能预估：延迟、吞吐、资源占用、风险与优化方向",
        },
    },
    "required": [
        "background_and_goals",
        "model_training_plan",
        "annotation_data_format",
        "annotation_howto",
        "data_volume_requirements",
        "evaluation_metrics",
        "deployment_plan",
        "performance_estimate",
    ],
}

solution_report_response_format = dict(
    type="json_schema",
    json_schema=dict(
        name="solution_report_schema",
        schema=solution_report_response_schema,
    ),
)