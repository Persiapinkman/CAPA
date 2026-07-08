# Planner DPO Train Seed Review

## Summary

- Pair count: 120
- By error_type: `{"general_answer_over_rag": 40, "single_image_detection_over_pipeline": 40, "asset_inventory_over_migration": 40}`

## Review Instructions

- Mark each row as `approve`, `reject`, or `fix` in the CSV.
- Approve only if the chosen routing decision is clearly better for single-step Planner routing.
- Reject state-machine, slot-repair, or multi-step workflow cases; those belong in a different harness.

## Index

| # | Case | Error Type | Chosen > Rejected | Status |
|---:|---|---|---|---|
| 1 | `DPO-ANS-001` | `general_answer_over_rag` | `answerer` > `rag_answer` | approve |
| 2 | `DPO-ANS-002` | `general_answer_over_rag` | `answerer` > `rag_answer` | approve |
| 3 | `DPO-ANS-003` | `general_answer_over_rag` | `answerer` > `rag_answer` | approve |
| 4 | `DPO-ANS-004` | `general_answer_over_rag` | `answerer` > `rag_answer` | approve |
| 5 | `DPO-ANS-005` | `general_answer_over_rag` | `answerer` > `rag_answer` | reject |
| 6 | `DPO-ANS-006` | `general_answer_over_rag` | `answerer` > `rag_answer` | approve |
| 7 | `DPO-ANS-007` | `general_answer_over_rag` | `answerer` > `rag_answer` | approve |
| 8 | `DPO-ANS-008` | `general_answer_over_rag` | `answerer` > `rag_answer` | approve |
| 9 | `DPO-ANS-009` | `general_answer_over_rag` | `answerer` > `rag_answer` | approve |
| 10 | `DPO-ANS-010` | `general_answer_over_rag` | `answerer` > `rag_answer` | approve |
| 11 | `DPO-ANS-011` | `general_answer_over_rag` | `answerer` > `rag_answer` | approve |
| 12 | `DPO-ANS-012` | `general_answer_over_rag` | `answerer` > `rag_answer` | approve |
| 13 | `DPO-ANS-013` | `general_answer_over_rag` | `answerer` > `rag_answer` | approve |
| 14 | `DPO-ANS-014` | `general_answer_over_rag` | `answerer` > `rag_answer` | approve |
| 15 | `DPO-ANS-015` | `general_answer_over_rag` | `answerer` > `rag_answer` | approve |
| 16 | `DPO-ANS-016` | `general_answer_over_rag` | `answerer` > `rag_answer` | approve |
| 17 | `DPO-ANS-017` | `general_answer_over_rag` | `answerer` > `rag_answer` | approve |
| 18 | `DPO-ANS-018` | `general_answer_over_rag` | `answerer` > `rag_answer` | approve |
| 19 | `DPO-ANS-019` | `general_answer_over_rag` | `answerer` > `rag_answer` | approve |
| 20 | `DPO-ANS-020` | `general_answer_over_rag` | `answerer` > `rag_answer` | approve |
| 21 | `DPO-ANS-021` | `general_answer_over_rag` | `answerer` > `rag_answer` | approve |
| 22 | `DPO-ANS-022` | `general_answer_over_rag` | `answerer` > `rag_answer` | approve |
| 23 | `DPO-ANS-023` | `general_answer_over_rag` | `answerer` > `rag_answer` | reject |
| 24 | `DPO-ANS-024` | `general_answer_over_rag` | `answerer` > `rag_answer` | approve |
| 25 | `DPO-ANS-025` | `general_answer_over_rag` | `answerer` > `rag_answer` | approve |
| 26 | `DPO-ANS-026` | `general_answer_over_rag` | `answerer` > `rag_answer` | approve |
| 27 | `DPO-ANS-027` | `general_answer_over_rag` | `answerer` > `rag_answer` | reject |
| 28 | `DPO-ANS-028` | `general_answer_over_rag` | `answerer` > `rag_answer` | approve |
| 29 | `DPO-ANS-029` | `general_answer_over_rag` | `answerer` > `rag_answer` | reject |
| 30 | `DPO-ANS-030` | `general_answer_over_rag` | `answerer` > `rag_answer` | reject |
| 31 | `DPO-ANS-031` | `general_answer_over_rag` | `answerer` > `rag_answer` | approve |
| 32 | `DPO-ANS-032` | `general_answer_over_rag` | `answerer` > `rag_answer` | approve |
| 33 | `DPO-ANS-033` | `general_answer_over_rag` | `answerer` > `rag_answer` | approve |
| 34 | `DPO-ANS-034` | `general_answer_over_rag` | `answerer` > `rag_answer` | reject |
| 35 | `DPO-ANS-035` | `general_answer_over_rag` | `answerer` > `rag_answer` | reject |
| 36 | `DPO-ANS-036` | `general_answer_over_rag` | `answerer` > `rag_answer` | approve |
| 37 | `DPO-ANS-037` | `general_answer_over_rag` | `answerer` > `rag_answer` | approve |
| 38 | `DPO-ANS-038` | `general_answer_over_rag` | `answerer` > `rag_answer` | approve |
| 39 | `DPO-ANS-039` | `general_answer_over_rag` | `answerer` > `rag_answer` | approve |
| 40 | `DPO-ANS-040` | `general_answer_over_rag` | `answerer` > `rag_answer` | approve |
| 41 | `DPO-VIS-001` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 42 | `DPO-VIS-002` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 43 | `DPO-VIS-003` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 44 | `DPO-VIS-004` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 45 | `DPO-VIS-005` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 46 | `DPO-VIS-006` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 47 | `DPO-VIS-007` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 48 | `DPO-VIS-008` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 49 | `DPO-VIS-009` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 50 | `DPO-VIS-010` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 51 | `DPO-VIS-011` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 52 | `DPO-VIS-012` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 53 | `DPO-VIS-013` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 54 | `DPO-VIS-014` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 55 | `DPO-VIS-015` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 56 | `DPO-VIS-016` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 57 | `DPO-VIS-017` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 58 | `DPO-VIS-018` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 59 | `DPO-VIS-019` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 60 | `DPO-VIS-020` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 61 | `DPO-VIS-021` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 62 | `DPO-VIS-022` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 63 | `DPO-VIS-023` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 64 | `DPO-VIS-024` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 65 | `DPO-VIS-025` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 66 | `DPO-VIS-026` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 67 | `DPO-VIS-027` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 68 | `DPO-VIS-028` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 69 | `DPO-VIS-029` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 70 | `DPO-VIS-030` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 71 | `DPO-VIS-031` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 72 | `DPO-VIS-032` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 73 | `DPO-VIS-033` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 74 | `DPO-VIS-034` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 75 | `DPO-VIS-035` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 76 | `DPO-VIS-036` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 77 | `DPO-VIS-037` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 78 | `DPO-VIS-038` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 79 | `DPO-VIS-039` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 80 | `DPO-VIS-040` | `single_image_detection_over_pipeline` | `qwen_detection` > `pipeline_eval` | approve |
| 81 | `DPO-RAG-001` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 82 | `DPO-RAG-002` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 83 | `DPO-RAG-003` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 84 | `DPO-RAG-004` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 85 | `DPO-RAG-005` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 86 | `DPO-RAG-006` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 87 | `DPO-RAG-007` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 88 | `DPO-RAG-008` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 89 | `DPO-RAG-009` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 90 | `DPO-RAG-010` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 91 | `DPO-RAG-011` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 92 | `DPO-RAG-012` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 93 | `DPO-RAG-013` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 94 | `DPO-RAG-014` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 95 | `DPO-RAG-015` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 96 | `DPO-RAG-016` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 97 | `DPO-RAG-017` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 98 | `DPO-RAG-018` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 99 | `DPO-RAG-019` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 100 | `DPO-RAG-020` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 101 | `DPO-RAG-021` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 102 | `DPO-RAG-022` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 103 | `DPO-RAG-023` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 104 | `DPO-RAG-024` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 105 | `DPO-RAG-025` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 106 | `DPO-RAG-026` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 107 | `DPO-RAG-027` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 108 | `DPO-RAG-028` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 109 | `DPO-RAG-029` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 110 | `DPO-RAG-030` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 111 | `DPO-RAG-031` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 112 | `DPO-RAG-032` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 113 | `DPO-RAG-033` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 114 | `DPO-RAG-034` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 115 | `DPO-RAG-035` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 116 | `DPO-RAG-036` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 117 | `DPO-RAG-037` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 118 | `DPO-RAG-038` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 119 | `DPO-RAG-039` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |
| 120 | `DPO-RAG-040` | `asset_inventory_over_migration` | `rag_answer` > `migration_advisor` | approve |

## Samples

### 1. DPO-ANS-001 - 标注框质量建议

- Error type: `general_answer_over_rag`
- User query: 做目标检测数据集时，标注框质量一般要注意哪些问题？
- Rationale: 通用方法/概念问题不应过度路由到 RAG。

Chosen:

```json
{
  "thought": "用户询问通用概念或方法建议，不依赖内部资产库，也没有要求执行工具评测，应直接回答。",
  "decision_type": "tool",
  "action": "answerer",
  "action_input": {
    "mode": "direct",
    "finish_after_tool": true
  },
  "final_answer": ""
}
```

Rejected:

```json
{
  "thought": "用户问题涉及目标检测评估方法，可能属于企业规范或业务文档，应优先检索知识库。",
  "decision_type": "tool",
  "action": "rag_answer",
  "action_input": {
    "query": "做目标检测数据集时，标注框质量一般要注意哪些问题？",
    "finish_after_tool": true
  },
  "final_answer": ""
}
```

### 2. DPO-ANS-002 - IoU 概念解释

- Error type: `general_answer_over_rag`
- User query: IoU 是什么，目标检测里为什么常用它？
- Rationale: 通用方法/概念问题不应过度路由到 RAG。

Chosen:

```json
{
  "thought": "用户询问通用概念或方法建议，不依赖内部资产库，也没有要求执行工具评测，应直接回答。",
  "decision_type": "tool",
  "action": "answerer",
  "action_input": {
    "mode": "direct",
    "finish_after_tool": true
  },
  "final_answer": ""
}
```

Rejected:

```json
{
  "thought": "用户问题涉及目标检测评估方法，可能属于企业规范或业务文档，应优先检索知识库。",
  "decision_type": "tool",
  "action": "rag_answer",
  "action_input": {
    "query": "IoU 是什么，目标检测里为什么常用它？",
    "finish_after_tool": true
  },
  "final_answer": ""
}
```

### 3. DPO-ANS-003 - PR 曲线解释

- Error type: `general_answer_over_rag`
- User query: PR 曲线对目标检测评估有什么帮助？
- Rationale: 通用方法/概念问题不应过度路由到 RAG。

Chosen:

```json
{
  "thought": "用户询问通用概念或方法建议，不依赖内部资产库，也没有要求执行工具评测，应直接回答。",
  "decision_type": "tool",
  "action": "answerer",
  "action_input": {
    "mode": "direct",
    "finish_after_tool": true
  },
  "final_answer": ""
}
```

Rejected:

```json
{
  "thought": "用户问题涉及目标检测评估方法，可能属于企业规范或业务文档，应优先检索知识库。",
  "decision_type": "tool",
  "action": "rag_answer",
  "action_input": {
    "query": "PR 曲线对目标检测评估有什么帮助？",
    "finish_after_tool": true
  },
  "final_answer": ""
}
```

### 4. DPO-ANS-004 - 召回优先指标

- Error type: `general_answer_over_rag`
- User query: 如果业务更怕漏检，评估指标应该怎么侧重？
- Rationale: 通用方法/概念问题不应过度路由到 RAG。

Chosen:

```json
{
  "thought": "用户询问通用概念或方法建议，不依赖内部资产库，也没有要求执行工具评测，应直接回答。",
  "decision_type": "tool",
  "action": "answerer",
  "action_input": {
    "mode": "direct",
    "finish_after_tool": true
  },
  "final_answer": ""
}
```

Rejected:

```json
{
  "thought": "用户问题涉及目标检测评估方法，可能属于企业规范或业务文档，应优先检索知识库。",
  "decision_type": "tool",
  "action": "rag_answer",
  "action_input": {
    "query": "如果业务更怕漏检，评估指标应该怎么侧重？",
    "finish_after_tool": true
  },
  "final_answer": ""
}
```

### 5. DPO-ANS-005 - 置信度阈值选择

- Error type: `general_answer_over_rag`
- User query: 检测模型上线时置信度阈值通常怎么选？
- Rationale: 通用方法/概念问题不应过度路由到 RAG。

Chosen:

```json
{
  "thought": "用户询问通用概念或方法建议，不依赖内部资产库，也没有要求执行工具评测，应直接回答。",
  "decision_type": "tool",
  "action": "answerer",
  "action_input": {
    "mode": "direct",
    "finish_after_tool": true
  },
  "final_answer": ""
}
```

Rejected:

```json
{
  "thought": "用户问题涉及目标检测评估方法，可能属于企业规范或业务文档，应优先检索知识库。",
  "decision_type": "tool",
  "action": "rag_answer",
  "action_input": {
    "query": "检测模型上线时置信度阈值通常怎么选？",
    "finish_after_tool": true
  },
  "final_answer": ""
}
```

### 6. DPO-ANS-006 - 误报漏报归因

- Error type: `general_answer_over_rag`
- User query: 分析检测误报和漏报时，一般从哪些方面排查？
- Rationale: 通用方法/概念问题不应过度路由到 RAG。

Chosen:

```json
{
  "thought": "用户询问通用概念或方法建议，不依赖内部资产库，也没有要求执行工具评测，应直接回答。",
  "decision_type": "tool",
  "action": "answerer",
  "action_input": {
    "mode": "direct",
    "finish_after_tool": true
  },
  "final_answer": ""
}
```

Rejected:

```json
{
  "thought": "用户问题涉及目标检测评估方法，可能属于企业规范或业务文档，应优先检索知识库。",
  "decision_type": "tool",
  "action": "rag_answer",
  "action_input": {
    "query": "分析检测误报和漏报时，一般从哪些方面排查？",
    "finish_after_tool": true
  },
  "final_answer": ""
}
```

### 7. DPO-ANS-007 - 训练集划分

- Error type: `general_answer_over_rag`
- User query: 目标检测训练集、验证集、测试集一般怎么划分？
- Rationale: 通用方法/概念问题不应过度路由到 RAG。

Chosen:

```json
{
  "thought": "用户询问通用概念或方法建议，不依赖内部资产库，也没有要求执行工具评测，应直接回答。",
  "decision_type": "tool",
  "action": "answerer",
  "action_input": {
    "mode": "direct",
    "finish_after_tool": true
  },
  "final_answer": ""
}
```

Rejected:

```json
{
  "thought": "用户问题涉及目标检测评估方法，可能属于企业规范或业务文档，应优先检索知识库。",
  "decision_type": "tool",
  "action": "rag_answer",
  "action_input": {
    "query": "目标检测训练集、验证集、测试集一般怎么划分？",
    "finish_after_tool": true
  },
  "final_answer": ""
}
```

### 8. DPO-ANS-008 - 长尾类别处理

- Error type: `general_answer_over_rag`
- User query: 检测类别长尾严重时有哪些常见处理办法？
- Rationale: 通用方法/概念问题不应过度路由到 RAG。

Chosen:

```json
{
  "thought": "用户询问通用概念或方法建议，不依赖内部资产库，也没有要求执行工具评测，应直接回答。",
  "decision_type": "tool",
  "action": "answerer",
  "action_input": {
    "mode": "direct",
    "finish_after_tool": true
  },
  "final_answer": ""
}
```

Rejected:

```json
{
  "thought": "用户问题涉及目标检测评估方法，可能属于企业规范或业务文档，应优先检索知识库。",
  "decision_type": "tool",
  "action": "rag_answer",
  "action_input": {
    "query": "检测类别长尾严重时有哪些常见处理办法？",
    "finish_after_tool": true
  },
  "final_answer": ""
}
```

### 9. DPO-ANS-009 - 负样本作用

- Error type: `general_answer_over_rag`
- User query: 目标检测任务里为什么需要收集负样本？
- Rationale: 通用方法/概念问题不应过度路由到 RAG。

Chosen:

```json
{
  "thought": "用户询问通用概念或方法建议，不依赖内部资产库，也没有要求执行工具评测，应直接回答。",
  "decision_type": "tool",
  "action": "answerer",
  "action_input": {
    "mode": "direct",
    "finish_after_tool": true
  },
  "final_answer": ""
}
```

Rejected:

```json
{
  "thought": "用户问题涉及目标检测评估方法，可能属于企业规范或业务文档，应优先检索知识库。",
  "decision_type": "tool",
  "action": "rag_answer",
  "action_input": {
    "query": "目标检测任务里为什么需要收集负样本？",
    "finish_after_tool": true
  },
  "final_answer": ""
}
```

### 10. DPO-ANS-010 - 数据增强风险

- Error type: `general_answer_over_rag`
- User query: 用数据增强提升检测效果时有什么常见风险？
- Rationale: 通用方法/概念问题不应过度路由到 RAG。

Chosen:

```json
{
  "thought": "用户询问通用概念或方法建议，不依赖内部资产库，也没有要求执行工具评测，应直接回答。",
  "decision_type": "tool",
  "action": "answerer",
  "action_input": {
    "mode": "direct",
    "finish_after_tool": true
  },
  "final_answer": ""
}
```

Rejected:

```json
{
  "thought": "用户问题涉及目标检测评估方法，可能属于企业规范或业务文档，应优先检索知识库。",
  "decision_type": "tool",
  "action": "rag_answer",
  "action_input": {
    "query": "用数据增强提升检测效果时有什么常见风险？",
    "finish_after_tool": true
  },
  "final_answer": ""
}
```

### 11. DPO-ANS-011 - 开放词表检测

- Error type: `general_answer_over_rag`
- User query: open-vocabulary detection 和普通检测有什么区别？
- Rationale: 通用方法/概念问题不应过度路由到 RAG。

Chosen:

```json
{
  "thought": "用户询问通用概念或方法建议，不依赖内部资产库，也没有要求执行工具评测，应直接回答。",
  "decision_type": "tool",
  "action": "answerer",
  "action_input": {
    "mode": "direct",
    "finish_after_tool": true
  },
  "final_answer": ""
}
```

Rejected:

```json
{
  "thought": "用户问题涉及目标检测评估方法，可能属于企业规范或业务文档，应优先检索知识库。",
  "decision_type": "tool",
  "action": "rag_answer",
  "action_input": {
    "query": "open-vocabulary detection 和普通检测有什么区别？",
    "finish_after_tool": true
  },
  "final_answer": ""
}
```

### 12. DPO-ANS-012 - 标注一致性

- Error type: `general_answer_over_rag`
- User query: 多人标注同一批图片时，怎么检查标注一致性？
- Rationale: 通用方法/概念问题不应过度路由到 RAG。

Chosen:

```json
{
  "thought": "用户询问通用概念或方法建议，不依赖内部资产库，也没有要求执行工具评测，应直接回答。",
  "decision_type": "tool",
  "action": "answerer",
  "action_input": {
    "mode": "direct",
    "finish_after_tool": true
  },
  "final_answer": ""
}
```

Rejected:

```json
{
  "thought": "用户问题涉及目标检测评估方法，可能属于企业规范或业务文档，应优先检索知识库。",
  "decision_type": "tool",
  "action": "rag_answer",
  "action_input": {
    "query": "多人标注同一批图片时，怎么检查标注一致性？",
    "finish_after_tool": true
  },
  "final_answer": ""
}
```



## Human Review Notes

- `DPO-ANS-001`: `approve` - 人工审核：review 文档展示样例无问题，通用方法/概念问题可走 answerer。
- `DPO-ANS-002`: `approve` - 人工审核：review 文档展示样例无问题，通用方法/概念问题可走 answerer。
- `DPO-ANS-003`: `approve` - 人工审核：review 文档展示样例无问题，通用方法/概念问题可走 answerer。
- `DPO-ANS-004`: `approve` - 人工审核：review 文档展示样例无问题，通用方法/概念问题可走 answerer。
- `DPO-ANS-005`: `reject` - 人工审核：上线置信度阈值可能依赖企业内部上线规范/验收标准，rag_answer 并非明显更差，不适合作为 answerer > rag_answer 的 DPO 样本。
- `DPO-ANS-006`: `approve` - 人工审核：review 文档展示样例无问题，通用方法/概念问题可走 answerer。
- `DPO-ANS-007`: `approve` - 人工审核：review 文档展示样例无问题，通用方法/概念问题可走 answerer。
- `DPO-ANS-008`: `approve` - 人工审核：review 文档展示样例无问题，通用方法/概念问题可走 answerer。
- `DPO-ANS-009`: `approve` - 人工审核：review 文档展示样例无问题，通用方法/概念问题可走 answerer。
- `DPO-ANS-010`: `approve` - 人工审核：review 文档展示样例无问题，通用方法/概念问题可走 answerer。
- `DPO-ANS-011`: `approve` - 人工审核：review 文档展示样例无问题，通用方法/概念问题可走 answerer。
- `DPO-ANS-012`: `approve` - 人工审核：review 文档展示样例无问题，通用方法/概念问题可走 answerer。
- `DPO-ANS-013`: `approve` - 批量审核：通用检测方法/概念问题，chosen answerer 明显优于 rejected rag_answer。
- `DPO-ANS-014`: `approve` - 批量审核：通用检测方法/概念问题，chosen answerer 明显优于 rejected rag_answer。
- `DPO-ANS-015`: `approve` - 批量审核：通用检测方法/概念问题，chosen answerer 明显优于 rejected rag_answer。
- `DPO-ANS-016`: `approve` - 批量审核：通用检测方法/概念问题，chosen answerer 明显优于 rejected rag_answer。
- `DPO-ANS-017`: `approve` - 批量审核：通用检测方法/概念问题，chosen answerer 明显优于 rejected rag_answer。
- `DPO-ANS-018`: `approve` - 批量审核：通用检测方法/概念问题，chosen answerer 明显优于 rejected rag_answer。
- `DPO-ANS-019`: `approve` - 批量审核：通用检测方法/概念问题，chosen answerer 明显优于 rejected rag_answer。
- `DPO-ANS-020`: `approve` - 批量审核：通用检测方法/概念问题，chosen answerer 明显优于 rejected rag_answer。
- `DPO-ANS-021`: `approve` - 批量审核：通用检测方法/概念问题，chosen answerer 明显优于 rejected rag_answer。
- `DPO-ANS-022`: `approve` - 批量审核：通用检测方法/概念问题，chosen answerer 明显优于 rejected rag_answer。
- `DPO-ANS-023`: `reject` - 批量审核：该问题可能依赖企业内部规范、上线流程、质检或业务验收标准，rag_answer 并非明显更差。
- `DPO-ANS-024`: `approve` - 批量审核：通用检测方法/概念问题，chosen answerer 明显优于 rejected rag_answer。
- `DPO-ANS-025`: `approve` - 批量审核：通用检测方法/概念问题，chosen answerer 明显优于 rejected rag_answer。
- `DPO-ANS-026`: `approve` - 批量审核：通用检测方法/概念问题，chosen answerer 明显优于 rejected rag_answer。
- `DPO-ANS-027`: `reject` - 批量审核：该问题可能依赖企业内部规范、上线流程、质检或业务验收标准，rag_answer 并非明显更差。
- `DPO-ANS-028`: `approve` - 批量审核：通用检测方法/概念问题，chosen answerer 明显优于 rejected rag_answer。
- `DPO-ANS-029`: `reject` - 批量审核：该问题可能依赖企业内部规范、上线流程、质检或业务验收标准，rag_answer 并非明显更差。
- `DPO-ANS-030`: `reject` - 批量审核：该问题可能依赖企业内部规范、上线流程、质检或业务验收标准，rag_answer 并非明显更差。
- `DPO-ANS-031`: `approve` - 批量审核：通用检测方法/概念问题，chosen answerer 明显优于 rejected rag_answer。
- `DPO-ANS-032`: `approve` - 批量审核：通用检测方法/概念问题，chosen answerer 明显优于 rejected rag_answer。
- `DPO-ANS-033`: `approve` - 批量审核：通用检测方法/概念问题，chosen answerer 明显优于 rejected rag_answer。
- `DPO-ANS-034`: `reject` - 批量审核：该问题可能依赖企业内部规范、上线流程、质检或业务验收标准，rag_answer 并非明显更差。
- `DPO-ANS-035`: `reject` - 批量审核：该问题可能依赖企业内部规范、上线流程、质检或业务验收标准，rag_answer 并非明显更差。
- `DPO-ANS-036`: `approve` - 批量审核：通用检测方法/概念问题，chosen answerer 明显优于 rejected rag_answer。
- `DPO-ANS-037`: `approve` - 批量审核：通用检测方法/概念问题，chosen answerer 明显优于 rejected rag_answer。
- `DPO-ANS-038`: `approve` - 批量审核：通用检测方法/概念问题，chosen answerer 明显优于 rejected rag_answer。
- `DPO-ANS-039`: `approve` - 批量审核：通用检测方法/概念问题，chosen answerer 明显优于 rejected rag_answer。
- `DPO-ANS-040`: `approve` - 批量审核：通用检测方法/概念问题，chosen answerer 明显优于 rejected rag_answer。
- `DPO-VIS-001`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-002`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-003`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-004`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-005`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-006`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-007`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-008`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-009`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-010`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-011`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-012`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-013`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-014`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-015`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-016`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-017`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-018`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-019`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-020`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-021`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-022`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-023`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-024`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-025`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-026`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-027`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-028`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-029`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-030`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-031`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-032`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-033`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-034`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-035`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-036`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-037`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-038`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-039`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-VIS-040`: `approve` - 批量审核：单图检测请求，chosen detection 明显优于 rejected pipeline_eval。
- `DPO-RAG-001`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-002`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-003`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-004`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-005`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-006`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-007`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-008`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-009`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-010`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-011`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-012`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-013`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-014`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-015`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-016`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-017`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-018`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-019`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-020`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-021`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-022`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-023`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-024`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-025`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-026`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-027`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-028`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-029`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-030`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-031`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-032`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-033`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-034`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-035`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-036`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-037`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-038`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-039`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
- `DPO-RAG-040`: `approve` - 批量审核：纯历史资产盘点，chosen rag_answer 明显优于 rejected migration_advisor。
