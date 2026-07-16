# Human Review: planner_runtime_probe_curriculum_v1

这是人工审阅本训练集的首要入口。这里只展示 train 样例；dev 只用于门槛评估，test 在开发门通过前封存，不在本文展示内容。

## 训练目标

模型学习的不是工具输出内容，而是四类 Planner 决策：选对规范工具、保留关键参数、正确设置 `finish_after_tool`、依据前一步 observation 决定是否继续下一工具。

主实验刻意构造一组近邻对照：

| 场景 | 第一步 | 第二步 | 关键区别 |
|---|---|---|---|
| `qwen_probe_then_migration` | `qwen_detection`, `finish=false` | `migration_advisor`, `finish=true` | 探针只是证据，必须继续迁移分析 |
| `qwen_probe_only_contrast` | `qwen_detection`, `finish=true` | 无 | 用户只要探针结果，必须停止 |

这样训练是为了防止模型只看到“Qwen/图片”就机械选工具；它必须理解用户的终极目标和终止语义。

## 实际 Train 样例

### 两步探针后迁移

用户输入：

> 请处理先用Qwen看这张图有没有肩背包；探针完成后无论框数多少，继续给出市政泵站32号需求的低成本迁移方案，不要提前结束。

期望轨迹：

```json
[
  {
    "decision_type": "tool",
    "action": "qwen_detection",
    "action_input": {"label": "肩背包", "finish_after_tool": false}
  },
  {
    "decision_type": "tool",
    "action": "migration_advisor",
    "action_input": {
      "user_query": "保留市政泵站32号需求、肩背包和迁移语义",
      "use_image": true,
      "use_visual_probe": true,
      "finish_after_tool": true
    }
  }
]
```

训练 observation 明确说明候选框可能为 0，且视觉探针不等于已有专用模型或迁移结论。第二步不能依赖“检测到框”这一捷径。

### 单步探针对照

用户输入：

> 请处理这张图里是否有肩背包，明确用Qwen快速检测；只返回本次探针结果，不要继续做迁移方案。项目背景：市政泵站32号需求。

期望只有 `qwen_detection(label="肩背包", finish_after_tool=true)`。它与上一例共享实体和工具词，但终止目标相反。

### 完整评测而非快速探针

用户输入：

> 请处理以这张肩背包参考图扩增样本，对比 Qwen 和 Rex-Omni，输出误检漏检与效果评估报告。项目背景：市政泵站32号需求。

期望 `pipeline_eval(finish_after_tool=true)`，不能拆成单图 Qwen/Rex 探针。

### 私有知识与通用回答对照

“按公司资料给出现有内部模型推荐版本”期望 `rag_answer`；“不查内部知识库，给一般误报分析步骤”期望 `answerer(mode="direct")`。这组样例约束是否访问内部资产。

### 意图不完整时澄清

“评估一下这个模型在目标机器上的效果”既缺模型也缺平台，期望 `decision_type="clarify"`，不能编造参数或直接触发有副作用的 Adela/pipeline。

## 奖励为什么这样设计

- 动作匹配权重 `0.55`，参数 `0.25`，终止标志 `0.10`，其余用于 JSON 和禁用动作约束。
- 错动作最高只得 `0.20`，避免格式正确和局部参数正确掩盖路由错误。
- `strict_action_match=true`，显式要求 Qwen 时不能用 Rex-Omni 等价替代。
- 中间工具提前结束、跳过要求的视觉探针、最终工具不结束均有过程惩罚。

## 人工审阅清单

- query 的终极目标是否唯一，还是应该标成 `clarify`。
- 期望工具是否真由当前 Planner 控制，而非 Orchestrator 强制流转。
- `finish_after_tool` 是否符合“中间证据/最终结果”语义。
- `arg_contains` 是否只约束用户明确给出的实体，未引入虚构资产或平台。
- mock observation 是否只提供下一步所需状态，未泄露期望动作文字。
- 相邻对照是否只改变目标语义，不靠明显模板标记区分。
- 是否包含真实用户 query、回答、session/client 标识、内部资产 ID 或 RAG 文本；任一出现即拒绝。
- train/dev/test 是否在 entity、query、template 和 case ID 上隔离。

原始 train case 位于 `training/planner_grpo_seed_v1/cases/planner_runtime_probe_curriculum_v1_train_cases.jsonl`。人工修改后必须重新生成 manifest/hash，并重新做泄漏审计；不得查看或据此调参 test 内容。
