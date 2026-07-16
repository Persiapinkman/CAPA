# Dataset Card: planner_runtime_routing_v1

## 研究问题

GRPO 是否能改善 Demo Agent 运行时真正交给 Planner 的动作：首步路由、参数、结束标志，以及视觉探针后的迁移决策？

## 数据规模

| Split | Entity groups | Cases | Steps |
|---|---:|---:|---:|
| train | 30 | 390 | 450 |
| dev | 8 | 104 | 120 |
| test | 16 | 208 | 240 |

## 场景

覆盖九个工具、`clarify`、`end`、Qwen/Rex 单图探针、完整评测、Flux、Adela，以及探针后继续迁移顾问的两步路径。
主对照是 `qwen_probe_then_migration` 与 `qwen_probe_only_contrast`：两者首步工具相同，但 `finish_after_tool` 必须不同。

## 来源与隐私

数据依据 `demo/sessions` 与 `demo/llm_debug` 的聚合动作分布、三轮 miss 结构和请求类型合成。
未复制原始用户 query、回答、客户端地址、session ID、模型资产 ID 或 RAG 文档内容。
所有项目名、模型名和实体均为合成值；图片只使用仓库 fixtures。

## 完整性

train/dev/test 的 entity、case ID、精确 query 与 template ID 均不重叠。
Test 在开发门预注册后保持封存，只有固定候选通过开发门才可打开一次。
同一实体下的不同场景相关，统计必须按 `entity_id` 聚类。

## 边界

这是路由策略评测，不执行 Flux、Adela 或完整 pipeline，因此不能证明外部服务效果。
RAG miss 后的强制改写与重试由编排器控制，不属于本数据的 GRPO 主张范围。

人工审阅顺序、实际 train 样例和拒绝条件见 `HUMAN_REVIEW.md`，哈希与分布见 `manifest.json`。
