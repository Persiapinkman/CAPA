# Dataset Card: planner_first_miss_transition_v1

## 目的

这是 `planner_stateful_retrieval_v2` 的训练采样视图，不是新的评估集。它针对首轮 RAG miss 后的 step 2 状态，提高 `re_question` 的训练出现概率，同时保留全部 1080 个原始状态和所有 guardrail。

## 触发证据

首个 v2 GRPO 臂在五步主场景把严格动作从 `61/80` 提到 `67/80`，但低于预注册的 `+0.10` 门槛。7 个 step-1 错误被修正，step 3 回退 1 个；最难的 step 2 贪心仍为 `5/16`。温度 0.7 下，step 2 正确样本已从 `26/64` 增到 `29/64`，表明方向正确但原始采样曝光不足。

## 组成

- 原始 v2 train：1080 个唯一 source step，全部保留，默认权重 1。
- `rag_double_miss_recovery#step1`：权重 2，防止聚焦 step 2 时丢失首次检索。
- `rag_double_miss_recovery#step2`：权重 4。
- `rag_single_miss_recovery#step2`：权重 4，提供同一“观察到 miss 后改写”的较短对照轨迹。
- 加权训练行总数：1584；唯一 source step 仍为 1080；实体仍为 72。

采样副本只是训练概率，不是新的独立实验单位。开发和测试始终使用未加权的 `planner_stateful_retrieval_v2`。

## 锁定项

奖励、SFTv3 初始化器、LoRA 结构、80 steps、LR `1e-5`、8 generations 和 temperature `0.7` 均与上一臂相同。唯一改变是训练状态的采样分布。

## 人工审阅

查看 `HUMAN_REVIEW.md`，重点确认默认权重为 1、只有上述三个 category-step 被覆盖，以及 evaluation 没有指向加权文件。
