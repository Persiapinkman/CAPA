# Dataset Card: planner_multistep_grpo_hard_v1

## 研究问题

构造一个能力差分明确的 Planner challenge：`Qwen3.5-35B-A3B` 应能稳定解决，
而待训练的 `/raid/zkq/models/Qwen2.5-7B-Instruct` 在多步状态转移、typed 参数或
最终停止上仍明显失败。该数据用于确认场景是否值得做 GRPO，不用于证明真实工具
执行效果。

## 两阶段设计

| Split | 当前/目标规模 | Entity bundle | 用途 |
|---|---:|---:|---|
| calibration | 288 cases / 432 decisions | 24 x 12 | 可见；按完整 scenario family 筛选能力差分 |
| confirmation | 600 cases | 由准入 family 数决定 | 新实体、新模板、新 fixture family；整集一次性验收 |

calibration 可以查看逐 case 结果并修订场景。confirmation 只能在 family allowlist、
阈值和生成器冻结后生成；不得根据 confirmation 的单题输出删题。若 confirmation
未达到门槛，整个版本判失败，下一版使用新实体重新生成。

## 预注册能力差分门

Calibration family 准入：

- 35B deterministic 1x strict pass `>= 0.95`；
- raw Qwen2.5-7B strict pass `<= 0.70`；
- 两者差值 `>= 0.25`；
- 至少 6 个 family 准入，其中至少 3 个是两步 family；
- 任一 API error、timeout 或 empty decision 先判 run invalid，不计为模型失败。

Confirmation 整体验收：

- 600 cases、deterministic 3x；
- 35B pass-all `>= 0.95`；
- raw 7B pass-all `<= 0.60`；
- gap `>= 0.30`；
- 每个 family 的 35B pass-all `>= 0.90`、7B pass-all `<= 0.75`；
- 无 API error、timeout、empty decision。

## 场景矩阵

每个 calibration entity 含 12 个槽位：

- Qwen 三个 observation 反事实：低置信转 migration、高置信 end、技术失败重试；
- Rex-Omni 同样三个反事实；
- 有图且允许迁移顾问内部 probe；
- 有图但明确禁止 probe；
- 无图纯文本迁移；
- 完整 pipeline 评测；
- 仅 Flux 生成的副作用 guardrail；
- 检测/生图/迁移三义不明时 clarify。

同一 entity 下，Qwen/Rex 的三个 observation 分支共享完全相同的初始 query，只改变
mock observation 和第二步 gold。这是有意的反事实阻断设计，用来防止模型只背
`probe -> migration` 固定动作串。

## GRPO 适用性

- 所有决策均由 Planner policy 产生，不执行真实工具；
- gold action、参数、停止和 forbidden action 可由确定性 verifier 判定；
- wrong action reward 上限为 0.20；
- `strict_action_match=true`，点名 Qwen/Rex 不允许互换；
- `strict_argument_types=true`，字符串 `"true"/"false"` 不能冒充 JSON boolean；
- 启用 no-premature-stop、no-unexpected-repeat、no-skip-probe 和 final-finish 奖励；
- 最长两步，使用 deterministic mock observation。

## 数据来源与泄漏边界

数据只从 compound245 的聚合 failure taxonomy 抽象场景，不复制、改写或翻译原
query。实体、模板和 case ID 为合成；calibration 与现有仓库 case 的 case ID 和
规范化 query 零交叉。confirmation 使用独立实体、模板措辞和图片 fixture family。
具体 hash 与扫描结果见 `build_report.json`。

## 当前状态

- calibration 已生成并通过结构、reward、split 和 exact/normalized-query 自动审计；
- calibration 双臂 288-case 1x 已完成：35B `0.7292`，raw 7B `0.4271`，均无
  error/empty；整体分数不是准入依据，family gate 见 `calibration_family_gate.json`；
- family gate 仅准入 `qwen_uncertain_to_migration`（35B `0.9583` / base `0`）和
  `image_migration_internal_probe`（35B `1.0` / base `0`），不足预注册的 6 个，
  因此 V1 判失败且没有生成 confirmation；
- 独立 V2 calibration 已启用，不从 V1 逐题抽取；
- 未生成 train split，未启动训练；
- 人工业务审核尚未签字，见 `HUMAN_REVIEW.md`。

## 局限

图片仅用于路由上下文，评测不验证真实视觉识别质量。多个 entity 会共享一个 fixture
family，因此统计以 entity bundle 聚类并把 fixture family 作为 blocking factor；
不能把 432 个 decision 当作 432 个独立实验单位。
