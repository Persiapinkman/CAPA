# Qwen2.5-7B 多步工具路由 GRPO 技术方案 v3

## 文档状态

- 日期：2026-07-14
- 状态：`phase_b_v2_confirmation_pass_training_not_started`
- 基座：`/raid/zkq/models/Qwen2.5-7B-Instruct`
- 研究范围：方向一，Planner 多步工具路由；不包含方向二 Agent RL
- 目标参考：`Qwen3.5-35B-A3B`
- 已完成 raw base 与既有 GRPO seed43 candidate 的 compound245 3x；已据 bad
  case 完成两轮完整场景族 calibration，并生成独立 confirmation
- V2 的 GRPO support audit 与 600-case confirmation 3x 已完成，整版 gate 通过；
  未生成 train split，未启动训练

## 2026-07-14 阶段 B 执行更新

原方案中的 360-case hard-dev / 600-case sealed-test 已落实为更严格的两阶段
`family calibration -> frozen confirmation` 设计。模型结果只能筛完整场景族，不能
筛单题；confirmation 一旦生成即整版接受或拒绝。

| 数据版本 | Calibration | 35B strict | raw 7B strict | family gate |
|---|---:|---:|---:|---|
| V1 | 288 cases / 12 families | 72.92% | 42.71% | fail，仅 2 个 family 合格 |
| V2 | 384 cases / 12 families | 99.74% | 38.54% | pass，8 个两步 family 合格 |

V2 准入的 8 个 family 在 calibration 上合计 256 cases：35B 为
`255/256 = 99.61%`，raw 7B 为 `20/256 = 7.81%`。六个 observation 驱动的
Qwen/Rex 检测后迁移 family 中 35B 均为 100%、7B 均为 0%；两个高置信停止
反事实 family 中 35B 为 96.88%/100%，7B 为 43.75%/18.75%。四个单步直接迁移
family 因两模型均为 100% 被整族排除。

冻结 allowlist 后，使用新实体、新模板和新 fixture family 生成 600-case
confirmation（75 entity groups x 8 families，1,200 decisions）。它与 calibration
在 case ID、entity、template、规范化 query 和 fixture family 上均零交叉，且没有
按模型结果删除任何 confirmation case。raw 7B 的 8-sample GRPO support audit
已经完成。confirmation 首次运行中，35B 出现 4 次 384-token 首 completion 截断
和 3 个 timeout；该尝试在查看 correctness 分数前整批作废并保留到
`invalid_t384`。随后 768-token 重跑仍有 1 次首 completion 截断，也在计分前
整批作废到 `invalid_t768`。最终运行协议统一为 2,048 tokens / 300 秒，两模型从
case 1 完整重跑 600 x 3；数据、gold、verifier 和 allowlist 始终不变。训练仍未
获准，也未启动。

2,048-token 正式运行仍观察到少量非终止的超长首答，但部署时的 parser retry
均成功。由于任何有限上限都不能排除这种行为，最终有效性与真实 Planner 合同
一致：recovered first-call truncation 单独作为稳定性缺陷报告；retry 仍截断、最终
error 或空 decision 才判整轮无效。该规则对两模型完全相同。

最终 confirmation 结果：35B 三轮 strict pass 为 593/594/591，pass-all
`583/600 = 97.1667%`；raw 7B 为 146/147/147，pass-all
`136/600 = 22.6667%`；pass-all gap 为 74.5pp。75-cluster paired bootstrap 的
gap 95% CI 为 69.67–79.33pp。8 个 family 均通过 35B >=90%、7B <=75% 的门，
整版 confirmation gate 为 pass。35B 三轮累计 5 次 recovered first-call
truncation，两个模型均无最终 error、空 decision 或 retry truncation。

Support audit 共覆盖 128 个 calibration step、1,024 个随机 completion。把已饱和
step1 与 evaluation-only stop guardrail 混合统计时，严格全局门失败：reward
variance 54.69%、usable support 44.53%、near-exact support 57.03%、饱和率
43.75%。该失败结果被保留，不能据此宣称 raw base 可直接启动 GRPO。

按 B2 定义的角色边界，只看六个 primary family 的 48 个待学习 step2，则 reward
variance 为 91.67%、usable support 83.33%、gold action support 85.42%、平均
distinct valid action 1.8125、饱和率 0%，通过状态转移的 GRPO 可探索性门。但
full action+args exact support 只有 2/48（4.17%）。因此执行结论是：场景适合
dense-reward GRPO，raw base 尚不适合直接做 strict GRPO；阶段 C 必须先做短 SFT，
并在 SFT 后复验 strict full-decision support，未过门不得启动 GRPO。

## 2026-07-14 阶段 A 实际结果

| Arm | 每轮通过 | pass-all | mean score |
|---|---:|---:|---:|
| Raw Qwen2.5-7B | 112 / 112 / 112 | 0.457143 | 0.910996 |
| SFTv3 + runtime-probe GRPO seed43 | 117 / 117 / 117 | 0.477551 | 0.929784 |

Candidate 相对 base 增加 5 个 strict pass，未出现 strict regression，但
`probe_then_migration` 和 `migration_feasibility_with_image` 仍均为 0% strict
pass，距离历史 35B 的 235/245 仍差 118 case。完整报告见
`reports/QWEN25_7B_COMPOUND245_BASE_VS_GRPO_SEED43_3X_20260714.md`。

血缘审计显示：GRPO 阶段的 540 条新 train cases 与 compound245 的 case_id 和
规范化 query 均零交叉；但 candidate 的 SFTv3 initializer train split 与
compound245 有 69 条重合。因此该结果只能证明“新 GRPO 数据零交叉”，不能证明
candidate 的完整训练血缘零交叉，也不能把 raw-base 到 candidate 的全部增量归因
于 GRPO。

## 1. 结论先行

仓库中确实存在对应的 245-case 多步路由评测集：

`training/planner_grpo_seed_v1/cases/planner_grpo_compound245_eval_cases.jsonl`

Qwen2.5-7B-Instruct 原始基座已经完成三次确定性评测，每轮均严格通过
`112/245 = 45.7143%`，3x 平均 verifier score 为 `0.910996`。历史 35B 在三次
确定性重复下达到 `235/245 = 95.9184%` 的 pass-all，平均单次通过率为
`96.4626%`。

后续不应直接把 245 条题目拿来训练。它与历史训练资产重合，只能作为
legacy regression。阶段 A 的两臂 3x 已完成，下一阶段改为：

1. 只从 245 的聚合 bad-case taxonomy 抽象状态变量、边界和 verifier 规则，
   不复制或改写原 query。
2. 先构建可见的 `hard-dev` 双生集，在其上确认场景确实具有 GRPO 所需的
   正确动作采样支持和组内 reward 方差。
3. 按冻结后的同一因子矩阵，使用完全独立的实体、模板、图片 fixture 和
   observation seed 生成 `sealed-test`；完成泄漏审计后封存。
4. 对 `/raid/zkq/models/Qwen2.5-7B-Instruct` 做 3x base 评测：hard-dev 结果
   可见，sealed-test 的逐 case 输出和分层指标进入 escrow，训练期间不读取。
5. 之后才构建独立 train split、SFT initializer 和 GRPO；模型冻结后一次性
   解封 base/candidate/35B 的配对结果。checkpoint 只由 hard-dev 选择。

如果 base 的 sealed 逐 case bad case 会被用于改数据、reward 或训练，则该 split
必须立即降级为 dev，并另建一个从未读取的 final sealed-test，不能继续称为 sealed。

## 2. 任务边界

本研究学习 Planner 在每一步输出的结构化 JSON：

- `decision_type`
- `action`
- 必要的 `action_input`
- `finish_after_tool`
- 基于前一步 observation 的继续、停止、迁移或澄清决策

多步轨迹使用确定性的 mock/replay observation，不执行真实 Flux、检测、RAG、
pipeline 或迁移工具。它是多步工具路由训练，但仍不是 Agent RL：不优化真实
工具结果、调用成本、环境延迟或长轨迹业务成功率。

## 3. 已找到的 245-case 评测与历史结果

### 3.1 数据集事实

| 项目 | 数值 |
|---|---:|
| case 数 | 245 |
| 单步 case | 184 |
| 两步 case | 61 |
| 总 Planner decision | 306 |
| 最大 gold 步数 | 2 |
| 文件 SHA256 | `cd0f03acdb15f586386c50936b4f997b7b1226732b37eb1a3409076ca2e02388` |

类别分布：

| category | cases |
|---|---:|
| `single_image_probe` | 74 |
| `probe_then_migration` | 61 |
| `migration_feasibility_with_image` | 51 |
| `migration_feasibility` | 38 |
| `historical_asset_qa` | 7 |
| `general_answer` | 6 |
| `full_detection_eval` | 5 |
| `adela_eval` | 2 |
| `intent_ambiguity` | 1 |

### 3.2 同一 compound245 协议的模型结果

| 模型 | repeats | pass-all | pass-rate mean | mean score | 备注 |
|---|---:|---:|---:|---:|---|
| Qwen2.5-7B-Instruct | 3 | 0.457143 | 0.457143 | 0.910996 | formal 3x，112/245，三轮 pass 集合一致 |
| SFTv3 + runtime-probe GRPO seed43 | 3 | 0.477551 | 0.477551 | 0.929784 | formal 3x；完整血缘与 245 有 69 条 SFT 重合 |
| Qwen3.5-4B | 3 | 0.8041 | 0.8517 | — | 纯文本迁移较弱；有 2 次 rollout timeout |
| Qwen3.5-9B | 3 | 0.6933 | 0.8914 | — | 单次均值高，但跨 repeat 不稳定 |
| Qwen3.5-35B-A3B | 3 | 0.959184 | 0.964626 | 0.991211 | 235/245 三次全过 |

35B 的残余失败共 10 条：9 条 `probe_then_migration`，1 条
`general_answer`。其 `probe_then_migration` 类 pass-all 为 `0.852459`，
其余除 `general_answer=0.833333` 外均为 `1.0`。

以下结果不进入该表：Qwen3-32B 的所谓 `compound245` 实际跑了 313 case；
后来一次本地 Qwen3.5-9B V100 FP16 服务出现乱码、非 JSON 和大量 timeout，
已被标记为 invalid，不能解释为模型质量。

主要证据位置：

- Qwen2.5-7B formal 3x：`/raid/zkq/artifacts/CAPA/evals/20260714_compound245_base_vs_grpo3x/base/qwen25_7b_base_compound245_t0_3x_20260714_aggregate.json`
- 两臂正式报告：`reports/QWEN25_7B_COMPOUND245_BASE_VS_GRPO_SEED43_3X_20260714.md`
- 历史模型表：`experiments/archive/qwen35_legacy_EXPERIMENT_LOG_20260710.md`
- 35B 机器可读记录：`experiments/archive/qwen35_legacy_manifest_20260710.jsonl`

### 3.3 3x bad-case 结构与扩增优先级

两臂均没有空 decision、解析错误或 timeout。以下为三轮平均 failure count；
同一 case 可以命中多个 failure：

| 失败信号 | Raw base | GRPO seed43 | 变化 |
|---|---:|---:|---:|
| 最终 `finish_after_tool` 应为 true、实际为 false | 113.7 | 115.7 | +2.0 |
| `use_visual_probe=true`、实际为 false | 32.3 | 13.7 | -18.7 |
| 缺失 `use_image=true` | 19.3 | 11.0 | -8.3 |
| 缺失 `use_visual_probe=true` | 19.3 | 11.0 | -8.3 |
| 应转 `migration_advisor`、却重复 `qwen_detection` | 19.3 | 11.0 | -8.3 |
| step1 false 输出成字符串 `"false"` | 9.0 | 0.0 | -9.0 |

类别上，`single_image_probe` 已是 100%，但两臂的 `probe_then_migration` 和
`migration_feasibility_with_image` strict pass 均为 0%。因此 bad-case 扩增
优先级固定为：最终停止语义 > observation 驱动的继续/迁移分叉 > 图片迁移参数
和 typed boolean > pipeline/Flux、general/RAG、clarify/副作用 guardrail。
普通单图探测只作为最小对照和防遗忘样本，不作为主要扩增方向。

## 4. 证据边界和必须修正的评测问题

1. `compound245` 与 `planner_grpo_train_cases.jsonl` 前 245 行逐对象完全相同，
   不是 held-out test。
2. 历史 focused 数据 154 case 中有 86 case 与 compound245 重合；不能在该
   数据上训练后再把 245 的提升解释为泛化。
3. 245 的 61 个两步 case 会产生 122 个 step，但独立实验单位仍是 case，
   不能把 step 当独立样本扩大显著性。
4. legacy verifier 为保持历史可比性必须冻结；新的 verifier v2 另行严格要求
   JSON boolean 类型，不能把字符串 `"false"` 当作合法布尔值。
5. 旧 reward 默认允许 Qwen/Rex 检测等价。新数据中，用户点名模型时必须
   strict named-action；只有未点名时才允许等价。
6. 当前 Demo 默认禁用 Adela，而 legacy245 含 2 个 Adela case。legacy 轨按
   历史 action space 独立复现；新的可部署主轨不启用 Adela，两轨结果不合并。

已有的 runtime route-GRPO 三种子实验也必须纳入设计：它把明确的
Qwen probe -> migration 完整 case 动作率平均提高 `+0.3333`，总体动作率提高
`+0.1333`，但 seed44 新增两次错误 Flux 调用，导致错误副作用动作均值
`+0.6667`，预注册安全门失败。新的数据集必须同时训练
`pipeline_eval <-> Flux` 和 `clarify/unsupported <-> side-effecting tool` 边界，
不能只强化固定的 probe -> migration 序列。

## 5. “达到 35B”的操作定义

采用两层目标，避免在一个已污染的 245 集上宣布成功。

### 5.1 Legacy 工程目标

在冻结的 compound245、legacy prompt/schema/verifier 上：

- 3x `pass_all_runs_rate >= 235/245 = 0.959184`；
- `pass_rate_mean >= 0.96`；
- `probe_then_migration` pass-all 不低于 35B 的 `0.852459`；
- 不通过放宽 verifier、增加 alias 或修改 gold 达标。

这是历史能力对齐目标，不是泛化结论。

### 5.2 主研究目标

在新建 sealed test 上，以人工/规则 gold 为准，比较冻结的 7B candidate 与
同协议 35B reference：

- primary：完整 case 多步 strict pass-all；
- 非劣界：`7B - 35B >= -0.03`，配对 entity-clustered 单侧 95% CI 下界
  大于 `-0.03`；
- 关键类别相对 35B 回退不超过 5 个百分点；
- JSON valid >= 99.5%，typed argument、exact stop 和 repeat agreement 单独报告；
- 错误副作用调用不多于 initializer 和 35B，critical violation 必须为 0。

## 6. 阶段 A：compound245 两臂 3x 回归（已完成）

### A0. 冻结协议

本阶段已按冻结协议记录：

- model/tokenizer revision 和目录清单 hash；
- git commit、dirty state；
- system prompt、工具 schema、response schema hash；
- compound245 路径、245 行断言和 SHA256；
- action space、Adela 开关和 evaluator version；
- serving 参数、GPU、dtype、attention backend、超时和完整命令。

### A1. 评测矩阵

| arm | 结果 | 作用 |
|---|---:|---|
| raw Qwen2.5-7B-Instruct | 112/245 pass-all | 本轮 T0 base |
| SFTv3 + runtime route-GRPO seed43 | 117/245 pass-all | 读取 245 前由既有 dev 证据固定的 candidate |

两臂均完成 3x，pass 集合跨 repeat 一致。35B 只复用历史 235/245 作为 legacy
参考，没有与本轮结果混写为同一新 sealed-test 证据。

### A2. 固定生成协议

- `temperature=0`
- `top_p=1`
- `do_sample=false`
- `seed=42`
- 3 deterministic repeats
- `max_steps=3`
- formal step timeout 180 秒；服务异常导致的 timeout 先判 invalid run
- 显式传入 compound245 文件，禁止依赖脚本默认的 313-case train 文件
- 保存 raw completion、逐 run reward、aggregate、全量 case audit 和 failed CSV

本地 7B 使用 fp16 + SDPA 的本地服务；contract smoke 不并入 formal run。
compound245 使用冻结的 legacy verifier 保持历史可比性，bad-case 分析另按严格
JSON 类型和状态转移字段汇总。

### A3. 报告指标

Primary unit 是 `case_id`。必须报告：

- 3x pass-all、pass-rate mean/stdev、case-macro score；
- 类别级 pass-all 和样本数；
- step1/step2 action、argument、finish accuracy；
- action transition matrix，特别是 detection -> migration；
- JSON valid、boolean type、missing arg、extra text、exact stop；
- repeat agreement、timeout/error、延迟、token 和峰值显存；
- 相对 raw base、SFTv3、35B 的 paired case delta。

阶段 A 产物见
`reports/QWEN25_7B_COMPOUND245_BASE_VS_GRPO_SEED43_3X_20260714.md`。下一阶段
只按上述残余 failure family 扩增，不重复扩写已饱和的普通单图探测类别。

## 7. 阶段 B：bad-case 驱动的 sealed-test 扩增与 base 评测

实际数据版本为 `planner_multistep_grpo_hard_v1`（探索失败）和
`planner_multistep_grpo_hard_v2`（当前冻结版本）。本阶段只构建评测/support
资产并评测 raw base/reference，不构建 train split、不启动 SFT/GRPO。

### B0. 研究问题与 sealed 边界

要验证的不是“模型能否复述 245”，而是 7B 是否能在新实体和新表达上，根据
当前 observation 做出正确的下一步工具路由、typed 参数和停止决策。245 只提供
failure taxonomy；原 query、答案、case ID、模板骨架和图片 fixture 均不得进入
新集。

一个可持续使用的 sealed test 不能在 base 评测后打开逐 case bad case 再指导
训练。因此实际采用“可见 family calibration -> 冻结 all-or-nothing confirmation”：

| 资产 | 规模 | 实验单位 | 可见性与用途 |
|---|---:|---:|---|
| V1 calibration | 288 | 24 entity x 12 family | 可见探索；family gate 失败，不生成 confirmation |
| V2 calibration | 384 | 32 entity x 12 family | 可见；完整 family 筛选和 GRPO support audit |
| V2 confirmation | 600 | 75 entity x 8 frozen family | 不逐题筛选；base/35B 3x 整版接受或拒绝 |
| `compound245` | 245 | 245 case | 已暴露 legacy regression，不参与新集筛选或模型选择 |

case 是评分单位，`entity_id` 是 bootstrap cluster；多步 case 内的 step 是嵌套
观测，不能当作独立样本放大样本量。600 cases 固定为 75 clusters x 8 families。
预先完成的 beta-binomial clustered simulation 显示：75 x 8、ICC=0.10 时，若
真实 35B/base 为 0.96/0.55，整套门通过概率约 83.3%；若为 0.97/0.55，则约
95.2%。manifest/hash 冻结后不得按 confirmation 模型结果调整样本量。

当前 confirmation cases/gold 位于工作区以便启用评测，因此它是“冻结、无逐题
筛选的 confirmation”，不是对训练操作者不可见的密码学 sealed final test。若
后续读取它的逐题 bad case 来改训练，必须将它降级为 test-visible，并另建外部
隔离的 final sealed test。

### B1. 从 3x bad case 冻结因子矩阵

V2 calibration 的 12 个 family 先整族筛选。准入后，confirmation 只保留 8 个
两步 family；实体词表、template family、图片 fixture、observation 文本和 seed
与 calibration 完全隔离：

| 场景族 | 每 entity | calibration | confirmation | 主要针对的 failure |
|---|---:|---:|---:|---|
| Qwen x 框波动/空结果/域偏移 -> migration | 3 | 96 | 225 | observation 驱动迁移、图片参数、最终停止 |
| Rex x 框波动/空结果/域偏移 -> migration | 3 | 96 | 225 | 点名 Rex、observation 驱动迁移、最终停止 |
| Qwen/Rex 高置信 -> end | 2 | 64 | 150 | 防止学成固定 detection -> migration 串 |
| direct image migration calibration probes | 4 | 128 | 0 | 7B 已 100%，整族排除 |
| 合计 | 12/8 | 384 | 600 | — |

以下因素在 entity block 内做平衡或最小对照，而不是各自复制一批同义句：

- observation outcome：高置信、低置信、空结果、工具失败、超时；
- 用户目标：根据检测 observation 继续迁移或直接停止；
- 图片状态：有图且先 probe，迁移顾问必须复用当前图片并内部视觉核验；
- 模型约束：明确点名 Qwen 或 Rex；
- 停止语义：中间 step 必须 false，最终 step 必须 true，且必须是 JSON boolean。

普通 `single_image_probe` 和 direct image migration 已在 base 饱和，不进入
confirmation 主体；不能按 failure 频数机械扩写。当前部署主轨禁用 Adela，
所以 Adela 不进入 confirmation 分数；如需兼容性验证，另建 legacy slice。

### B2. “适合 GRPO”的准入门

场景进入冻结矩阵前必须同时满足：

1. **Policy-owned**：决策确实由 Planner 模型产生，不是 orchestrator 的硬编码分支。
2. **可验证**：gold action、typed args、finish、forbidden action 和 observation
   transition 可由确定性 verifier 判断，不依赖 35B/LLM judge。
3. **可探索**：在 hard-dev/support pool 上由 raw base 每题采样 8 次
   (`temperature=0.7, top_p=0.9`)；至少 80% hard prompts 出现过一次 gold
   action，至少 80% prompt 组内 reward 方差非零。
4. **不饱和也非不可达**：主要学习场景不能全部 8/8 正确或 0/8 正确；平均每题
   至少 1.4 个 distinct valid action。饱和 case 只作 guardrail，不占主要训练配额。
5. **奖励方向正确**：wrong action 总 reward 上限 0.20；错误 Flux/pipeline 等
   副作用动作 reward=0；格式分不能补偿动作、参数或停止错误。
6. **必须读状态**：同一初始请求至少有两个 observation 反事实分支，且导向不同
   后继决策，避免模型只背 `probe -> migration` 固定动作串。
7. **轨迹可控**：使用 deterministic mock/replay observation，最多 3 个 Planner
   step，不执行真实工具；prompt/completion 长度落在既定训练上限内。

这些支持性统计只允许在 calibration 或独立 support pool 上计算，绝不在 confirmation
上筛题。某一场景不过门时，修 verifier/场景或先补 SFT 支持；不能靠增加 GRPO
步数解决零支持。

### B3. 构造、人工审核与防泄漏

执行顺序是先构造 calibration、完成完整 family gate 与 B2，再冻结 schema、
allowlist 和配额，最后用独立资源生成 confirmation。两者都必须：

- 包含 `case_id`、`entity_bundle_id`、`scenario_family`、`template_family_id`、
  `expected_decisions`、typed required args、`mock_observations`、forbidden actions；
- 对 compound245、所有既有 SFT/DPO/GRPO 数据和 calibration 做 case ID、exact/
  normalized query、模板、实体和 fixture 审计；n-gram/embedding 近邻作为后续
  人审辅助，不可用模型正确性做筛选；
- 100% 人工审核 gold、自然性和可判定性；两步轨迹及副作用边界双人复核；
- 自动跑 schema、transition、reward 单元测试，确保每个反事实分支可达且唯一；
- dataset card 记录来源、生成模型、审核记录、hash、配额和 split lineage。

当前工作区保存了 confirmation cases/gold 以启用评测，并用 `DO_NOT_TRAIN.md`
明确禁止训练。自动 exact/normalized query、entity、template、fixture 隔离已通过；
业务人员 100% gold 审核仍待签字。未来真正不可见的 final sealed 应放在训练进程
不可挂载的外部 artifact/escrow 路径。

### B4. 35B 和外部调用的角色

所有 35B/9B、RAG、检测或其它非本地 Qwen2.5 调用前统一：

```bash
source /raid/zkq/projects/CAPA/init_env.sh
```

35B 在 calibration 上只参与完整 family 的能力门，不允许筛单题；confirmation
按冻结 allowlist 和新实体一次性生成。35B 不能单独决定 gold，正式 gold 由确定性
因子矩阵/verifier 和人工 adjudication 确定。环境变量值、token 和 endpoint 凭据
不得写入命令记录、Git 或 raw report。

### B5. Raw base 评测协议

固定模型：`/raid/zkq/models/Qwen2.5-7B-Instruct`。在数据和 evaluator hash 冻结后：

- deterministic 3x：`temperature=0, top_p=1, do_sample=false, seed=42`；
- `max_steps=3`, `max_tokens=2048`，timeout 300 秒，本地 fp16 + SDPA；该值来自
  `protocol_amendment_t2048.json`，384/768-token 尝试均不进入结果；
- 使用当前部署 action space，confirmation 主轨 `CAPA_ENABLE_ADELA=0`；
- smoke 只验证合同，formal run 必须从 case 1 重跑；
- 保存 JSON valid、strict action/args/finish、完整 case pass-all、transition、
  repeat agreement、latency/error 和错误副作用计数。

本阶段公开 calibration 和 confirmation 的 aggregate/family 指标，以验证用户要求的
“35B >=95%、raw 7B 明显较差”。逐 case confirmation bad case 不得用于改数据、
reward、checkpoint 或训练超参；若后续确实读取并使用，应把该集降级为
`test-visible-v2`，并在最终泛化声明前另建不可见 final sealed。

### B6. 预注册指标和阶段完成标准

Primary 是 75 个 confirmation entity bundle 上的完整 case strict pass-all；同时报告
bundle-clustered CI，不能把 600 case 或多步 decision 当成完全独立观测。Secondary：

- scenario-family macro pass-all；
- step1/step2 action、typed args、exact finish 和 transition accuracy；
- JSON valid、repeat agreement、wrong side-effect/critical violation；
- 相对 raw base、同一 SFT initializer 和 35B 的 paired bundle delta。

阶段 B 只有在以下产物齐全时完成：calibration/confirmation manifest 与 hash、100% review
记录、泄漏报告、verifier 单测、GRPO-support 报告、评测预注册和 base 3x run record。
自动生成、hash、泄漏审计、23 项 verifier/data/eval 单测和 GRPO-support 报告已
完成；base/35B confirmation 3x 和整版 gate 已完成，业务人员 100% gold review
仍待签字。

## 8. 阶段 C：SFT initializer 与 GRPO

### C0. Sampling-support gate

复用 B2 已冻结的定义，在独立 train support subset 上复验；不得读取 sealed-test。
每个 prompt 采样 8 个 completion：

- JSON-valid >= 99%；
- 至少 80% hard prompts 出现过 gold action；
- 至少 80% hard prompts 的组内 reward variance 非零；
- 平均 distinct valid action >= 1.4；
- 错误副作用动作处于预设上限内。

本轮 raw base 的 primary step2 已有 gold-action 与 dense reward support，但
full-decision exact support 只有 4.17%，因此短 SFT 不再是可选项。SFT 后必须在
相同 support pool 上复验；若 strict action+typed args+finish support 仍不足，则
停止 GRPO，先修 prompt/data，而不是增加训练步数。

### C1. 安全约束 SFTv4

- 从 `/raid/zkq/models/Qwen2.5-7B-Instruct` 开始；
- native Qwen ChatML；
- target 是一个 canonical JSON 后立即 EOS；
- fp16、SDPA、LoRA r16/alpha32、q/k/v/o；
- 0.5–1 epoch，小数据早停；
- raw base、SFTv4、SFTv4+GRPO 三臂始终分别报告。

SFT 的作用是建立 typed JSON、停止和安全边界支持；最终必须证明 GRPO 相对同一
SFT initializer 的独立增益。

### C2. GRPO 主配方

复用本机已经跑通的 TRL/Transformers 路线：

| 参数 | 初始固定值 |
|---|---|
| precision / attention | fp16 / SDPA |
| update | LoRA `q_proj,k_proj,v_proj,o_proj`, r16, alpha32 |
| GPUs | 每个 seed 4 x V100-32GB |
| max prompt / completion | 6144 / 256 |
| generations per prompt | 8 |
| sampling | temperature 0.7, top_p 0.9 |
| optimizer LR | `1e-5` |
| loss | DR-GRPO, beta 0 |
| effective batch | 16 |
| candidate steps | 20/40/80，按预注册 dev 规则选一个 |
| seeds | 42/43/44 |

训练 rollout 先不用 vLLM，继续使用本机已验证的 Transformers generation，
并保持 `remove_invalid_values` 与 `renormalize_logits`。不使用 eager attention；
本机已有记录显示它会导致 Qwen2.5-7B 输出异常。

实现上先把每条轨迹展开为 state-conditioned step：step2 prompt 必须包含前一步
动作和 mock observation。GRPO 做 step-level policy update，但训练采样按 case/
entity 平衡；checkpoint 选择和最终结论只看完整轨迹 pass-all。只有当 step2
分数提高但完整 case 不提高时，才进入 trajectory-level GRPO 的下一实验臂。

### C3. Reward v2

Reward 必须版本化并在训练前单元测试：

- strict action 是主项；错误 action 总奖励上限 `0.20`；
- required args 严格校验存在性、类型和值；
- `finish_after_tool` 必须是 JSON boolean；
- step1 continuation、no skip probe、no unexpected repeat、final finish 单独计分；
- 错误 Flux/pipeline 等副作用动作 reward 为 0，并计 safety violation；
- named Qwen/Rex 不互相 alias；
- 格式奖励只占小比例，不能让错误动作靠 JSON 格式获得高 reward；
- 在线 reward 不调用 35B judge，避免非确定性和 endpoint 漂移。

## 9. 顺序设计和决策门

### 9.1 执行顺序

1. 阶段 A 已完成：冻结 compound245 3x bad-case taxonomy，不再读原 query 做扩增。
2. 按 B1 构建 360-case hard-dev，完成 verifier 单测和 100% 人审。
3. 在 hard-dev/support pool 上运行 B2 GRPO-suitability gate；不过门的场景先修订。
4. 冻结 schema、12-slot 因子矩阵、配额、评测指标和模型选择规则。
5. 用独立实体/模板/fixture/seed 生成 600-case sealed-test，做全量泄漏审计并封存。
6. raw Qwen2.5-7B 做 hard-dev 3x 开放评测及 sealed 3x escrow 评测；此时不训练。
7. 依据 hard-dev 构建独立 train split，复验 support，必要时训练并冻结 SFTv4。
8. seed42 做 5-step contract smoke，再按预注册上限 screen；过门后跑 seeds43/44。
9. candidate/checkpoint 冻结后，对 sealed 一次性运行 candidate 与 35B，再解封三臂结果。
10. 最终再跑 compound245 3x regression；245 和 sealed 均不能选择 checkpoint。

### 9.2 开发门

必须同时满足：

- 三种子平均完整 case pass-all 相对 SFTv4 为正，entity-clustered 95% CI
  下界大于 0；
- 至少 2/3 seeds 为正；
- `probe_then_migration` 和 `migration_with_image` 均改善；
- 任一 guardrail category 回退不超过 3 个百分点；
- 错误副作用动作均值不增加，critical violation 为 0；
- JSON-valid >= 99.5%，extra-text 和 typed-boolean failure 不增加；
- 三次确定性 repeat action agreement >= 99%。

任一硬门失败则 sealed 结果保持 escrow，不追加第四个 seed，也不在同一 hard-dev
上放宽阈值。下一实验臂必须新建版本化 train/dev 实体，不能继续消耗同一 dev。

## 10. 环境、资源和产物

### 10.1 环境隔离

- 本地 7B 训练只从给定绝对路径加载，不访问远端模型仓库。
- 本地 7B 服务显式使用独立 `MODEL/API_BASE`，避免被外部网关变量覆盖。
- 非 7B 外部调用统一在单独 shell 中 `source init_env.sh`。
- localhost 评测遵循 `NO_PROXY`；外部调用保留初始化脚本的代理配置。
- 任何脚本只记录环境变量名和非敏感 hash，不记录值。

### 10.2 资源

本机为 8 x Tesla V100-SXM2-32GB。训练前重新检查占用；预期 seed42 使用 4 卡，
通过后 seeds43/44 分别使用 4 卡并行。权重、adapter、raw completion 和 trace 写入
`/raid/zkq/artifacts/CAPA`，Git 只保存配置、manifest、hash、指标与有界审计报告。

### 10.3 计划新增资产

```text
data/datasets/planner_multistep_grpo_hard_v1/
  DATASET_CARD.md
  HUMAN_REVIEW.md
  manifest.json
  leakage_report.json
  grpo_support_report.json
training/planner_grpo_seed_v1/cases/
  planner_multistep_grpo_hard_v1_hard_dev_cases.jsonl
<sealed artifact root>/planner_multistep_grpo_hard_v1_sealed_test_cases.jsonl
configs/eval/planner_multistep_grpo_hard_v1_base_3x.json
configs/train/planner_multistep_grpo_hard_v1.json
experiments/studies/planner_multistep_tool_routing_grpo_qwen25_7b_v1/
  study.json
  preregistration.json
<artifact root>/evals/planner_multistep_grpo_hard_v1/base/
  hard_dev_3x/
  sealed_baseline_escrow_3x/
```

## 11. 停止条件

- 评测 prompt/schema 与历史 35B 无法对齐且没有重跑 35B；
- compound245 行数或 SHA 不符；
- 新数据发现实体、模板或 fixture 跨 split 泄漏；
- raw/SFT policy 没有正确动作采样支持或 reward 方差；
- 场景决策实际由 orchestrator 硬编码，或 gold 需要 LLM judge 才能判定；
- 增长只来自 alias、parser 宽松化或格式奖励；
- seed42 增加错误副作用动作；
- 多 seed 不复现；
- sealed test 被训练进程挂载，或 raw/逐 case/分层结果在 candidate 冻结前被读取；
- sealed base 结果被用于数据、reward、checkpoint 或训练超参选择。

当前已完成 compound245 两臂 3x、V1/V2 calibration、V2 family gate、600-case
confirmation 生成、GRPO support audit、base/35B confirmation 3x 和整版 gate。
训练数据 split 尚未生成，SFT/GRPO 均未启动。
